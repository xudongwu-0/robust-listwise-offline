#!/usr/bin/env python3
"""
Stage D — Controlled noise experiment sweep.

For each combination of (noise_type, noise_level, method):
  1. Build the UltraFeedback K=4 dataset with injected ranking noise.
  2. Train Qwen2.5-0.5B with the appropriate objective (nominal / robust).
  3. Evaluate pairwise accuracy on clean ultrafeedback_binarized test_prefs.
  4. Record results to outputs/noise_sweep/results.csv and print a summary table.

Grid:
  noise_type  : near_tie | top_rank
  noise_level : 0.0 | 0.4 | 1.0
  method      : nominal (rho=0) | robust (rho=0.1)

Expected hypothesis: robust outperforms nominal at high noise levels.

Usage:
  conda activate robust_listwise_llm
  cd ~/work/robust_listwise_llm

  # Full sweep (10 runs × ~90s ≈ 15 min on a single RTX 4090)
  CUDA_VISIBLE_DEVICES=1 python src/scripts/run_noise_sweep.py

  # Quick smoke test (fewer steps + fewer eval examples)
  CUDA_VISIBLE_DEVICES=1 python src/scripts/run_noise_sweep.py --quick

  # Single config for debugging
  CUDA_VISIBLE_DEVICES=1 python src/scripts/run_noise_sweep.py \\
      --noise_type near_tie --noise_level 0.4 --method robust

Output: outputs/noise_sweep/results.csv
"""

import argparse
import csv
import gc
import logging
import os
import sys
import time
from typing import Dict, Optional

_SRC = os.path.join(os.path.dirname(__file__), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from data.noise import make_noise_fn, verify_noise_functions
from data.ultrafeedback_listwise import ListwiseCollator, build_listwise_dataset
from eval.pairwise_accuracy import pairwise_accuracy
from trainers.listwise_trainer import RobustListwiseTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

MODEL_NAME     = "Qwen/Qwen2.5-0.5B-Instruct"
K              = 4
N_TRAIN        = 1000    # training samples per run
N_EVAL         = 200     # eval examples (ultrafeedback_binarized test_prefs)
MAX_STEPS      = 50      # reduced for sweep; increase for publication experiments
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
LR             = 5e-5
BETA           = 0.1
MAX_LENGTH     = 512
MAX_PROMPT     = 256
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
SEED           = 42
DEVICE         = "cuda:0"   # overridden by CUDA_VISIBLE_DEVICES

# Sweep configs: (noise_type, noise_level, method, rho)
# noise_level=0.0 gives the clean baseline (same regardless of noise_type)
SWEEP_CONFIGS = [
    # Clean baseline
    ("near_tie", 0.0, "nominal", 0.0),
    ("near_tie", 0.0, "robust",  0.1),
    # near_tie noise
    ("near_tie", 0.4, "nominal", 0.0),
    ("near_tie", 0.4, "robust",  0.1),
    ("near_tie", 1.0, "nominal", 0.0),
    ("near_tie", 1.0, "robust",  0.1),
    # top_rank noise
    ("top_rank", 0.4, "nominal", 0.0),
    ("top_rank", 0.4, "robust",  0.1),
    ("top_rank", 1.0, "nominal", 0.0),
    ("top_rank", 1.0, "robust",  0.1),
]

OUTPUT_DIR = "outputs/noise_sweep"


# ---------------------------------------------------------------------------
# Build / free model helpers
# ---------------------------------------------------------------------------

def _build_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": DEVICE},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg, autocast_adapter_dtype=False)
    return model, tokenizer


def _free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(
    noise_type: str,
    noise_level: float,
    method: str,
    rho: float,
    run_dir: str,
    max_steps: int,
    n_eval: int,
    tokenizer=None,   # pre-loaded tokenizer (reused across runs for speed)
    dataset_cache=None,  # (noise_type, noise_level) -> dataset cache
) -> Dict:
    """
    Train one model configuration and evaluate it.

    Returns a dict with training and evaluation metrics.
    """
    logger.info(
        "=== RUN: noise=%s/%.1f  method=%s(rho=%.2f) ===",
        noise_type, noise_level, method, rho
    )
    t0 = time.time()

    # ---- Dataset ----
    cache_key = (noise_type, noise_level)
    if dataset_cache is not None and cache_key in dataset_cache:
        dataset = dataset_cache[cache_key]
        logger.info("Dataset loaded from cache.")
    else:
        noise_fn = make_noise_fn(noise_type, noise_prob=noise_level, seed=SEED)
        dataset = build_listwise_dataset(
            tokenizer=tokenizer,
            n_samples=N_TRAIN,
            max_prompt_length=MAX_PROMPT,
            max_length=MAX_LENGTH,
            seed=SEED,
            noise_fn=noise_fn,
        )
        if dataset_cache is not None:
            dataset_cache[cache_key] = dataset

    # ---- Model ----
    model, _ = _build_model_and_tokenizer()

    # ---- Collator ----
    collator = ListwiseCollator(tokenizer=tokenizer, K=K)

    # ---- Training args ----
    os.makedirs(run_dir, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=run_dir,
        max_steps=max_steps,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, max_steps // 10),
        bf16=True,
        logging_steps=max(1, max_steps // 5),
        save_steps=max_steps,      # only save at end
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        seed=SEED,
    )

    # ---- Trainer ----
    trainer = RobustListwiseTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator,
        beta=BETA,
        K=K,
        rho=rho,
    )

    train_result = trainer.train()
    train_loss   = train_result.training_loss

    # ---- Evaluate ----
    eval_metrics = pairwise_accuracy(
        model=model,
        tokenizer=tokenizer,
        beta=BETA,
        n_eval=n_eval,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT,
        device=DEVICE,
        seed=SEED,
    )

    # ---- Save checkpoint ----
    trainer.save_model(run_dir)

    # ---- Clean up ----
    del trainer
    _free_model(model)

    elapsed = time.time() - t0
    result = {
        "noise_type":  noise_type,
        "noise_level": noise_level,
        "method":      method,
        "rho":         rho,
        "train_loss":  round(train_loss, 4),
        "accuracy":    round(eval_metrics["accuracy"], 4),
        "mean_margin": round(eval_metrics["mean_margin"], 4),
        "n_eval":      eval_metrics["n_eval"],
        "elapsed_s":   round(elapsed, 1),
    }
    logger.info("Result: %s", result)
    return result


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def _save_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    logger.info("Results saved to %s", path)


def _print_table(results):
    if not results:
        return
    header = f"{'noise_type':<10} {'noise_lvl':>9} {'method':<8} {'rho':>5} │ {'loss':>7} {'acc':>7} {'margin':>8}"
    sep    = "─" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['noise_type']:<10} {r['noise_level']:>9.1f} {r['method']:<8} {r['rho']:>5.2f} │"
            f" {r['train_loss']:>7.4f} {r['accuracy']:>7.4f} {r['mean_margin']:>8.4f}"
        )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="Use 20 steps and 100 eval examples (fast smoke test)")
    p.add_argument("--noise_type",  default=None, choices=["near_tie", "top_rank"],
                   help="Run only this noise type (single run mode)")
    p.add_argument("--noise_level", default=None, type=float,
                   help="Run only this noise level (single run mode)")
    p.add_argument("--method",      default=None, choices=["nominal", "robust"],
                   help="Run only this method (single run mode)")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Sanity check noise module ---
    verify_noise_functions()

    max_steps = 20  if args.quick else MAX_STEPS
    n_eval    = 100 if args.quick else N_EVAL

    # --- Determine configs to run ---
    if args.noise_type is not None:
        rho = 0.0 if args.method == "nominal" else 0.1
        configs = [(args.noise_type, args.noise_level, args.method, rho)]
    else:
        configs = SWEEP_CONFIGS

    # --- Load tokenizer once (shared across all runs) ---
    logger.info("Loading tokenizer once ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Cache datasets by (noise_type, noise_level) ---
    dataset_cache = {}

    results_path = os.path.join(OUTPUT_DIR, "results.csv")
    all_results  = []

    for cfg in configs:
        noise_type, noise_level, method, rho = cfg
        run_dir = os.path.join(
            OUTPUT_DIR,
            f"{noise_type}_lvl{noise_level:.1f}_{method}"
        )
        result = run_one(
            noise_type=noise_type,
            noise_level=noise_level,
            method=method,
            rho=rho,
            run_dir=run_dir,
            max_steps=max_steps,
            n_eval=n_eval,
            tokenizer=tokenizer,
            dataset_cache=dataset_cache,
        )
        all_results.append(result)
        # Save incrementally so partial results are preserved
        _save_csv(all_results, results_path)

    _print_table(all_results)
    logger.info("Sweep complete. Results: %s", results_path)


if __name__ == "__main__":
    main()
