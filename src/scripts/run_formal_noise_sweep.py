#!/usr/bin/env python3
"""
Formal noise experiment sweep — three-model comparison under ranking corruption.

Trains all 3 models (nominal BT, nominal PL, robust PL) on 5 noise conditions
and evaluates each on the clean held-out K=4 set and ultrafeedback_binarized.

Grid
----
noise_type  : near_tie, top_rank
noise_level : 0.0, 0.4, 1.0  (0.0 shared across types → 5 unique conditions)
model       : nominal_bt, nominal_pl, robust_pl

Total runs: 5 conditions × 3 models = 15 training runs

Dataset caching: within the same noise_condition, all 3 models share the same
noisy dataset (same tokenised tensors), so the dataset is built once per
noise condition.

The held-out evaluation set is ALWAYS CLEAN so metrics reflect performance
under distribution shift (trained on noisy, tested on clean).

Results saved to: outputs/formal/noise_sweep_results.csv

Usage
-----
conda activate robust_listwise_llm
cd ~/work/robust_listwise_llm

# Full sweep (~15 × 10 min ≈ 2.5 hr)
CUDA_VISIBLE_DEVICES=1 python src/scripts/run_formal_noise_sweep.py

# Quick smoke test (20 steps, 50 eval)
CUDA_VISIBLE_DEVICES=1 python src/scripts/run_formal_noise_sweep.py --quick

# Single noise condition:
CUDA_VISIBLE_DEVICES=1 python src/scripts/run_formal_noise_sweep.py \\
    --noise_type top_rank --noise_level 1.0
"""

import argparse
import csv
import gc
import logging
import os
import sys
import time
from typing import Dict, List, Optional

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
from eval.ranking_metrics import build_held_out_k4, compute_ranking_metrics
from trainers.listwise_trainer import (
    BTListwiseTrainer,
    NominalListwiseTrainer,
    RobustListwiseTrainer,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"
K            = 4
N_TRAIN      = 1000
N_EVAL_K4    = 300
N_EVAL_BIN   = 300
MAX_STEPS    = 200
BATCH_SIZE   = 2
GRAD_ACCUM   = 4
LR           = 5e-5
BETA         = 0.1
MAX_LENGTH   = 512
MAX_PROMPT   = 256
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
SEED         = 42
RHO          = 0.1
DEVICE       = "cuda:0"
OUTPUT_DIR   = "outputs/formal"

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

# 5 noise conditions: noise_level=0 is shared (near_tie 0.0 == top_rank 0.0)
NOISE_CONDITIONS = [
    ("near_tie", 0.0),
    ("near_tie", 0.4),
    ("near_tie", 1.0),
    ("top_rank", 0.4),
    ("top_rank", 1.0),
]

MODEL_NAMES = ["nominal_bt", "nominal_pl", "robust_pl"]

# Global rho for current run (overwritten in main when --rho is passed)
_RHO_FOR_RUN = RHO


# ---------------------------------------------------------------------------
# Model builder / free
# ---------------------------------------------------------------------------

def _build_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb,
        device_map={"": DEVICE}, trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    return get_peft_model(model, lora, autocast_adapter_dtype=False)


def _free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Single (noise_condition, model) run
# ---------------------------------------------------------------------------

def run_one(
    noise_type: str,
    noise_level: float,
    model_name: str,
    tokenizer,
    dataset,            # pre-built noisy dataset (shared across models)
    held_out_k4,        # pre-built clean held-out examples
    max_steps: int,
    n_eval_bin: int,
) -> Dict:
    label = f"{noise_type}/lvl{noise_level:.1f}/{model_name}"
    logger.info("--- RUN: %s (%d steps) ---", label, max_steps)
    t0 = time.time()

    run_dir = os.path.join(
        OUTPUT_DIR, "noise_sweep",
        f"{noise_type}_lvl{noise_level:.1f}_{model_name}"
    )
    os.makedirs(run_dir, exist_ok=True)

    model   = _build_model()
    collator = ListwiseCollator(tokenizer=tokenizer, K=K)

    targs = TrainingArguments(
        output_dir=run_dir,
        max_steps=max_steps,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, max_steps // 10),
        bf16=True,
        logging_steps=max(1, max_steps // 5),
        save_steps=max_steps,
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        seed=SEED,
    )

    if model_name == "nominal_bt":
        trainer = BTListwiseTrainer(
            model=model, args=targs, train_dataset=dataset,
            data_collator=collator, beta=BETA, K=K,
        )
    elif model_name == "nominal_pl":
        trainer = NominalListwiseTrainer(
            model=model, args=targs, train_dataset=dataset,
            data_collator=collator, beta=BETA, K=K,
        )
    else:  # robust_pl or robust_pl_rho*
        trainer = RobustListwiseTrainer(
            model=model, args=targs, train_dataset=dataset,
            data_collator=collator, beta=BETA, K=K, rho=_RHO_FOR_RUN,
        )

    train_result = trainer.train()
    train_loss   = train_result.training_loss
    trainer.save_model(run_dir)

    # --- Eval A: K=4 ranking metrics (clean held-out) ---
    rank_metrics = compute_ranking_metrics(
        model=model, tokenizer=tokenizer,
        beta=BETA, n_eval=len(held_out_k4),
        max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT,
        device=DEVICE, seed=SEED,
        held_out_examples=held_out_k4,
    )

    # --- Eval B: pairwise accuracy on binarized test_prefs ---
    bin_metrics = pairwise_accuracy(
        model=model, tokenizer=tokenizer,
        beta=BETA, n_eval=n_eval_bin,
        max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT,
        device=DEVICE, seed=SEED,
    )

    del trainer
    _free(model)

    result = {
        "noise_type":    noise_type,
        "noise_level":   noise_level,
        "model":         model_name,
        "train_loss":    round(train_loss, 4),
        "top1_acc":      rank_metrics["top1_acc"],
        "exact_match":   rank_metrics["exact_match"],
        "kendall_tau":   rank_metrics["kendall_tau"],
        "ndcg":          rank_metrics["ndcg"],
        "pairwise_acc_k4":       rank_metrics["pairwise_acc_k4"],
        "pairwise_acc_binarized": round(bin_metrics["accuracy"], 4),
        "elapsed_s":     round(time.time() - t0, 1),
    }
    logger.info("Result: %s", result)
    return result


# ---------------------------------------------------------------------------
# CSV / table helpers
# ---------------------------------------------------------------------------

def _save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("Results saved → %s", path)


def _print_table(rows):
    if not rows:
        return
    hdr = (f"{'noise_type':<10} {'lvl':>5} {'model':<12} "
           f"{'top1':>6} {'tau':>7} {'ndcg':>6} {'pw_bin':>7}")
    sep = "─" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)
    prev_cond = None
    for r in rows:
        cond = (r["noise_type"], r["noise_level"])
        if prev_cond and cond != prev_cond:
            print()
        prev_cond = cond
        print(
            f"{r['noise_type']:<10} {r['noise_level']:>5.1f} {r['model']:<12} "
            f"{r['top1_acc']:>6.4f} {r['kendall_tau']:>7.4f} {r['ndcg']:>6.4f} "
            f"{r['pairwise_acc_binarized']:>7.4f}"
        )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick",        action="store_true",
                   help="20 steps, 100 train, 50 eval per run (smoke test)")
    p.add_argument("--noise_type",   default=None, choices=["near_tie", "top_rank"],
                   help="Run only this noise type")
    p.add_argument("--noise_level",  default=None, type=float,
                   help="Run only this noise level (requires --noise_type)")
    p.add_argument("--models",       nargs="+",
                   default=MODEL_NAMES,
                   help="Models to run (default: all three). Pass robust_pl_rho* for custom rho.")
    p.add_argument("--rho",          type=float, default=None,
                   help="Override RHO for RobustPL (default: 0.1). "
                        "Model name becomes robust_pl_rho{rho*1000:.0f}.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve rho
    rho_val = args.rho if args.rho is not None else RHO
    rho_tag = f"_rho{rho_val * 1000:03.0f}" if args.rho is not None else ""
    # When custom rho, rename robust_pl → robust_pl_rho<tag> in output dirs/CSV
    models_to_run = [
        f"robust_pl{rho_tag}" if m == "robust_pl" else m
        for m in args.models
    ]

    # Expose the resolved rho to run_one via a global-ish mechanism
    global _RHO_FOR_RUN
    _RHO_FOR_RUN = rho_val

    # --- Sanity check noise module ---
    verify_noise_functions()

    max_steps  = 20  if args.quick else MAX_STEPS
    n_train    = 100 if args.quick else N_TRAIN
    n_eval_k4  = 50  if args.quick else N_EVAL_K4
    n_eval_bin = 50  if args.quick else N_EVAL_BIN

    # --- Determine which noise conditions to run ---
    if args.noise_type is not None:
        level = args.noise_level if args.noise_level is not None else 0.0
        noise_conditions = [(args.noise_type, level)]
    else:
        noise_conditions = NOISE_CONDITIONS

    results_path = os.path.join(OUTPUT_DIR, "noise_sweep_results.csv")
    all_results: List[Dict] = []

    # --- Load tokenizer once ---
    logger.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Build clean held-out set once (shared across ALL runs) ---
    logger.info("Building clean held-out K=4 eval set (%d samples) ...", n_eval_k4)
    held_out_k4 = build_held_out_k4(
        tokenizer=tokenizer,
        n_train_skip=N_TRAIN if not args.quick else 100,
        n_eval=n_eval_k4,
        max_prompt_length=MAX_PROMPT,
        max_length=MAX_LENGTH,
        seed=SEED,
    )

    # --- Main sweep loop ---
    for noise_type, noise_level in noise_conditions:
        logger.info("=== Noise condition: %s / %.1f ===", noise_type, noise_level)

        # Build noisy dataset once per condition; reuse for all 3 models
        noise_fn = make_noise_fn(noise_type, noise_prob=noise_level, seed=SEED)
        logger.info("  Building noisy dataset (%d samples) ...", n_train)
        noisy_dataset = build_listwise_dataset(
            tokenizer=tokenizer,
            n_samples=n_train,
            max_prompt_length=MAX_PROMPT,
            max_length=MAX_LENGTH,
            seed=SEED,
            noise_fn=noise_fn,
        )

        for model_name in models_to_run:
            result = run_one(
                noise_type=noise_type,
                noise_level=noise_level,
                model_name=model_name,
                tokenizer=tokenizer,
                dataset=noisy_dataset,
                held_out_k4=held_out_k4,
                max_steps=max_steps,
                n_eval_bin=n_eval_bin,
            )
            all_results.append(result)
            _save_csv(all_results, results_path)

        # Free the dataset after all 3 models are done with this condition
        del noisy_dataset
        gc.collect()

    _print_table(all_results)
    logger.info("Noise sweep complete.  Results: %s", results_path)


if __name__ == "__main__":
    main()
