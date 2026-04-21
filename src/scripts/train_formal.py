#!/usr/bin/env python3
"""
Formal training experiment — three-model comparison on clean data.

Trains all three main models sequentially on the same 5000-sample
openbmb/UltraFeedback split, then evaluates each on the full metric suite.

Models
------
1. nominal_bt   — Bradley-Terry pairwise DPO (BTListwiseTrainer, top vs bottom)
2. nominal_pl   — Nominal Plackett-Luce listwise DPO
3. robust_pl    — Robust Plackett-Luce (ρ=0.1)

All trained on the identical K=4 batches so comparisons are directly fair.

Evaluation (after each training run)
--------------------------------------
A. K=4 ranking metrics on 500 held-out UltraFeedback examples:
     top1_acc, exact_match, kendall_tau, ndcg, pairwise_acc_k4

B. Pairwise accuracy on 500 ultrafeedback_binarized test_prefs:
     pairwise_acc_binarized

C. RewardBench (allenai/reward-bench):
     overall, chat, chat_hard, safety, reasoning

Results saved to: outputs/formal/formal_clean_results.csv

Usage
-----
conda activate robust_listwise_llm
cd ~/work/robust_listwise_llm
CUDA_VISIBLE_DEVICES=1 python src/scripts/train_formal.py

# Skip RewardBench (faster, no network needed):
CUDA_VISIBLE_DEVICES=1 python src/scripts/train_formal.py --no_rewardbench

# Quick test (100 steps, 500 samples, 50 eval):
CUDA_VISIBLE_DEVICES=1 python src/scripts/train_formal.py --quick
"""

import argparse
import csv
import gc
import logging
import os
import sys
import time

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

from data.ultrafeedback_listwise import ListwiseCollator, build_listwise_dataset
from eval.pairwise_accuracy import pairwise_accuracy
from eval.ranking_metrics import build_held_out_k4, compute_ranking_metrics
from eval.rewardbench_eval import rewardbench_eval
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
# Hyperparameters
# ---------------------------------------------------------------------------
MODEL_NAME    = "Qwen/Qwen2.5-0.5B-Instruct"
K             = 4
N_TRAIN       = 5000
N_EVAL_K4     = 500     # held-out K=4 UltraFeedback examples
N_EVAL_BIN    = 500     # ultrafeedback_binarized test_prefs
MAX_STEPS     = 1000
BATCH_SIZE    = 2
GRAD_ACCUM    = 4       # effective batch = 8
LR            = 5e-5
BETA          = 0.1
MAX_LENGTH    = 512
MAX_PROMPT    = 256
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
SEED          = 42
RHO           = 0.1     # robust PL regularisation strength
DEVICE        = "cuda:0"
OUTPUT_DIR    = "outputs/formal"


# ---------------------------------------------------------------------------
# Model builder / free
# ---------------------------------------------------------------------------

def _build_model(tokenizer):
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
    return get_peft_model(model, lora_cfg, autocast_adapter_dtype=False)


def _free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Training argument factory
# ---------------------------------------------------------------------------

def _train_args(run_dir: str, max_steps: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=run_dir,
        max_steps=max_steps,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, max_steps // 10),
        bf16=True,
        logging_steps=max(1, max_steps // 20),
        save_steps=max(1, max_steps // 2),
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        seed=SEED,
    )


# ---------------------------------------------------------------------------
# Single model run
# ---------------------------------------------------------------------------

def run_model(
    model_name: str,        # "nominal_bt" | "nominal_pl" | "robust_pl"
    tokenizer,
    dataset,
    held_out_k4,
    max_steps: int,
    n_eval_bin: int,
    no_rewardbench: bool,
) -> dict:
    logger.info("=" * 60)
    logger.info("TRAINING: %s  (%d steps)", model_name, max_steps)
    logger.info("=" * 60)
    t0 = time.time()

    run_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(run_dir, exist_ok=True)

    # Build model fresh each run
    model = _build_model(tokenizer)

    # Choose trainer
    collator = ListwiseCollator(tokenizer=tokenizer, K=K)
    targs    = _train_args(run_dir, max_steps)

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
    elif model_name == "robust_pl":
        trainer = RobustListwiseTrainer(
            model=model, args=targs, train_dataset=dataset,
            data_collator=collator, beta=BETA, K=K, rho=RHO,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    train_result = trainer.train()
    train_loss   = train_result.training_loss
    trainer.save_model(run_dir)

    # ---- Evaluation ----
    # A: K=4 ranking metrics on held-out UltraFeedback
    rank_metrics = compute_ranking_metrics(
        model=model, tokenizer=tokenizer,
        beta=BETA, n_eval=len(held_out_k4),
        max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT,
        device=DEVICE, seed=SEED,
        held_out_examples=held_out_k4,
    )

    # B: pairwise accuracy on ultrafeedback_binarized test_prefs
    bin_metrics = pairwise_accuracy(
        model=model, tokenizer=tokenizer,
        beta=BETA, n_eval=n_eval_bin,
        max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT,
        device=DEVICE, seed=SEED,
    )

    # C: RewardBench
    rb_metrics = {}
    if not no_rewardbench:
        rb_metrics = rewardbench_eval(
            model=model, tokenizer=tokenizer,
            beta=BETA, n_eval=None,
            max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT,
            device=DEVICE, seed=SEED,
        )

    del trainer
    _free(model)

    elapsed = round(time.time() - t0, 1)
    result = {
        "model":              model_name,
        "train_loss":         round(train_loss, 4),
        "top1_acc":           rank_metrics["top1_acc"],
        "exact_match":        rank_metrics["exact_match"],
        "kendall_tau":        rank_metrics["kendall_tau"],
        "ndcg":               rank_metrics["ndcg"],
        "pairwise_acc_k4":    rank_metrics["pairwise_acc_k4"],
        "pairwise_acc_binarized": round(bin_metrics["accuracy"], 4),
        "rb_overall":         rb_metrics.get("overall", float("nan")),
        "rb_chat":            rb_metrics.get("chat",    float("nan")),
        "rb_chat_hard":       rb_metrics.get("chat_hard", float("nan")),
        "rb_safety":          rb_metrics.get("safety",  float("nan")),
        "rb_reasoning":       rb_metrics.get("reasoning", float("nan")),
        "elapsed_s":          elapsed,
    }
    logger.info("Result: %s", result)
    return result


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved: %s", path)


def _print_table(rows):
    if not rows:
        return
    hdr = f"{'model':<12} {'loss':>7} {'top1':>6} {'exact':>6} {'tau':>7} {'ndcg':>6} {'pw_k4':>7} {'pw_bin':>7} {'rb':>6}"
    sep = "─" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)
    for r in rows:
        rb = r.get("rb_overall", float("nan"))
        rb_str = f"{rb:.4f}" if rb == rb else "  N/A "   # NaN check
        print(
            f"{r['model']:<12} {r['train_loss']:>7.4f} {r['top1_acc']:>6.4f}"
            f" {r['exact_match']:>6.4f} {r['kendall_tau']:>7.4f} {r['ndcg']:>6.4f}"
            f" {r['pairwise_acc_k4']:>7.4f} {r['pairwise_acc_binarized']:>7.4f} {rb_str:>6}"
        )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick",          action="store_true",
                   help="100 steps / 500 train samples / 50 eval (smoke test)")
    p.add_argument("--no_rewardbench", action="store_true",
                   help="Skip RewardBench evaluation")
    p.add_argument("--models",         nargs="+",
                   default=["nominal_bt", "nominal_pl", "robust_pl"],
                   choices=["nominal_bt", "nominal_pl", "robust_pl"],
                   help="Which models to train (default: all three)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    max_steps  = 100  if args.quick else MAX_STEPS
    n_train    = 500  if args.quick else N_TRAIN
    n_eval_k4  = 50   if args.quick else N_EVAL_K4
    n_eval_bin = 50   if args.quick else N_EVAL_BIN

    results_path = os.path.join(OUTPUT_DIR, "formal_clean_results.csv")
    all_results  = []

    # ---- Load tokenizer once ----
    logger.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Build shared training dataset ----
    logger.info("Building training dataset (%d samples, clean) ...", n_train)
    dataset = build_listwise_dataset(
        tokenizer=tokenizer,
        n_samples=n_train,
        max_prompt_length=MAX_PROMPT,
        max_length=MAX_LENGTH,
        seed=SEED,
        noise_fn=None,
    )

    # ---- Build shared held-out K=4 eval set ----
    logger.info("Building held-out K=4 eval set (%d samples) ...", n_eval_k4)
    held_out_k4 = build_held_out_k4(
        tokenizer=tokenizer,
        n_train_skip=N_TRAIN if not args.quick else 500,
        n_eval=n_eval_k4,
        max_prompt_length=MAX_PROMPT,
        max_length=MAX_LENGTH,
        seed=SEED,
    )

    # ---- Train each model in sequence ----
    for model_name in args.models:
        result = run_model(
            model_name=model_name,
            tokenizer=tokenizer,
            dataset=dataset,
            held_out_k4=held_out_k4,
            max_steps=max_steps,
            n_eval_bin=n_eval_bin,
            no_rewardbench=args.no_rewardbench,
        )
        all_results.append(result)
        _save_csv(all_results, results_path)

    _print_table(all_results)
    logger.info("Formal training complete.  Results: %s", results_path)


if __name__ == "__main__":
    main()
