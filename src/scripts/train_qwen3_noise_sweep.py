#!/usr/bin/env python3
"""
Qwen3 unified noise sweep — runs one (noise_type, noise_level) condition
over multiple models for either Qwen3-0.6B or Qwen3-8B.

All conditions use 1000 training samples / 200 steps for a fast,
matched-budget comparison. Output is appended to a single CSV per model size
so multiple invocations accumulate cleanly.

Usage
-----
conda activate robust_listwise_llm
cd ~/work/robust_listwise_llm

# 0.6B, near_tie at 0.4 over all 4 models
CUDA_VISIBLE_DEVICES=4 python src/scripts/train_qwen3_noise_sweep.py \
    --model_size 05b --noise_type near_tie --noise_level 0.4

# 8B, top_rank at 1.0
CUDA_VISIBLE_DEVICES=5 python src/scripts/train_qwen3_noise_sweep.py \
    --model_size 8b --noise_type top_rank --noise_level 1.0

# Quick smoke test
CUDA_VISIBLE_DEVICES=4 python src/scripts/train_qwen3_noise_sweep.py \
    --model_size 05b --noise_type near_tie --noise_level 0.4 --quick
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

from data.noise import make_noise_fn
from data.ultrafeedback_listwise import ListwiseCollator, build_listwise_dataset, build_listwise_dataset_k8
from eval.pairwise_accuracy import pairwise_accuracy
from eval.ranking_metrics import build_held_out_k4, compute_ranking_metrics
from trainers.listwise_trainer import (
    BTListwiseTrainer,
    NominalListwiseTrainer,
    RobustListwiseTrainer,
)
from trainers.dr_dpo_trainer import DRDPOTrainer, KLDPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Per-model-size config
# --------------------------------------------------------------------------
CONFIG = {
    "05b": dict(
        model_name="Qwen/Qwen3-0.6B",
        lora_r=16,
        lora_alpha=32,
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
        batch_size=2,
        grad_accum=4,
        lr=5e-5,
        output_dir="outputs/qwen3/05b/noise_sweep",
    ),
    "8b": dict(
        model_name="Qwen/Qwen3-8B",
        lora_r=32,
        lora_alpha=64,
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
        batch_size=1,
        grad_accum=4,
        lr=3e-5,
        output_dir="outputs/qwen3/8b/noise_sweep",
    ),
}

# Shared constants (matched-budget design)
K            = 4
N_TRAIN      = 1000
N_EVAL_K4    = 300
N_EVAL_BIN   = 300
MAX_STEPS    = 200
BETA         = 0.1
MAX_LENGTH   = 512
MAX_PROMPT   = 256
LORA_DROPOUT = 0.05
SEED         = 42
RHO_DEFAULT  = 0.1
RHO_050      = 0.05
RHO_015      = 0.15
M_DRO        = 5     # cross-microbatch DRO group size for tv_dr_dpo / kldpo

# TV-DR-DPO radii swept; each becomes its own model name registered below.
TV_DR_DPO_RHOS = {
    "tv_dr_dpo_rho005": 0.05,
    "tv_dr_dpo_rho010": 0.10,
    "tv_dr_dpo_rho020": 0.20,
    "tv_dr_dpo_rho040": 0.40,
    "tv_dr_dpo_rho080": 0.80,
}

# KLDPO temperatures swept; the dev-promoted value is also exposed as its
# own model name for the main table.
KLDPO_TAUS   = {
    "kldpo_tau005":  0.05,
    "kldpo_tau010":  0.10,
    "kldpo_tau020":  0.20,
    "kldpo_tau050":  0.50,
    "kldpo_tau100":  1.00,
}

# Robust PL clean-sweep grid (rho > 0.15).
# Naming: suffix = left-zero-padded 3-char representation of rho × 100.
# Exceptions: rho50 (2-char) avoids collision with rho050 (= 0.05).
# rho=0.10 -> "robust_pl" (legacy), rho=0.05 -> "robust_pl_rho050" (legacy),
# rho=0.15 -> "robust_pl_rho015" (legacy); those remain unchanged.
ROBUST_PL_SWEEP_RHOS = {
    "robust_pl_rho020": 0.20,
    "robust_pl_rho030": 0.30,
    "robust_pl_rho50":  0.50,   # 2-char suffix to avoid collision with rho050=0.05
    "robust_pl_rho070": 0.70,
    "robust_pl_rho100": 1.00,
}

ALL_MODELS = [
    "nominal_bt", "nominal_pl",
    "robust_pl", "robust_pl_rho050", "robust_pl_rho015",
] + list(ROBUST_PL_SWEEP_RHOS.keys()) \
  + list(TV_DR_DPO_RHOS.keys()) + list(KLDPO_TAUS.keys())


# --------------------------------------------------------------------------
# Model builder
# --------------------------------------------------------------------------

def _build_model(cfg: dict):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    n_gpu = torch.cuda.device_count()
    device_map = "auto" if n_gpu >= 2 else {"": "cuda:0"}

    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb,
        device_map=device_map,
        trust_remote_code=True,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=LORA_DROPOUT,
        target_modules=cfg["lora_targets"],
        bias="none",
    )
    model = get_peft_model(base, lora_cfg, autocast_adapter_dtype=False)
    return model


def _free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Trainer factory
# --------------------------------------------------------------------------

def _make_trainer(model_name: str, model, targs, dataset, collator, list_size: int = 4):
    common = dict(
        model=model, args=targs, train_dataset=dataset,
        data_collator=collator, beta=BETA, K=list_size,
    )
    if model_name == "nominal_bt":
        return BTListwiseTrainer(**common)
    elif model_name == "nominal_pl":
        return NominalListwiseTrainer(**common)
    elif model_name == "robust_pl":
        return RobustListwiseTrainer(**common, rho=RHO_DEFAULT)
    elif model_name == "robust_pl_rho050":
        return RobustListwiseTrainer(**common, rho=RHO_050)
    elif model_name == "robust_pl_rho015":
        return RobustListwiseTrainer(**common, rho=RHO_015)
    elif model_name in ROBUST_PL_SWEEP_RHOS:
        return RobustListwiseTrainer(**common, rho=ROBUST_PL_SWEEP_RHOS[model_name])
    elif model_name in TV_DR_DPO_RHOS:
        return DRDPOTrainer(**common, rho=TV_DR_DPO_RHOS[model_name], m_dro=M_DRO)
    elif model_name in KLDPO_TAUS:
        return KLDPOTrainer(**common, tau=KLDPO_TAUS[model_name], m_dro=M_DRO)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def _train_args(run_dir: str, max_steps: int, cfg: dict,
                model_name: str = "") -> TrainingArguments:
    # tv_dr_dpo* / kldpo* require per_device_batch=1 and grad_accum=m_dro so
    # that the cross-microbatch DRO buffer aligns with HF Trainer's optimizer
    # cadence. See docs/tv_dr_dpo_baseline.md and docs/kldpo_baseline.md for
    # the comparability discussion.
    if model_name.startswith("tv_dr_dpo") or model_name.startswith("kldpo"):
        per_dev_bs = 1
        grad_accum = M_DRO
    else:
        per_dev_bs = cfg["batch_size"]
        grad_accum = cfg["grad_accum"]
    return TrainingArguments(
        output_dir=run_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_dev_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=max(1, max_steps // 10),
        bf16=True,
        logging_steps=max(1, max_steps // 20),
        save_steps=max_steps,        # save once at end
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        seed=SEED,
    )


# --------------------------------------------------------------------------
# Single run
# --------------------------------------------------------------------------

def run_one(
    model_name: str,
    cond_slug: str,
    cfg: dict,
    out_dir: str,
    tokenizer,
    dataset,
    held_out_k4,
    max_steps: int,
    n_eval_bin: int,
    eval_device: str,
    list_size: int = 4,
) -> dict:
    run_dir = os.path.join(out_dir, cond_slug, model_name)
    os.makedirs(run_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("RUN  %s / %s  (%d steps, K=%d)", cond_slug, model_name, max_steps, list_size)
    logger.info("=" * 70)
    t0 = time.time()

    model    = _build_model(cfg)
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_train:,}")

    collator = ListwiseCollator(tokenizer=tokenizer, K=list_size)
    targs    = _train_args(run_dir, max_steps, cfg, model_name=model_name)
    trainer  = _make_trainer(model_name, model, targs, dataset, collator, list_size=list_size)

    train_result = trainer.train()
    train_loss   = train_result.training_loss
    trainer.save_model(run_dir)
    del trainer

    rank_metrics = compute_ranking_metrics(
        model=model, tokenizer=tokenizer, beta=BETA,
        n_eval=len(held_out_k4), max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT, device=eval_device, seed=SEED,
        held_out_examples=held_out_k4,
        eval_batch_size=1 if cfg["model_name"].endswith("8B") else 4,
    )
    bin_metrics = pairwise_accuracy(
        model=model, tokenizer=tokenizer, beta=BETA, n_eval=n_eval_bin,
        max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT,
        device=eval_device, seed=SEED,
        eval_batch_size=1 if cfg["model_name"].endswith("8B") else 4,
    )
    _free(model)

    elapsed = round(time.time() - t0, 1)
    return {
        "condition":              cond_slug,
        "model":                  model_name,
        "base_model":             cfg["model_name"],
        "train_loss":             round(train_loss, 4),
        "top1_acc":               rank_metrics["top1_acc"],
        "exact_match":            rank_metrics["exact_match"],
        "kendall_tau":            rank_metrics["kendall_tau"],
        "ndcg":                   rank_metrics["ndcg"],
        "pairwise_acc_k4":        rank_metrics["pairwise_acc_k4"],
        "pairwise_acc_binarized": round(bin_metrics["accuracy"], 4),
        "elapsed_s":              elapsed,
    }


def _append_csv(row: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    logger.info("Saved row to: %s", path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_size", required=True, choices=["05b", "8b"])
    p.add_argument("--noise_type", required=True,
                   choices=["clean", "near_tie", "top_rank"])
    p.add_argument("--noise_level", type=float, default=None,
                   help="Required unless --noise_type=clean")
    p.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument("--steps", type=int, default=None,
                   help="Override MAX_STEPS (default: 200)")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Override output directory (default: from CONFIG)")
    p.add_argument("--list_size", type=int, default=4, choices=[4, 8],
                   help="Candidate list size K for training (default: 4). "
                        "K=8 pairs consecutive UltraFeedback examples.")
    p.add_argument("--n_train", type=int, default=None,
                   help="Override number of training samples (default: 1000). "
                        "Also shifts the eval held-out window to avoid overlap.")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 20 steps, 100 samples, 30 eval")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = CONFIG[args.model_size]

    n_train    = 100 if args.quick else (args.n_train if args.n_train else N_TRAIN)
    max_steps  = 20  if args.quick else (args.steps if args.steps else MAX_STEPS)
    n_eval_k4  = 30  if args.quick else N_EVAL_K4
    n_eval_bin = 30  if args.quick else N_EVAL_BIN
    list_size  = args.list_size   # 4 (default) or 8

    out_dir = args.out_dir if args.out_dir else cfg["output_dir"]

    if args.noise_type == "clean":
        cond_slug = "clean"
        noise_fn  = None
    else:
        if args.noise_level is None:
            raise SystemExit("--noise_level is required for non-clean conditions")
        cond_slug = f"{args.noise_type}_lvl{args.noise_level:.1f}".replace(".", "")
        noise_fn  = make_noise_fn(args.noise_type,
                                  noise_prob=args.noise_level, seed=SEED)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    if args.list_size == 8:
        csv_suffix = "k8_sweep"
    elif args.n_train and args.n_train != N_TRAIN:
        csv_suffix = f"k4_{args.n_train}n_ablation"
    elif args.out_dir:
        csv_suffix = "long3x"
    else:
        csv_suffix = "noise_sweep"
    csv_path = os.path.join(out_dir, f"qwen3_{args.model_size}_{csv_suffix}.csv")

    logger.info("Qwen3-%s noise sweep", args.model_size.upper())
    logger.info("  base_model = %s", cfg["model_name"])
    logger.info("  condition  = %s  (noise=%s, level=%s)",
                cond_slug, args.noise_type, args.noise_level)
    logger.info("  models     = %s", args.models)
    logger.info("  steps      = %d  n_train = %d", max_steps, n_train)
    logger.info("  csv        = %s", csv_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Building training dataset (K=%d, %d samples) ...", list_size, n_train)
    if list_size == 8:
        dataset = build_listwise_dataset_k8(
            tokenizer=tokenizer, n_samples=n_train,
            max_prompt_length=MAX_PROMPT, max_length=MAX_LENGTH,
            seed=SEED, noise_fn=noise_fn,
        )
    else:
        dataset = build_listwise_dataset(
            tokenizer=tokenizer, n_samples=n_train,
            max_prompt_length=MAX_PROMPT, max_length=MAX_LENGTH,
            seed=SEED, noise_fn=noise_fn,
        )

    logger.info("Building held-out K=4 eval set (%d samples) ...", n_eval_k4)
    held_out_k4 = build_held_out_k4(
        tokenizer=tokenizer, n_train_skip=n_train, n_eval=n_eval_k4,
        max_prompt_length=MAX_PROMPT, max_length=MAX_LENGTH, seed=SEED,
    )

    eval_device = "cuda:0"
    for i, model_name in enumerate(args.models):
        logger.info("[%d/%d] %s ...", i + 1, len(args.models), model_name)
        result = run_one(
            model_name=model_name, cond_slug=cond_slug, cfg=cfg,
            out_dir=out_dir,
            tokenizer=tokenizer, dataset=dataset, held_out_k4=held_out_k4,
            max_steps=max_steps, n_eval_bin=n_eval_bin, eval_device=eval_device,
            list_size=list_size,
        )
        _append_csv(result, csv_path)
        logger.info("[%d/%d] done %s/%s  τ=%.4f  top1=%.4f",
                    i + 1, len(args.models), cond_slug, model_name,
                    result["kendall_tau"], result["top1_acc"])

    logger.info("Sweep complete. Results: %s", csv_path)


if __name__ == "__main__":
    main()
