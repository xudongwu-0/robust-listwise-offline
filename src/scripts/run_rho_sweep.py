#!/usr/bin/env python3
"""
ρ-sweep experiment on Qwen2.5-0.5B-Instruct.

Trains RobustPL at different ρ values under a fixed noise condition and
evaluates each on the clean held-out K=4 set.  Goal: find the optimal ρ*
and observe the robustness–performance trade-off curve (τ vs. ρ).

Mathematical background
-----------------------
    ell_robust = (1-ρ) * ell_PL(σ_obs) + ρ * ell_PL(σ_wc)

where σ_wc is the worst-case ranking (ascending model score order).
Setting ρ=0 recovers Nominal PL exactly; ρ=1 trains entirely on σ_wc.

Default settings
----------------
  noise_type  : top_rank
  noise_level : 0.4   (40% of rank-1 replaced with a random low-quality response)
  ρ values    : [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
  budget      : 1000 samples / 200 steps  (matched to noise-sweep experiments)
  References  : nominal_pl + nominal_bt trained on same noise condition
                Note: ρ=0.0 in the sweep is mathematically identical to nominal_pl
                (loss = (1-0)*PL + 0*worst_case = PL). The explicit nominal_pl
                reference serves as a sanity-check and clear baseline label.

Output
------
  outputs/formal/rho_sweep_results.csv
  outputs/formal/plots/rho_sweep_tau.pdf  (requires matplotlib)
  outputs/formal/plots/rho_sweep_tau.png

Usage
-----
  # Full sweep (≈9 × 7 min ≈ 1 hr)
  CUDA_VISIBLE_DEVICES=2 python src/scripts/run_rho_sweep.py

  # Quick smoke test (20 steps, 50 eval)
  CUDA_VISIBLE_DEVICES=2 python src/scripts/run_rho_sweep.py --quick

  # Custom noise condition
  CUDA_VISIBLE_DEVICES=2 python src/scripts/run_rho_sweep.py \\
      --noise_type top_rank --noise_level 0.4

  # Custom ρ values
  CUDA_VISIBLE_DEVICES=2 python src/scripts/run_rho_sweep.py \\
      --rho_values 0.0 0.1 0.2 0.5 1.0

  # Skip all reference runs (nominal_pl + nominal_bt)
  CUDA_VISIBLE_DEVICES=2 python src/scripts/run_rho_sweep.py --no_ref

  # Skip only nominal_bt (keep nominal_pl reference)
  CUDA_VISIBLE_DEVICES=2 python src/scripts/run_rho_sweep.py --no_ref_bt

RHO_SWEEP_DONE is printed when all runs finish (used as shell sentinel).
"""

import argparse
import csv
import gc
import logging
import math
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

from data.noise import make_noise_fn
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
# Hyper-parameters (matched to noise-sweep budget)
# ---------------------------------------------------------------------------
MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"
K            = 4
N_TRAIN      = 1_000
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
DEVICE       = "cuda:0"
OUTPUT_DIR   = "outputs/formal"

# Default ρ sweep values — dense near 0 where the knee is expected
DEFAULT_RHO_VALUES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

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
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    return get_peft_model(model, lora, autocast_adapter_dtype=False)


def _free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(
    model_name: str,          # "robust_pl" or "nominal_bt"
    rho: float,               # ρ value; NaN for nominal_bt
    noise_type: str,
    noise_level: float,
    tokenizer,
    dataset,                  # pre-built noisy training dataset
    held_out_k4,              # pre-built clean held-out examples
    max_steps: int,
    n_eval_bin: int,
) -> Dict:
    rho_str = "ref" if math.isnan(rho) else f"{rho:.2f}"
    label = f"{model_name}/rho={rho_str} ({noise_type}/{noise_level:.1f})"
    logger.info("--- RUN: %s  max_steps=%d ---", label, max_steps)
    t0 = time.time()

    run_dir = os.path.join(
        OUTPUT_DIR, "rho_sweep",
        f"{noise_type}_lvl{noise_level:.1f}_{model_name}_rho{rho_str}",
    )
    os.makedirs(run_dir, exist_ok=True)

    model = _build_model()
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
    else:  # robust_pl at given rho
        trainer = RobustListwiseTrainer(
            model=model, args=targs, train_dataset=dataset,
            data_collator=collator, beta=BETA, K=K, rho=rho,
        )

    train_result = trainer.train()
    train_loss = train_result.training_loss
    trainer.save_model(run_dir)

    # Eval A: K=4 ranking metrics on clean held-out
    rank_metrics = compute_ranking_metrics(
        model=model, tokenizer=tokenizer,
        beta=BETA, n_eval=len(held_out_k4),
        max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT,
        device=DEVICE, seed=SEED,
        held_out_examples=held_out_k4,
    )

    # Eval B: pairwise accuracy on binarized prefs
    bin_metrics = pairwise_accuracy(
        model=model, tokenizer=tokenizer,
        beta=BETA, n_eval=n_eval_bin,
        max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT,
        device=DEVICE, seed=SEED,
    )

    del trainer
    _free(model)

    result = {
        "rho":                    rho,
        "model":                  model_name,
        "noise_type":             noise_type,
        "noise_level":            noise_level,
        "train_loss":             round(train_loss, 4),
        "top1_acc":               rank_metrics["top1_acc"],
        "exact_match":            rank_metrics["exact_match"],
        "kendall_tau":            rank_metrics["kendall_tau"],
        "ndcg":                   rank_metrics["ndcg"],
        "pairwise_acc_k4":        rank_metrics["pairwise_acc_k4"],
        "pairwise_acc_binarized": round(bin_metrics["accuracy"], 4),
        "elapsed_s":              round(time.time() - t0, 1),
    }
    logger.info("Result: %s", result)
    return result


# ---------------------------------------------------------------------------
# CSV / table helpers
# ---------------------------------------------------------------------------

def _save_csv(rows: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("CSV saved → %s", path)


def _print_table(rows: List[Dict]):
    if not rows:
        return
    hdr = (f"{'model':<12} {'rho':>6} {'top1':>6} {'exact':>6} "
           f"{'tau':>7} {'ndcg':>6} {'pw_k4':>6} {'pw_bin':>7}")
    sep = "─" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)
    for r in rows:
        rho_str = "  ref" if math.isnan(r["rho"]) else f"{r['rho']:>6.2f}"
        print(
            f"{r['model']:<12} {rho_str} "
            f"{r['top1_acc']:>6.4f} {r['exact_match']:>6.4f} "
            f"{r['kendall_tau']:>7.4f} {r['ndcg']:>6.4f} "
            f"{r['pairwise_acc_k4']:>6.4f} {r['pairwise_acc_binarized']:>7.4f}"
        )
    print(sep + "\n")

    # Print τ-vs-ρ summary
    sweep_rows = [r for r in rows if not math.isnan(r["rho"]) and r["model"] == "robust_pl"]
    if sweep_rows:
        best = max(sweep_rows, key=lambda r: r["kendall_tau"])
        print(f"  ρ* (best τ) = {best['rho']:.2f}  →  τ={best['kendall_tau']:.4f}  "
              f"top1={best['top1_acc']:.4f}")
        pl_ref = next((r for r in rows if math.isnan(r["rho"]) and r["model"] == "nominal_pl"), None)
        bt_ref = next((r for r in rows if math.isnan(r["rho"]) and r["model"] == "nominal_bt"), None)
        pl0_rows = [r for r in rows if not math.isnan(r["rho"]) and r["rho"] == 0.0]
        if pl_ref:
            print(f"  Nominal PL  = τ={pl_ref['kendall_tau']:.4f}  top1={pl_ref['top1_acc']:.4f}  (reference)")
        elif pl0_rows:
            pl = pl0_rows[0]
            print(f"  Nominal PL  = τ={pl['kendall_tau']:.4f}  top1={pl['top1_acc']:.4f}  (ρ=0, equiv. to nominal_pl)")
        if bt_ref:
            print(f"  Nominal BT  = τ={bt_ref['kendall_tau']:.4f}  top1={bt_ref['top1_acc']:.4f}  (reference)")
        if pl_ref and sweep_rows:
            delta = best["kendall_tau"] - pl_ref["kendall_tau"]
            print(f"  Δτ(ρ* − NominalPL) = {delta:+.4f}")
        print()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(rows: List[Dict], noise_type: str, noise_level: float, save_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot.")
        return

    sweep_rows = sorted(
        [r for r in rows if not math.isnan(r["rho"]) and r["model"] == "robust_pl"],
        key=lambda r: r["rho"],
    )
    if not sweep_rows:
        return

    rhos  = [r["rho"]          for r in sweep_rows]
    taus  = [r["kendall_tau"]  for r in sweep_rows]
    top1s = [r["top1_acc"]     for r in sweep_rows]
    pw_bins = [r["pairwise_acc_binarized"] for r in sweep_rows]

    ref_pl = next((r for r in rows if math.isnan(r["rho"]) and r["model"] == "nominal_pl"), None)
    ref_bt = next((r for r in rows if math.isnan(r["rho"]) and r["model"] == "nominal_bt"), None)
    # Fallback: use ρ=0 as nominal_pl if explicit ref not present
    if ref_pl is None:
        ref_pl = next((r for r in rows if not math.isnan(r["rho"]) and r["rho"] == 0.0), None)
    best_idx = taus.index(max(taus))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Robust PL ρ sweep  ·  noise={noise_type}/{noise_level}  ·  "
        f"Qwen2.5-0.5B  ·  {N_TRAIN} samples / {MAX_STEPS} steps",
        fontsize=12,
    )

    for ax, ys, ylabel, title in zip(
        axes,
        [taus, top1s, pw_bins],
        ["Kendall τ", "Top-1 Acc", "PairAcc (bin)"],
        ["τ vs. ρ (primary)", "Top-1 vs. ρ", "PairAcc(bin) vs. ρ"],
    ):
        ax.plot(rhos, ys, "b-o", label="Robust PL", zorder=3)

        # Mark ρ*
        ax.axvline(rhos[best_idx], color="green", linestyle=":", alpha=0.7,
                   label=f"ρ* = {rhos[best_idx]:.2f}")
        ax.scatter([rhos[best_idx]], [ys[best_idx]], color="green", s=80, zorder=5)

        # Primary reference: nominal_pl (orange solid)
        if ref_pl:
            ref_y = ref_pl["kendall_tau"] if ylabel == "Kendall τ" else (
                ref_pl["top1_acc"] if ylabel == "Top-1 Acc" else
                ref_pl["pairwise_acc_binarized"]
            )
            ax.axhline(ref_y, color="orange", linestyle="-", alpha=0.8, label="Nominal PL")

        # Secondary reference: nominal_bt (red dashed)
        if ref_bt:
            ref_y = ref_bt["kendall_tau"] if ylabel == "Kendall τ" else (
                ref_bt["top1_acc"] if ylabel == "Top-1 Acc" else
                ref_bt["pairwise_acc_binarized"]
            )
            ax.axhline(ref_y, color="red", linestyle="--", alpha=0.7, label="Nominal BT")

        ax.set_xlabel("ρ")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = os.path.join(save_dir, f"rho_sweep_{noise_type}_lvl{noise_level:.1f}.{ext}")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        logger.info("Plot saved → %s", out)
    plt.close()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ρ sweep for RobustPL on 0.5B model")
    p.add_argument("--quick",        action="store_true",   help="Smoke test (20 steps, 50 eval)")
    p.add_argument("--noise_type",   default="top_rank",    choices=["near_tie", "top_rank"])
    p.add_argument("--noise_level",  type=float, default=0.4)
    p.add_argument("--rho_values",   type=float, nargs="+", default=None,
                   help="ρ values to sweep (default: 0.0 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0)")
    p.add_argument("--no_ref",       action="store_true",   help="Skip all reference runs (nominal_pl + nominal_bt)")
    p.add_argument("--no_ref_bt",    action="store_true",   help="Skip nominal_bt reference run (keep nominal_pl)")
    p.add_argument("--no_plot",      action="store_true",   help="Skip matplotlib plot")
    p.add_argument("--n_train",      type=int, default=None)
    p.add_argument("--max_steps",    type=int, default=None)
    p.add_argument("--n_eval_k4",    type=int, default=None)
    p.add_argument("--n_eval_bin",   type=int, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Hyperparams
    quick      = args.quick
    n_train    = args.n_train    or (50     if quick else N_TRAIN)
    max_steps  = args.max_steps  or (20     if quick else MAX_STEPS)
    n_eval_k4  = args.n_eval_k4  or (50     if quick else N_EVAL_K4)
    n_eval_bin = args.n_eval_bin or (50     if quick else N_EVAL_BIN)
    rho_values = args.rho_values or DEFAULT_RHO_VALUES

    noise_type  = args.noise_type
    noise_level = args.noise_level

    csv_path  = os.path.join(OUTPUT_DIR, "rho_sweep_results.csv")
    plot_dir  = os.path.join(OUTPUT_DIR, "plots")

    logger.info("=" * 60)
    logger.info("ρ sweep — RobustPL on Qwen2.5-0.5B-Instruct")
    logger.info("  noise      = %s / %.1f", noise_type, noise_level)
    logger.info("  ρ values   = %s", rho_values)
    logger.info("  n_train    = %d  max_steps = %d", n_train, max_steps)
    logger.info("  n_eval_k4  = %d  n_eval_bin = %d", n_eval_k4, n_eval_bin)
    logger.info("  output     = %s", csv_path)
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build noisy dataset once (shared across all ρ)
    logger.info("Building noisy training dataset (%d samples, %s/%.1f) ...",
                n_train, noise_type, noise_level)
    noise_fn = make_noise_fn(noise_type, noise_prob=noise_level, seed=SEED)
    dataset = build_listwise_dataset(
        tokenizer=tokenizer,
        n_samples=n_train,
        noise_fn=noise_fn,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT,
        seed=SEED,
    )
    logger.info("Training dataset: %d examples", len(dataset))

    # Build clean held-out once (skip training data)
    logger.info("Building clean held-out K=4 set (%d examples) ...", n_eval_k4)
    held_out_k4 = build_held_out_k4(
        tokenizer=tokenizer,
        n_train_skip=n_train,
        n_eval=n_eval_k4,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT,
        seed=SEED,
    )
    logger.info("Held-out K=4 ready: %d examples", len(held_out_k4))

    all_results = []

    # 1a. Primary reference: nominal_pl on same noise
    #     (ρ=0.0 in the sweep is mathematically identical, but this gives a
    #      clearly-labelled baseline row for easy comparison)
    if not args.no_ref:
        logger.info("--- Reference 1/2: nominal_pl on %s/%.1f ---", noise_type, noise_level)
        result = run_one(
            model_name="nominal_pl",
            rho=float("nan"),
            noise_type=noise_type,
            noise_level=noise_level,
            tokenizer=tokenizer,
            dataset=dataset,
            held_out_k4=held_out_k4,
            max_steps=max_steps,
            n_eval_bin=n_eval_bin,
        )
        all_results.append(result)
        _save_csv(all_results, csv_path)

    # 1b. Secondary reference: nominal_bt (loss uses only top/bottom pair)
    if not args.no_ref and not getattr(args, "no_ref_bt", False):
        logger.info("--- Reference 2/2: nominal_bt on %s/%.1f ---", noise_type, noise_level)
        result = run_one(
            model_name="nominal_bt",
            rho=float("nan"),
            noise_type=noise_type,
            noise_level=noise_level,
            tokenizer=tokenizer,
            dataset=dataset,
            held_out_k4=held_out_k4,
            max_steps=max_steps,
            n_eval_bin=n_eval_bin,
        )
        all_results.append(result)
        _save_csv(all_results, csv_path)

    # 2. ρ sweep: RobustPL at each ρ value
    for rho in rho_values:
        result = run_one(
            model_name="robust_pl",
            rho=rho,
            noise_type=noise_type,
            noise_level=noise_level,
            tokenizer=tokenizer,
            dataset=dataset,
            held_out_k4=held_out_k4,
            max_steps=max_steps,
            n_eval_bin=n_eval_bin,
        )
        all_results.append(result)
        _save_csv(all_results, csv_path)

    # Summary
    _print_table(all_results)

    # Plot
    if not args.no_plot:
        _plot(all_results, noise_type, noise_level, plot_dir)

    logger.info("RHO_SWEEP_DONE  →  %s", csv_path)
    print("RHO_SWEEP_DONE")


if __name__ == "__main__":
    main()
