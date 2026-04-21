#!/usr/bin/env python3
"""
Stage C — Robust Listwise DPO training (fixed-list, K=4).

Stack : Qwen/Qwen2.5-0.5B-Instruct + LoRA (r=16) + 4-bit NF4
Dataset: openbmb/UltraFeedback (K=4 candidates per prompt)
Loss  : Robust Plackett-Luce listwise DPO (docs/ROBUST_LISTWISE_DPO_MATH.md §5)

    ell_robust = (1-rho) * ell_PL(sigma_obs) + rho * ell_PL(sigma_wc)
    sigma_wc = argsort(g, ascending=True)   [§6]

Usage:
    conda activate robust_listwise_llm
    cd ~/work/robust_listwise_llm

    # rho=0 → identical to nominal listwise (sanity check)
    CUDA_VISIBLE_DEVICES=0 python src/scripts/train_robust_listwise.py --rho 0.0

    # rho=0.1 (default)
    CUDA_VISIBLE_DEVICES=0 python src/scripts/train_robust_listwise.py --rho 0.1

    # rho=0.5
    CUDA_VISIBLE_DEVICES=0 python src/scripts/train_robust_listwise.py --rho 0.5
"""

import argparse
import logging
import os
import sys

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
from losses.plackett_luce import (
    plackett_luce_loss,
    robust_pl_loss,
    worst_case_ranking,
)
from trainers.listwise_trainer import RobustListwiseTrainer

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
# Fixed hyper-parameters
# ---------------------------------------------------------------------------
MODEL_NAME    = "Qwen/Qwen2.5-0.5B-Instruct"
K             = 4
N_SAMPLES     = 1000
MAX_STEPS     = 50
BATCH_SIZE    = 2
GRAD_ACCUM    = 4
LR            = 5e-5
BETA          = 0.1
MAX_LENGTH    = 512
MAX_PROMPT    = 256
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
SEED          = 42


# ===========================================================================
# Sanity checks (run before model loads, CPU only)
# ===========================================================================

def _check_rho0_equals_nominal():
    """
    SANITY CHECK 1: robust_pl_loss with rho=0 must equal plackett_luce_loss exactly.
    (§5: rho=0 → (1-0)*nominal + 0*worst = nominal)
    """
    torch.manual_seed(0)
    g        = torch.randn(4, K)          # [B=4, K=4]  random scores
    ranking  = torch.argsort(g, dim=1, descending=True)  # descending = nominal order

    nominal  = plackett_luce_loss(g.gather(1, ranking))
    robust0  = robust_pl_loss(g, ranking, rho=0.0)

    delta = abs(nominal.item() - robust0.item())
    assert delta < 1e-6, (
        f"rho=0 robust loss ({robust0.item():.8f}) != nominal ({nominal.item():.8f}), "
        f"delta={delta:.2e}"
    )
    logger.info(
        "SANITY CHECK 1 PASSED: rho=0 robust == nominal  "
        "(nominal=%.6f, robust0=%.6f)", nominal.item(), robust0.item()
    )


def _check_worst_case_is_ascending():
    """
    SANITY CHECK 2: worst_case_ranking must return ascending-score order (§6).
    The index of the SMALLEST score must appear at position 0.
    The index of the LARGEST score must appear at position K-1.
    """
    g = torch.tensor([[3.0, 1.0, 4.0, 2.0]])   # sorted asc: 1<2<3<4 → indices [1,3,0,2]
    wc = worst_case_ranking(g)
    expected = torch.argsort(g, dim=1, descending=False)
    assert torch.equal(wc, expected), (
        f"worst_case_ranking {wc.tolist()} != expected {expected.tolist()}"
    )
    # Sanity-sub: last index should point to the highest score
    last_idx = wc[0, -1].item()
    assert g[0, last_idx] == g[0].max(), "highest-scored candidate must be last"
    logger.info(
        "SANITY CHECK 2 PASSED: worst-case ranking is ascending-score order  "
        "(wc=%s)", wc[0].tolist()
    )


def _check_rho1_worst_case():
    """
    SANITY CHECK 3: with rho=1, robust loss must equal PL evaluated on worst-case ranking.
    """
    torch.manual_seed(1)
    g        = torch.randn(3, K)
    ranking  = torch.argsort(g, dim=1, descending=True)
    sigma_wc = worst_case_ranking(g)

    loss_rho1     = robust_pl_loss(g, ranking, rho=1.0)
    loss_wc_direct = plackett_luce_loss(g.gather(1, sigma_wc))

    delta = abs(loss_rho1.item() - loss_wc_direct.item())
    assert delta < 1e-6, (
        f"rho=1 robust ({loss_rho1.item():.8f}) != direct wc PL "
        f"({loss_wc_direct.item():.8f}), delta={delta:.2e}"
    )
    logger.info(
        "SANITY CHECK 3 PASSED: rho=1 robust == worst-case PL  "
        "(%.6f == %.6f)", loss_rho1.item(), loss_wc_direct.item()
    )


def _check_loss_ordering():
    """
    SANITY CHECK 4: with aligned scores (model agrees with ranking),
    nominal PL loss should be small; worst-case PL loss should be large.
    This verifies score direction is consistent with the robust objective.
    """
    # Scores perfectly aligned with ranking (best=10, worst=-10)
    g = torch.tensor([[10.0, 5.0, -5.0, -10.0]])   # already in descending order
    ranking = torch.tensor([[0, 1, 2, 3]])           # identity permutation

    nominal_loss = plackett_luce_loss(g.gather(1, ranking)).item()
    wc_ranking   = worst_case_ranking(g)
    wc_loss      = plackett_luce_loss(g.gather(1, wc_ranking)).item()

    assert nominal_loss < wc_loss, (
        f"Expected nominal_loss ({nominal_loss:.4f}) < wc_loss ({wc_loss:.4f})"
    )
    logger.info(
        "SANITY CHECK 4 PASSED: aligned scores → nominal (%.4f) < worst-case (%.4f)",
        nominal_loss, wc_loss
    )


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Robust listwise DPO training")
    parser.add_argument(
        "--rho", type=float, default=0.1,
        help="Robustness coefficient in [0,1]. rho=0 → nominal listwise."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override default output directory."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rho = args.rho
    output_dir = args.output_dir or f"outputs/checkpoints/robust_listwise_rho{rho:.2f}"

    # ------------------------------------------------------------------
    # Run sanity checks (CPU, no model needed)
    # ------------------------------------------------------------------
    _check_rho0_equals_nominal()
    _check_worst_case_is_ascending()
    _check_rho1_worst_case()
    _check_loss_ordering()

    logger.info("All sanity checks passed. Starting training with rho=%.2f", rho)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    logger.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = build_listwise_dataset(
        tokenizer=tokenizer,
        n_samples=N_SAMPLES,
        max_prompt_length=MAX_PROMPT,
        max_length=MAX_LENGTH,
        seed=SEED,
    )
    logger.info("Dataset size: %d", len(dataset))

    # ------------------------------------------------------------------
    # Model — 4-bit NF4
    # ------------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    logger.info("Loading model with 4-bit quantization ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg, autocast_adapter_dtype=False)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Collator
    # ------------------------------------------------------------------
    collator = ListwiseCollator(tokenizer=tokenizer, K=K)

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        bf16=True,
        logging_steps=5,
        save_steps=25,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        seed=SEED,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = RobustListwiseTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator,
        beta=BETA,
        K=K,
        rho=rho,
    )

    logger.info("=" * 60)
    logger.info("Starting ROBUST listwise DPO training")
    logger.info("  rho   = %.2f  (0=nominal, 1=fully worst-case)", rho)
    logger.info("  steps = %d  |  batch = %d prompts", MAX_STEPS, BATCH_SIZE * GRAD_ACCUM)
    logger.info("  model = %s  (4-bit NF4 + LoRA r=%d)", MODEL_NAME, LORA_R)
    logger.info("  data  = openbmb/UltraFeedback  (%d prompts × K=%d)", len(dataset), K)
    logger.info("=" * 60)

    result = trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info("  rho         : %.2f", rho)
    logger.info("  train_loss  : %.4f", result.training_loss)
    logger.info("  checkpoint  : %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
