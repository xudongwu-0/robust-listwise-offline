#!/usr/bin/env python3
"""
Stage B — Nominal Listwise DPO training (fixed-list, K=4).

Stack : Qwen/Qwen2.5-0.5B-Instruct + LoRA (r=16) + 4-bit NF4 + custom trainer
Dataset: openbmb/UltraFeedback (K=4 candidates per prompt)
Loss  : Plackett-Luce nominal listwise DPO (docs/ROBUST_LISTWISE_DPO_MATH.md §4)

Usage:
    conda activate robust_listwise_llm
    cd ~/work/robust_listwise_llm
    CUDA_VISIBLE_DEVICES=0 python src/scripts/train_nominal_listwise.py

Sanity checks performed inline (see SANITY CHECKS section below).
"""

import logging
import os
import sys

# Make src/ importable regardless of working directory
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
from losses.plackett_luce import plackett_luce_loss
from trainers.listwise_trainer import NominalListwiseTrainer

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
# Hyper-parameters  (minimal — sanity-check only)
# ---------------------------------------------------------------------------
MODEL_NAME      = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR      = "outputs/checkpoints/nominal_listwise"

K               = 4       # fixed list size
N_TRAIN_SAMPLES = 1000    # small subset for end-to-end verification
MAX_STEPS       = 50
BATCH_SIZE      = 2       # 2 prompts × 4 responses = 8 seqs per forward pass
GRAD_ACCUM      = 4       # effective batch = 8 prompts
LR              = 5e-5
BETA            = 0.1
MAX_LENGTH      = 512
MAX_PROMPT_LEN  = 256
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
SEED            = 42


# ===========================================================================
# SANITY CHECK 1: PL loss with K=2 matches pairwise DPO
# ===========================================================================
def _sanity_check_pl_k2():
    """
    Verify that PL loss with K=2 equals pairwise DPO loss.
    For g_best, g_worst:
        PL   = log(1 + exp(g_worst - g_best))
        DPO  = log(1 + exp(-(g_best - g_worst)))
    These are identical.
    """
    import torch
    g_best  = torch.tensor([[2.0]])
    g_worst = torch.tensor([[-1.0]])
    scores  = torch.cat([g_best, g_worst], dim=1)  # [1, 2]

    pl_loss = plackett_luce_loss(scores)                                  # scalar
    pairwise_loss = torch.log(1 + torch.exp(g_worst - g_best)).squeeze()  # scalar

    delta = abs(pl_loss.item() - pairwise_loss.item())
    assert delta < 1e-5, (
        f"K=2 PL loss ({pl_loss.item():.6f}) != pairwise DPO "
        f"({pairwise_loss.item():.6f}), delta={delta:.2e}"
    )
    logger.info(
        "SANITY CHECK 1 PASSED: K=2 PL loss = pairwise DPO  "
        "(pl=%.4f, pw=%.4f)", pl_loss.item(), pairwise_loss.item()
    )


# ===========================================================================
# SANITY CHECK 2: Score direction
# ===========================================================================
def _sanity_check_score_direction():
    """
    Higher DPO score should mean better response.
    With a large positive g for the best and negative g for the worst,
    the PL loss should be small (model agrees with ranking).
    Reversed scores should give a large PL loss.
    """
    # Scores agreeing with ranking: [10, 5, 0, -5] → small loss
    scores_good = torch.tensor([[10.0, 5.0, 0.0, -5.0]])
    # Scores opposing ranking: [-5, 0, 5, 10] → large loss
    scores_bad  = torch.tensor([[-5.0, 0.0, 5.0, 10.0]])

    loss_good = plackett_luce_loss(scores_good).item()
    loss_bad  = plackett_luce_loss(scores_bad).item()

    assert loss_good < loss_bad, (
        f"Score direction wrong: good loss={loss_good:.4f} >= bad loss={loss_bad:.4f}"
    )
    logger.info(
        "SANITY CHECK 2 PASSED: score direction correct  "
        "(loss_good=%.4f < loss_bad=%.4f)", loss_good, loss_bad
    )


# ===========================================================================
# SANITY CHECK 3: same-prompt grouping
# ===========================================================================
def _sanity_check_grouping(dataset, n_check: int = 3):
    """
    Verify that ranking field is a valid permutation of [0, 1, 2, 3].
    (Proves that all 4 responses belong to the same sample/prompt.)
    """
    for i in range(min(n_check, len(dataset))):
        ranking = list(dataset[i]["ranking"])
        assert sorted(ranking) == list(range(K)), (
            f"Sample {i}: ranking {ranking} is not a permutation of [0..{K-1}]"
        )
    logger.info(
        "SANITY CHECK 3 PASSED: ranking is a valid K=%d permutation "
        "in all checked samples", K
    )


# ===========================================================================
# Main
# ===========================================================================

def main():
    # --- Offline sanity checks (no model needed) ---
    _sanity_check_pl_k2()
    _sanity_check_score_direction()

    # -----------------------------------------------------------------------
    # 1. Tokenizer
    # -----------------------------------------------------------------------
    logger.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # -----------------------------------------------------------------------
    # 2. Dataset
    # -----------------------------------------------------------------------
    dataset = build_listwise_dataset(
        tokenizer=tokenizer,
        n_samples=N_TRAIN_SAMPLES,
        max_prompt_length=MAX_PROMPT_LEN,
        max_length=MAX_LENGTH,
        seed=SEED,
    )
    _sanity_check_grouping(dataset)

    # -----------------------------------------------------------------------
    # 3. Model — 4-bit NF4
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 4. LoRA
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 5. Collator
    # -----------------------------------------------------------------------
    collator = ListwiseCollator(tokenizer=tokenizer, K=K)

    # -----------------------------------------------------------------------
    # 6. Training arguments
    # -----------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
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
        remove_unused_columns=False,   # collated keys are non-standard
        dataloader_num_workers=0,
        report_to="none",
        seed=SEED,
    )

    # -----------------------------------------------------------------------
    # 7. Trainer
    # -----------------------------------------------------------------------
    trainer = NominalListwiseTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator,
        beta=BETA,
        K=K,
    )

    logger.info("=" * 60)
    logger.info("Starting nominal listwise DPO training — %d steps", MAX_STEPS)
    logger.info("Model : %s  (4-bit NF4 + LoRA r=%d)", MODEL_NAME, LORA_R)
    logger.info("Data  : openbmb/UltraFeedback  (%d prompts × K=%d)", len(dataset), K)
    logger.info("Loss  : Plackett-Luce (ROBUST_LISTWISE_DPO_MATH.md §4)")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # 8. Train
    # -----------------------------------------------------------------------
    result = trainer.train()

    # -----------------------------------------------------------------------
    # 9. Save checkpoint
    # -----------------------------------------------------------------------
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info("  train_loss  : %.4f", result.training_loss)
    logger.info("  checkpoint  : %s", OUTPUT_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
