#!/usr/bin/env python3
"""
Minimal pairwise DPO baseline — sanity check.

Stack : Qwen/Qwen2.5-0.5B-Instruct + LoRA (r=16) + 4-bit NF4 + TRL DPOTrainer
Dataset: HuggingFaceH4/ultrafeedback_binarized (500-sample subset)
Goal  : verify TRL + Qwen + LoRA + 4-bit training stack is functional
        and that DPO loss decreases over 50 steps.

Usage:
    conda activate robust_listwise_llm
    CUDA_VISIBLE_DEVICES=0 python src/scripts/train_dpo_baseline.py
"""

import logging
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

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
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"
OUTPUT_DIR = "outputs/checkpoints/dpo_baseline"

N_TRAIN_SAMPLES = 500   # small subset; full split has ~60 k rows
MAX_STEPS = 50          # enough to see loss trend
BATCH_SIZE = 4
GRAD_ACCUM = 2          # effective batch = 8
LR = 5e-5
BETA = 0.1              # KL penalty coefficient
MAX_LENGTH = 512        # total (prompt + response) token budget
MAX_PROMPT_LENGTH = 256
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
SEED = 42


# ---------------------------------------------------------------------------
# 1. Tokenizer
# ---------------------------------------------------------------------------
logger.info("Loading tokenizer: %s", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# decoder-only models: pad on the right is fine for DPO (labels mask prompt)
tokenizer.padding_side = "right"


# ---------------------------------------------------------------------------
# 2. Model — 4-bit NF4 quantization, single GPU
# ---------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

logger.info("Loading base model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": "cuda:0"},
    trust_remote_code=True,
)
model.config.use_cache = False

# Required before applying LoRA to a quantized model
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)


# ---------------------------------------------------------------------------
# 3. LoRA adapter
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 4. Dataset preprocessing
#    ultrafeedback_binarized schema:
#      prompt      : str
#      chosen      : list[dict]  — [{"role": "user", ...}, {"role": "assistant", ...}]
#      rejected    : list[dict]  — same shape
#    DPOTrainer expects plain strings: prompt / chosen / rejected
# ---------------------------------------------------------------------------
logger.info("Loading dataset: %s", DATASET_NAME)
raw = load_dataset(DATASET_NAME, split="train_prefs").shuffle(seed=SEED)
raw = raw.select(range(N_TRAIN_SAMPLES))


def preprocess(example):
    """Extract plain-string prompt / chosen / rejected from message lists."""
    chosen_msgs = example["chosen"]    # list of {role, content}
    rejected_msgs = example["rejected"]

    prompt = example["prompt"]                        # already a plain string
    chosen_resp = chosen_msgs[-1]["content"]          # last assistant turn
    rejected_resp = rejected_msgs[-1]["content"]

    return {"prompt": prompt, "chosen": chosen_resp, "rejected": rejected_resp}


dataset = raw.map(preprocess, remove_columns=raw.column_names, num_proc=4)
logger.info("Dataset size: %d", len(dataset))
logger.info("Sample — prompt[:80]: %s", dataset[0]["prompt"][:80])


# ---------------------------------------------------------------------------
# 5. Training configuration
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    # ---- scale ----
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    # ---- DPO ----
    beta=BETA,
    loss_type="sigmoid",
    max_length=MAX_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGTH,
    # ---- optimizer ----
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    # ---- precision ----
    bf16=True,
    # ---- logging / saving ----
    logging_steps=5,
    save_steps=25,
    save_total_limit=2,
    # ---- misc ----
    remove_unused_columns=False,
    dataloader_num_workers=0,
    report_to="none",
    seed=SEED,
)


# ---------------------------------------------------------------------------
# 6. DPO Trainer
#    ref_model=None: because the model is a PEFT model, TRL automatically
#    uses the frozen base (adapter disabled) as the reference.
# ---------------------------------------------------------------------------
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

logger.info("=" * 60)
logger.info("Starting DPO training — %d steps", MAX_STEPS)
logger.info("Model : %s  (4-bit NF4 + LoRA r=%d)", MODEL_NAME, LORA_R)
logger.info("Data  : %s  (%d samples)", DATASET_NAME, len(dataset))
logger.info("=" * 60)

train_result = trainer.train()

# ---------------------------------------------------------------------------
# 7. Save checkpoint
# ---------------------------------------------------------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

logger.info("=" * 60)
logger.info("Training complete.")
logger.info("  train_loss       : %.4f", train_result.training_loss)
logger.info("  checkpoint saved : %s", OUTPUT_DIR)
logger.info("=" * 60)
