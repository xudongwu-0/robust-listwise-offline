"""
Pairwise accuracy evaluation on HuggingFaceH4/ultrafeedback_binarized test_prefs.

Metric: fraction of test examples where
    g_theta(x, chosen) > g_theta(x, rejected)

where the DPO score g_theta is exactly what was used during training (§1):
    g_theta(x, y) = beta * log( pi_theta(y|x) / pi_ref(y|x) )

This is the canonical "on-policy reward" metric and measures whether the
trained policy correctly prefers the annotated-better response.
"""

import logging
from typing import Dict, List

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize_pair(
    tokenizer,
    prompt: str,
    response: str,
    max_prompt_length: int,
    max_length: int,
):
    """Return (input_ids, labels) as 1-D Python lists for one (prompt, response) pair."""
    _enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    # transformers>=5.x may return BatchEncoding instead of list
    if hasattr(_enc, "input_ids"):
        prompt_ids: List[int] = list(_enc.input_ids)
    else:
        prompt_ids = list(_enc)
    prompt_ids = prompt_ids[:max_prompt_length]

    resp_ids: List[int] = list(tokenizer(response, add_special_tokens=False).input_ids)
    max_resp = max_length - len(prompt_ids)
    max_resp = max(max_resp, 1)
    resp_ids = resp_ids[:max_resp] + [tokenizer.eos_token_id]

    input_ids = prompt_ids + resp_ids
    labels    = [-100] * len(prompt_ids) + resp_ids
    return input_ids, labels


def _batch_log_probs(model, input_ids_list, labels_list, device):
    """
    Run a forward pass over a padded batch and return per-sequence log-prob sums [N].
    """
    N = len(input_ids_list)
    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = model.config.pad_token_id or 0

    input_ids_t = torch.full((N, max_len), pad_id, dtype=torch.long, device=device)
    attn_t      = torch.zeros(N, max_len, dtype=torch.long, device=device)
    labels_t    = torch.full((N, max_len), -100, dtype=torch.long, device=device)

    for i, (ids, lbl) in enumerate(zip(input_ids_list, labels_list)):
        L = len(ids)
        input_ids_t[i, :L] = torch.tensor(ids, dtype=torch.long)
        attn_t[i, :L]      = 1
        labels_t[i, :L]    = torch.tensor(lbl, dtype=torch.long)

    with torch.no_grad():
        output = model(input_ids=input_ids_t, attention_mask=attn_t)

    # Move to CPU for device_map='auto' multi-GPU compatibility and memory safety
    logits       = output.logits.cpu().float()
    shift_logits = logits[:, :-1, :]
    shift_labels = labels_t[:, 1:].cpu()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    mask      = shift_labels != -100
    indices   = shift_labels.clamp(min=0).unsqueeze(-1)
    token_lp  = log_probs.gather(-1, indices).squeeze(-1) * mask

    return token_lp.sum(dim=1)  # [N] on CPU


# ---------------------------------------------------------------------------
# Public evaluation function
# ---------------------------------------------------------------------------

def pairwise_accuracy(
    model,
    tokenizer,
    beta: float = 0.1,
    n_eval: int = 200,
    max_length: int = 512,
    max_prompt_length: int = 256,
    eval_batch_size: int = 4,
    device: str = "cuda:0",
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate the DPO pairwise accuracy on ultrafeedback_binarized test_prefs.

    For each (prompt, chosen, rejected) in the test set, compute:
        g_chosen   = beta * (log_pi_theta(chosen|prompt) - log_pi_ref(chosen|prompt))
        g_rejected = beta * (log_pi_theta(rejected|prompt) - log_pi_ref(rejected|prompt))
    Accuracy = mean(g_chosen > g_rejected).

    The reference model is obtained by disabling the LoRA adapter.

    Args:
        model          : PEFT model after training (on GPU, adapter enabled)
        tokenizer      : matching tokenizer
        beta           : DPO temperature (must match training)
        n_eval         : number of test examples to evaluate
        eval_batch_size: number of sequences per forward pass (4 per prompt → 8 seqs)
        device         : CUDA device string

    Returns:
        {"accuracy": float, "mean_margin": float, "n_eval": int}
    """
    from datasets import load_dataset

    logger.info("Loading ultrafeedback_binarized test_prefs (%d examples) ...", n_eval)
    test_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    test_ds = test_ds.shuffle(seed=seed).select(range(min(n_eval, len(test_ds))))

    model.eval()

    # Pre-tokenize all examples (chosen and rejected)
    chosen_ids_list,   chosen_lbl_list   = [], []
    rejected_ids_list, rejected_lbl_list = [], []

    for ex in test_ds:
        prompt   = ex["prompt"]
        chosen   = ex["chosen"][-1]["content"]
        rejected = ex["rejected"][-1]["content"]

        c_ids, c_lbl = _tokenize_pair(tokenizer, prompt, chosen,   max_prompt_length, max_length)
        r_ids, r_lbl = _tokenize_pair(tokenizer, prompt, rejected, max_prompt_length, max_length)

        chosen_ids_list.append(c_ids);   chosen_lbl_list.append(c_lbl)
        rejected_ids_list.append(r_ids); rejected_lbl_list.append(r_lbl)

    N = len(test_ds)

    def batched_log_probs(ids_list, lbl_list):
        """Run all N sequences in mini-batches, return [N] tensor."""
        all_lp = []
        for start in range(0, N, eval_batch_size):
            ids_b = ids_list[start : start + eval_batch_size]
            lbl_b = lbl_list[start : start + eval_batch_size]
            all_lp.append(_batch_log_probs(model, ids_b, lbl_b, device))
        return torch.cat(all_lp)

    # --- Policy log probs (adapter ON) ---
    logger.info("Computing policy log probs ...")
    lp_c_policy = batched_log_probs(chosen_ids_list,   chosen_lbl_list)
    lp_r_policy = batched_log_probs(rejected_ids_list, rejected_lbl_list)

    # --- Reference log probs (adapter OFF) ---
    logger.info("Computing reference log probs ...")
    with model.disable_adapter():
        lp_c_ref = batched_log_probs(chosen_ids_list,   chosen_lbl_list)
        lp_r_ref = batched_log_probs(rejected_ids_list, rejected_lbl_list)

    # --- DPO scores ---
    g_chosen   = beta * (lp_c_policy - lp_c_ref)
    g_rejected = beta * (lp_r_policy - lp_r_ref)

    correct = (g_chosen > g_rejected).float()
    margins = (g_chosen - g_rejected)

    acc  = correct.mean().item()
    marg = margins.mean().item()

    logger.info(
        "Eval done: accuracy=%.4f  mean_margin=%.4f  n=%d",
        acc, marg, N
    )
    model.train()
    return {"accuracy": acc, "mean_margin": marg, "n_eval": N}
