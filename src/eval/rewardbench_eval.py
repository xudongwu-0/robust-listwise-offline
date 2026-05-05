"""
Minimal RewardBench integration.

RewardBench (Lambert et al., 2024) evaluates reward models / preference
proxies on a curated set of (prompt, chosen, rejected) triples spanning
multiple capability categories:

    Chat       — conversational quality
    Chat Hard  — difficult / adversarial chat
    Safety     — harmlessness / refusal
    Reasoning  — math, code, logical reasoning

Dataset: allenai/reward-bench  (HuggingFace Hub)

We evaluate our model using the DPO implicit reward proxy:
    g_θ(x, y) = β · log(π_θ(y|x) / π_ref(y|x))

and report pairwise accuracy  acc = P(g_θ(chosen) > g_θ(rejected))
for each subset, plus the overall average.

Note: RewardBench was designed for explicit reward heads that output scalars.
Using g_θ as a proxy reward is a reasonable zero-shot baseline, but may
under-estimate the model's true preference capability (the log-ratio was
calibrated vs a reference model, not against a broad reward distribution).
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DATASET_NAME = "allenai/reward-bench"

# RewardBench subset names as they appear in the 'subset' column
SUBSET_GROUPS = {
    "chat":      ["alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard",
                  "mt-bench-easy", "mt-bench-medium"],
    "chat_hard": ["mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor",
                  "llmbar-adver-GPTInst", "llmbar-adver-GPTOut",
                  "llmbar-adver-manual"],
    "safety":    ["refusals-dangerous", "refusals-offensive", "xstest-should-refuse",
                  "xstest-should-respond", "donotanswer"],
    "reasoning": ["math-prm", "hep-cpp", "hep-go", "hep-java", "hep-js",
                  "hep-python", "hep-rust"],
}


# ---------------------------------------------------------------------------
# Tokenisation helpers (shared with pairwise_accuracy.py)
# ---------------------------------------------------------------------------

def _tokenize_pair(
    tokenizer,
    prompt: str,
    response: str,
    max_prompt_length: int,
    max_length: int,
) -> Tuple[List[int], List[int]]:
    """Return (input_ids, labels) for one (prompt, response) pair."""
    prompt_ids: List[int] = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )[:max_prompt_length]
    resp_ids: List[int] = tokenizer(response, add_special_tokens=False).input_ids
    max_resp = max(max_length - len(prompt_ids), 1)
    resp_ids = resp_ids[:max_resp] + [tokenizer.eos_token_id]
    input_ids = prompt_ids + resp_ids
    labels    = [-100] * len(prompt_ids) + resp_ids
    return input_ids, labels


def _batch_log_probs(
    model,
    ids_list: List[List[int]],
    lbl_list: List[List[int]],
    device: str,
) -> torch.Tensor:
    """Padded batch forward → per-sequence log-prob sums [N]."""
    N       = len(ids_list)
    max_len = max(len(x) for x in ids_list)
    pad_id  = getattr(model.config, "pad_token_id", 0) or 0

    input_ids = torch.full((N, max_len), pad_id, dtype=torch.long, device=device)
    attn      = torch.zeros(N, max_len,  dtype=torch.long, device=device)
    labels    = torch.full((N, max_len), -100, dtype=torch.long, device=device)
    for i, (ids, lbl) in enumerate(zip(ids_list, lbl_list)):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        attn[i, :L]      = 1
        labels[i, :L]    = torch.tensor(lbl, dtype=torch.long)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn)

    # Move logits to CPU to handle device_map='auto' multi-GPU models and
    # avoid OOM from large vocab-size (Qwen-7B: ~152 K) logit tensors.
    logits       = out.logits.cpu().float()       # [N, L, V]
    shift_logits = logits[:, :-1, :]              # [N, L-1, V]
    shift_labels = labels[:, 1:].cpu()            # [N, L-1]
    log_probs    = F.log_softmax(shift_logits, dim=-1)
    mask         = shift_labels != -100
    gathered     = log_probs.gather(-1, shift_labels.clamp(0).unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(1)  # [N] on CPU


# ---------------------------------------------------------------------------
# Helper: extract prompt/chosen/rejected text from various data formats
# ---------------------------------------------------------------------------

def _extract_text(field) -> str:
    """
    RewardBench rows may have 'chosen'/'rejected' as:
      - a plain string
      - a list of message dicts ({'role': ..., 'content': ...})
    Extract the assistant content in all cases.
    """
    if isinstance(field, str):
        return field
    if isinstance(field, list):
        for msg in reversed(field):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        # fallback: last item
        if field:
            last = field[-1]
            if isinstance(last, dict):
                return last.get("content", str(last))
            return str(last)
    return str(field)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def rewardbench_eval(
    model,
    tokenizer,
    beta: float = 0.1,
    n_eval: Optional[int] = None,     # None = use all
    max_length: int = 512,
    max_prompt_length: int = 256,
    eval_batch_size: int = 4,
    device: str = "cuda:0",
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate the model on RewardBench (allenai/reward-bench).

    Returns a dict:
        {
          "overall":    float,   # mean over all examples
          "chat":       float,   # subset group accuracy
          "chat_hard":  float,
          "safety":     float,
          "reasoning":  float,
          "n_eval":     int,
        }

    Falls back gracefully if the dataset is unavailable, returning empty dict
    with a warning rather than raising an exception.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets package not available; skipping RewardBench eval.")
        return {}

    try:
        logger.info("Loading %s ...", DATASET_NAME)
        ds = load_dataset(DATASET_NAME, split="filtered")
        logger.info("RewardBench size: %d", len(ds))
    except Exception as exc:
        logger.warning("Could not load RewardBench (%s); skipping.", exc)
        return {}

    if n_eval is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n_eval, len(ds))))

    # Pre-tokenise
    chosen_ids,   chosen_lbls   = [], []
    rejected_ids, rejected_lbls = [], []
    subsets = []
    skipped = 0

    for ex in ds:
        try:
            prompt   = ex.get("prompt", "") or ""
            chosen   = _extract_text(ex.get("chosen",   ""))
            rejected = _extract_text(ex.get("rejected", ""))
        except Exception:
            skipped += 1
            continue

        if not prompt or not chosen or not rejected:
            skipped += 1
            continue

        c_ids, c_lbl = _tokenize_pair(tokenizer, prompt, chosen,   max_prompt_length, max_length)
        r_ids, r_lbl = _tokenize_pair(tokenizer, prompt, rejected, max_prompt_length, max_length)
        chosen_ids.append(c_ids);   chosen_lbls.append(c_lbl)
        rejected_ids.append(r_ids); rejected_lbls.append(r_lbl)
        subsets.append(str(ex.get("subset", "unknown")))

    if skipped:
        logger.warning("Skipped %d mal-formed RewardBench examples.", skipped)

    N = len(subsets)
    if N == 0:
        logger.warning("No valid RewardBench examples after pre-processing.")
        return {}

    model.eval()

    def _all_log_probs(ids_list, lbl_list) -> torch.Tensor:
        parts = []
        for s in range(0, N, eval_batch_size):
            parts.append(_batch_log_probs(model, ids_list[s:s + eval_batch_size],
                                          lbl_list[s:s + eval_batch_size], device))
        return torch.cat(parts)

    logger.info("RewardBench: computing policy log-probs for %d examples ...", N)
    lp_c_pol = _all_log_probs(chosen_ids,   chosen_lbls)
    lp_r_pol = _all_log_probs(rejected_ids, rejected_lbls)

    logger.info("RewardBench: computing reference log-probs ...")
    with model.disable_adapter():
        lp_c_ref = _all_log_probs(chosen_ids,   chosen_lbls)
        lp_r_ref = _all_log_probs(rejected_ids, rejected_lbls)

    g_chosen   = beta * (lp_c_pol - lp_c_ref)   # [N]
    g_rejected = beta * (lp_r_pol - lp_r_ref)   # [N]
    correct    = (g_chosen > g_rejected).float().cpu().tolist()

    # Aggregate by subset group
    group_results: Dict[str, List[float]] = {k: [] for k in SUBSET_GROUPS}
    group_results["other"] = []

    for i, subset_name in enumerate(subsets):
        placed = False
        for group, members in SUBSET_GROUPS.items():
            if any(m in subset_name for m in members):
                group_results[group].append(correct[i])
                placed = True
                break
        if not placed:
            group_results["other"].append(correct[i])

    def _acc(lst):
        return round(sum(lst) / len(lst), 4) if lst else float("nan")

    results = {
        "overall":   round(sum(correct) / N, 4),
        "n_eval":    N,
    }
    for group in SUBSET_GROUPS:
        results[group] = _acc(group_results[group])
    if group_results["other"]:
        results["other"] = _acc(group_results["other"])

    logger.info(
        "RewardBench: overall=%.4f  chat=%.4f  chat_hard=%.4f  safety=%.4f  reasoning=%.4f (n=%d)",
        results["overall"], results.get("chat", float("nan")),
        results.get("chat_hard", float("nan")), results.get("safety", float("nan")),
        results.get("reasoning", float("nan")), N,
    )
    model.train()
    return results
