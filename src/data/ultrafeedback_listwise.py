"""
Fixed-list dataset and collator for nominal listwise DPO.

Data source: openbmb/UltraFeedback
    - Each sample has exactly 4 completions, each with an `overall_score`.
    - We build one training example per prompt:
        prompt  : instruction text
        responses: [y_1, y_2, y_3, y_4]  (original order)
        scores  : [s_1, s_2, s_3, s_4]   (overall_score for each)
        ranking : [i_1, i_2, i_3, i_4]   argsort(scores, descending=True),
                  i.e. ranking[0] = index of the best response

    noise_fn (optional): callable(ranking, scores) -> noisy_ranking
        Applied after ranking derivation, before tokenisation.
        See src/data/noise.py for factory functions.
"""

import logging
from typing import Callable, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_listwise_dataset(
    tokenizer: PreTrainedTokenizerBase,
    n_samples: Optional[int] = None,
    max_prompt_length: int = 256,
    max_length: int = 512,
    seed: int = 42,
    num_proc: int = 4,
    noise_fn: Optional[Callable[[List[int], List[float]], List[int]]] = None,
) -> Dataset:
    """
    Load openbmb/UltraFeedback and build a fixed-list K=4 dataset.

    Each output sample has columns:
        input_ids_k   (k=0..3): List[int]  tokenized (prompt + response_k)
        attention_mask_k         : List[int]
        labels_k                 : List[int]  same length; prompt tokens = -100
        ranking                  : List[int]  length K, indices in desc-score order

    Args:
        noise_fn: optional callable(ranking, scores) -> noisy_ranking.
                  Applied to each sample's ranking before tokenisation.
                  Use make_noise_fn() from src/data/noise.py to create one.

    Returns a HuggingFace Dataset ready to be passed to ListwiseCollator.
    """
    logger.info("Loading openbmb/UltraFeedback ...")
    raw = load_dataset("openbmb/UltraFeedback", split="train")

    if n_samples is not None:
        raw = raw.shuffle(seed=seed).select(range(n_samples))

    # Build prompt + responses + ranking for each sample
    def extract_fields(example):
        completions = example["completions"]
        # Require exactly 4 completions with numeric overall_score
        if len(completions) != 4:
            return None
        scores = []
        responses = []
        for c in completions:
            s = c.get("overall_score")
            if s is None:
                return None
            try:
                scores.append(float(s))
            except (TypeError, ValueError):
                return None
            responses.append(c["response"])

        # ranking[i] = index of the response at rank i (0 = best)
        ranking = sorted(range(4), key=lambda k: scores[k], reverse=True)

        return {
            "prompt": example["instruction"],
            "responses": responses,
            "scores": scores,
            "ranking": ranking,
        }

    logger.info("Extracting fields and filtering ...")
    processed = raw.map(
        extract_fields,
        remove_columns=raw.column_names,
        num_proc=num_proc,
    )
    # Remove None rows produced by the filter
    processed = processed.filter(lambda x: x["prompt"] is not None, num_proc=num_proc)
    logger.info("Samples after filtering: %d", len(processed))

    # --- Optional: inject ranking noise ---
    if noise_fn is not None:
        def apply_noise(example):
            noisy = noise_fn(example["ranking"], example["scores"])
            return {"ranking": noisy}
        # num_proc=1: noise_fn may use a stateful RNG; keep sequential for reproducibility
        processed = processed.map(apply_noise, num_proc=1)
        logger.info("Noise injection applied.")

    # Tokenise all (prompt, response_k) pairs up front
    def tokenise_sample(example):
        prompt = example["prompt"]
        responses = example["responses"]
        ranking = example["ranking"]

        # Format prompt with chat template (stops before the assistant response)
        _prompt_enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
        # Newer transformers may return BatchEncoding; normalise to plain list
        if hasattr(_prompt_enc, "input_ids"):
            prompt_ids: List[int] = list(_prompt_enc.input_ids)
        else:
            prompt_ids: List[int] = list(_prompt_enc)
        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[:max_prompt_length]
        prompt_len = len(prompt_ids)

        out = {"ranking": ranking}
        for k, resp in enumerate(responses):
            resp_ids: List[int] = tokenizer(
                resp, add_special_tokens=False
            ).input_ids
            # Append EOS
            resp_ids = resp_ids + [tokenizer.eos_token_id]
            # Truncate response so that total fits in max_length
            max_resp = max_length - prompt_len
            if max_resp <= 0:
                # Extreme case: prompt already fills budget — keep 1 token
                max_resp = 1
            resp_ids = resp_ids[:max_resp]

            full_ids = prompt_ids + resp_ids
            labels = [-100] * prompt_len + resp_ids
            attn = [1] * len(full_ids)

            out[f"input_ids_{k}"] = full_ids
            out[f"attention_mask_{k}"] = attn
            out[f"labels_{k}"] = labels

        return out

    logger.info("Tokenising samples ...")
    tokenised = processed.map(
        tokenise_sample,
        remove_columns=["prompt", "responses", "scores"],
        num_proc=num_proc,
    )
    tokenised.set_format(type=None)   # keep as Python lists for collator
    logger.info("Tokenisation done. Dataset size: %d", len(tokenised))
    return tokenised


# ---------------------------------------------------------------------------
# K=8 dataset builder (pair consecutive K=4 examples)
# ---------------------------------------------------------------------------

def build_listwise_dataset_k8(
    tokenizer: PreTrainedTokenizerBase,
    n_samples: Optional[int] = None,
    max_prompt_length: int = 256,
    max_length: int = 512,
    seed: int = 42,
    noise_fn: Optional[Callable[[List[int], List[float]], List[int]]] = None,
) -> Dataset:
    """
    Build a K=8 listwise training dataset by pairing consecutive K=4 examples.

    For each K=8 training example:
      - Prompt / responses 0-3 : from example_A (on-topic)
      - Responses 4-7          : from example_B (off-topic hard negatives)
      - Ranking                : all 8 responses sorted by their combined scores

    This preserves the mathematical structure needed to test the Plackett-Luce
    hypothesis (larger K → richer listwise signal) while reusing UltraFeedback
    whose raw examples each have exactly 4 completions.

    Evaluation is always performed on the K=4 held-out set for direct
    comparability with K=4 training experiments.

    Args:
        n_samples: number of K=8 training examples to create.
                   Each uses 2 raw examples, so (2*n_samples) raw rows are loaded.
    """
    n_raw = (2 * n_samples + 200) if n_samples is not None else None

    logger.info("Loading openbmb/UltraFeedback for K=8 dataset (need ~%s raw rows) ...",
                n_raw or "all")
    raw = load_dataset("openbmb/UltraFeedback", split="train")
    if n_raw is not None:
        raw = raw.shuffle(seed=seed).select(range(min(n_raw, len(raw))))

    def extract_fields(example):
        completions = example["completions"]
        if len(completions) != 4:
            return None
        scores, responses = [], []
        for c in completions:
            s = c.get("overall_score")
            if s is None:
                return None
            try:
                scores.append(float(s))
            except (TypeError, ValueError):
                return None
            responses.append(c["response"])
        ranking = sorted(range(4), key=lambda k: scores[k], reverse=True)
        return {"prompt": example["instruction"],
                "responses": responses, "scores": scores, "ranking": ranking}

    logger.info("Extracting fields ...")
    processed = raw.map(extract_fields, remove_columns=raw.column_names, num_proc=4)
    processed = processed.filter(lambda x: x["prompt"] is not None, num_proc=4)
    logger.info("K=4 rows after filtering: %d (need %d pairs)", len(processed),
                n_samples or len(processed) // 2)

    # Build paired K=8 examples
    k8_rows = []
    for i in range(0, len(processed) - 1, 2):
        ex_a = processed[i]
        ex_b = processed[i + 1]
        combined_responses = ex_a["responses"] + ex_b["responses"]   # length 8
        combined_scores    = ex_a["scores"]    + ex_b["scores"]       # length 8
        combined_ranking   = sorted(range(8), key=lambda k: combined_scores[k], reverse=True)
        k8_rows.append({
            "prompt":    ex_a["prompt"],   # anchor on first example's prompt
            "responses": combined_responses,
            "scores":    combined_scores,
            "ranking":   combined_ranking,
        })
        if n_samples is not None and len(k8_rows) >= n_samples:
            break

    logger.info("Created %d K=8 examples from %d raw rows", len(k8_rows), len(processed))

    from datasets import Dataset as HFDataset
    paired = HFDataset.from_list(k8_rows)

    if noise_fn is not None:
        def apply_noise(example):
            return {"ranking": noise_fn(example["ranking"], example["scores"])}
        paired = paired.map(apply_noise, num_proc=1)
        logger.info("Noise injection applied to K=8 dataset.")

    # Tokenise — same logic as K=4 version
    def tokenise_sample(example):
        prompt    = example["prompt"]
        responses = example["responses"]
        ranking   = example["ranking"]

        _enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=True,
        )
        prompt_ids: List[int] = list(_enc.input_ids if hasattr(_enc, "input_ids") else _enc)
        prompt_ids = prompt_ids[:max_prompt_length]
        prompt_len = len(prompt_ids)

        out = {"ranking": ranking}
        for k, resp in enumerate(responses):
            resp_ids: List[int] = tokenizer(resp, add_special_tokens=False).input_ids
            resp_ids = resp_ids + [tokenizer.eos_token_id]
            max_resp = max(max_length - prompt_len, 1)
            resp_ids = resp_ids[:max_resp]
            full_ids = prompt_ids + resp_ids
            out[f"input_ids_{k}"]      = full_ids
            out[f"attention_mask_{k}"] = [1] * len(full_ids)
            out[f"labels_{k}"]         = [-100] * prompt_len + resp_ids
        return out

    logger.info("Tokenising K=8 samples ...")
    tokenised = paired.map(
        tokenise_sample,
        remove_columns=["prompt", "responses", "scores"],
        num_proc=4,
    )
    tokenised.set_format(type=None)
    logger.info("K=8 tokenisation done. Dataset size: %d", len(tokenised))
    return tokenised


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class ListwiseCollator:
    """
    Collate a batch of listwise samples into model-ready tensors.

    Input (list of B dicts with keys input_ids_k, attention_mask_k, labels_k,
    ranking for k in 0..K-1):

    Output dict:
        input_ids     : [B*K, L]   — padded, row order: (s0k0, s0k1, ..., s1k0, ...)
        attention_mask: [B*K, L]
        labels        : [B*K, L]   — prompt tokens = -100
        ranking       : [B, K]     — ground-truth rank indices (0 = best)
        batch_size    : int B      — stored for reshaping in the trainer
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, K: int = 4):
        self.tokenizer = tokenizer
        self.K = K
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        B = len(features)
        K = self.K

        # Collect all B*K sequences
        all_input_ids: List[List[int]] = []
        all_attn: List[List[int]] = []
        all_labels: List[List[int]] = []
        rankings: List[List[int]] = []

        for feat in features:
            rankings.append(feat["ranking"])
            for k in range(K):
                all_input_ids.append(feat[f"input_ids_{k}"])
                all_attn.append(feat[f"attention_mask_{k}"])
                all_labels.append(feat[f"labels_{k}"])

        # Pad to max length in this batch
        max_len = max(len(ids) for ids in all_input_ids)

        input_ids_tensor = torch.full((B * K, max_len), self.pad_id, dtype=torch.long)
        attn_tensor = torch.zeros(B * K, max_len, dtype=torch.long)
        labels_tensor = torch.full((B * K, max_len), -100, dtype=torch.long)

        for i, (ids, attn, lbl) in enumerate(
            zip(all_input_ids, all_attn, all_labels)
        ):
            seq_len = len(ids)
            input_ids_tensor[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attn_tensor[i, :seq_len] = torch.tensor(attn, dtype=torch.long)
            labels_tensor[i, :seq_len] = torch.tensor(lbl, dtype=torch.long)

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attn_tensor,
            "labels": labels_tensor,
            "ranking": torch.tensor(rankings, dtype=torch.long),  # [B, K]
            "batch_size": B,
        }
