"""
Comprehensive K=4 ranking evaluation on a held-out split of openbmb/UltraFeedback.

Five metrics are computed for each held-out example, then averaged:

    top1_acc   — top-1 accuracy: is the model's rank-1 response the true best?
    exact_match — full 4-way ranking correct (all 4 positions match ground truth)
    kendall_tau — Kendall rank correlation between predicted and true ranking
    ndcg        — NDCG@4 using original overall_score as relevance
    pairwise_acc— fraction of C(4,2)=6 pairwise preferences that are correct

Definitions
-----------
Let π* = true ranking (argsort scores, descending) and π̂ = model ranking
(argsort g_θ, descending).  Let r*(i) (resp. r̂(i)) be the rank of response i
in π* (resp. π̂), where rank 0 = best.

top1_acc   = 1[π*(0) == π̂(0)]                     (1 if same best response)
exact_match= 1[π* == π̂]                             (4-way exact)
kendall_tau= τ computed between (r*(0)..r*(3)) and (r̂(0)..r̂(3)) via scipy
ndcg       = NDCG(y_true=scores, y_score=g_θ) via sklearn.metrics.ndcg_score
pairwise_acc= |{(i,j): i≻^* j AND g_θ(i)>g_θ(j)}| / C(4,2)

The held-out set is derived by shuffling UltraFeedback with seed=42 and
skipping the first `n_train_skip` samples.
"""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Held-out dataset builder
# ---------------------------------------------------------------------------

def build_held_out_k4(
    tokenizer,
    n_train_skip: int = 5000,    # number of training samples to skip
    n_eval: int = 500,
    max_prompt_length: int = 256,
    max_length: int = 512,
    seed: int = 42,
) -> List[Dict]:
    """
    Build a held-out list of K=4 examples from openbmb/UltraFeedback.

    Training split   : shuffle(seed) → first n_train_skip examples
    Held-out split   : shuffle(seed) → examples [n_train_skip, n_train_skip+n_eval)

    Returns a list of dicts, each with:
        prompt, responses (list of 4 texts), scores (list of 4 floats),
        true_ranking (list of 4 ints, rank 0 = best),
        pre_tokenised (list of 4 (input_ids, labels) pairs)
    """
    from datasets import load_dataset

    logger.info("Loading openbmb/UltraFeedback for held-out evaluation ...")
    raw = load_dataset("openbmb/UltraFeedback", split="train")
    shuffled = raw.shuffle(seed=seed)
    held_raw = shuffled.select(range(n_train_skip, n_train_skip + n_eval + 500))

    # Pre-tokenise helper
    def _tok(prompt_text: str, response_text: str) -> Tuple[List[int], List[int]]:
        _enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=True,
        )
        # transformers>=5.x may return BatchEncoding instead of list
        if hasattr(_enc, "input_ids"):
            prompt_ids: List[int] = list(_enc.input_ids)
        else:
            prompt_ids = list(_enc)
        prompt_ids = prompt_ids[:max_prompt_length]
        _resp_enc = tokenizer(response_text, add_special_tokens=False).input_ids
        resp_ids: List[int] = list(_resp_enc)
        max_resp = max(max_length - len(prompt_ids), 1)
        resp_ids = resp_ids[:max_resp] + [tokenizer.eos_token_id]
        input_ids = prompt_ids + resp_ids
        labels = [-100] * len(prompt_ids) + resp_ids
        return input_ids, labels

    examples = []
    for ex in held_raw:
        completions = ex.get("completions", [])
        if len(completions) != 4:
            continue
        try:
            scores = [float(c["overall_score"]) for c in completions]
            responses = [c["response"] for c in completions]
        except (TypeError, ValueError, KeyError):
            continue
        true_ranking = sorted(range(4), key=lambda k: scores[k], reverse=True)
        tokenised = [_tok(ex["instruction"], resp) for resp in responses]
        examples.append({
            "prompt":       ex["instruction"],
            "responses":    responses,
            "scores":       scores,
            "true_ranking": true_ranking,
            "tokenised":    tokenised,
        })
        if len(examples) >= n_eval:
            break

    logger.info("Held-out examples assembled: %d", len(examples))
    return examples


# ---------------------------------------------------------------------------
# Batched forward pass
# ---------------------------------------------------------------------------

def _batch_log_probs(
    model,
    ids_list: List[List[int]],
    lbl_list: List[List[int]],
    device: str,
) -> torch.Tensor:
    """Padded batch forward → per-sequence log-prob sums [N].

    Works with both single-GPU and device_map='auto' multi-GPU models:
    logits are moved to CPU before softmax to avoid device-mismatch errors
    and to keep peak GPU memory low (Qwen-7B vocab ≈ 152 K tokens,
    float32 logits for batch × seq_len can be several GB).
    """
    N      = len(ids_list)
    max_len = max(len(x) for x in ids_list)
    pad_id  = getattr(model.config, "pad_token_id", 0) or 0

    input_ids = torch.full((N, max_len), pad_id, dtype=torch.long, device=device)
    attn      = torch.zeros(N, max_len, dtype=torch.long, device=device)
    labels_t  = torch.full((N, max_len), -100, dtype=torch.long)  # kept on CPU

    for i, (ids, lbl) in enumerate(zip(ids_list, lbl_list)):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        attn[i, :L]      = 1
        labels_t[i, :L]  = torch.tensor(lbl, dtype=torch.long)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn)

    # Move logits to CPU immediately to free GPU memory and avoid device mismatch
    # when model layers are spread across GPUs via device_map="auto".
    logits       = out.logits.cpu().float()                 # [N, L, V]
    shift_logits = logits[:, :-1, :]                        # [N, L-1, V]
    shift_labels = labels_t[:, 1:]                          # [N, L-1]  (already CPU)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    mask      = shift_labels != -100
    gathered  = log_probs.gather(-1, shift_labels.clamp(0).unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(1)                         # [N] on CPU


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

def _top1_acc(true_ranking: List[int], model_ranking: List[int]) -> float:
    return float(true_ranking[0] == model_ranking[0])


def _exact_match(true_ranking: List[int], model_ranking: List[int]) -> float:
    return float(true_ranking == model_ranking)


def _kendall_tau(true_ranking: List[int], model_ranking: List[int]) -> float:
    """Kendall τ between the two ranking permutations (range: −1 to +1)."""
    try:
        from scipy.stats import kendalltau
    except ImportError:
        # Fallback: manual concordant/discordant count
        return _manual_kendall(true_ranking, model_ranking)
    K = len(true_ranking)
    true_rank  = [0] * K
    model_rank = [0] * K
    for pos, idx in enumerate(true_ranking):
        true_rank[idx] = pos
    for pos, idx in enumerate(model_ranking):
        model_rank[idx] = pos
    tau, _ = kendalltau(true_rank, model_rank)
    return float(tau)


def _manual_kendall(true_ranking: List[int], model_ranking: List[int]) -> float:
    K   = len(true_ranking)
    tr  = {idx: pos for pos, idx in enumerate(true_ranking)}
    mr  = {idx: pos for pos, idx in enumerate(model_ranking)}
    concordant = discordant = 0
    for i, j in combinations(range(K), 2):
        tc = (tr[i] < tr[j])
        mc = (mr[i] < mr[j])
        if tc == mc:
            concordant += 1
        else:
            discordant += 1
    n_pairs = K * (K - 1) // 2
    return (concordant - discordant) / n_pairs


def _ndcg(scores: List[float], g_theta: List[float]) -> float:
    """
    NDCG@K using original overall_score as relevance and g_theta as predicted score.

    sklearn.metrics.ndcg_score(y_true, y_score) computes NDCG with the ideal
    ordering derived from y_true.
    """
    try:
        from sklearn.metrics import ndcg_score as _ndcg_fn
        import numpy as np
        y_true  = np.array([scores], dtype=float)
        y_score = np.array([g_theta], dtype=float)
        return float(_ndcg_fn(y_true, y_score))
    except ImportError:
        # Fallback: manual DCG / IDCG
        return _manual_ndcg(scores, g_theta)


def _manual_ndcg(scores: List[float], g_theta: List[float]) -> float:
    K = len(scores)
    import math
    pred_order = sorted(range(K), key=lambda i: g_theta[i], reverse=True)
    true_order = sorted(range(K), key=lambda i: scores[i], reverse=True)
    min_s = min(scores)
    rel   = [max(0.0, s - min_s) for s in scores]
    def dcg(order):
        return sum(rel[idx] / math.log2(rank + 2) for rank, idx in enumerate(order))
    ideal = dcg(true_order)
    if ideal == 0.0:
        return 1.0
    return dcg(pred_order) / ideal


def _pairwise_acc_k4(true_ranking: List[int], g_theta: List[float]) -> float:
    """Fraction of C(4,2)=6 pairwise preferences the model gets correct."""
    K = len(true_ranking)
    correct = 0
    total   = 0
    # true preference: i ≻ j if i appears earlier in true_ranking
    true_pref = {idx: pos for pos, idx in enumerate(true_ranking)}
    for i, j in combinations(range(K), 2):
        total += 1
        # true: i preferred over j
        if true_pref[i] < true_pref[j]:
            if g_theta[i] > g_theta[j]:
                correct += 1
        else:
            if g_theta[j] > g_theta[i]:
                correct += 1
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def compute_ranking_metrics(
    model,
    tokenizer,
    beta: float = 0.1,
    n_eval: int = 500,
    n_train_skip: int = 5000,
    max_length: int = 512,
    max_prompt_length: int = 256,
    eval_batch_size: int = 4,   # sequences per forward pass; use 1 for large models
    device: str = "cuda:0",
    seed: int = 42,
    held_out_examples: Optional[List[Dict]] = None,  # pass to avoid re-loading
) -> Dict[str, float]:
    """
    Evaluate a trained model on the held-out K=4 UltraFeedback split.

    For each example:
      1. Compute g_theta[k] = beta*(log_pi_theta(y_k|x) - log_pi_ref(y_k|x)) for k=0..3
      2. model_ranking = argsort(g_theta, descending)
      3. Compare to true_ranking via 5 metrics

    Returns dict with keys:
        top1_acc, exact_match, kendall_tau, ndcg, pairwise_acc, n_eval
    """
    if held_out_examples is None:
        held_out_examples = build_held_out_k4(
            tokenizer=tokenizer,
            n_train_skip=n_train_skip,
            n_eval=n_eval,
            max_prompt_length=max_prompt_length,
            max_length=max_length,
            seed=seed,
        )

    model.eval()
    N = len(held_out_examples)

    # Flatten all (K=4 * N) sequences into flat lists
    all_ids  = []
    all_lbls = []
    for ex in held_out_examples:
        for ids, lbl in ex["tokenised"]:
            all_ids.append(ids)
            all_lbls.append(lbl)

    # Batched forward: policy
    logger.info("Computing policy log probs on %d×4=%d sequences ...", N, N * 4)
    lp_policy_flat = []
    for start in range(0, len(all_ids), eval_batch_size):
        lp_policy_flat.append(
            _batch_log_probs(model, all_ids[start:start + eval_batch_size],
                             all_lbls[start:start + eval_batch_size], device)
        )
    lp_policy = torch.cat(lp_policy_flat).view(N, 4)  # [N, K=4]

    # Batched forward: reference (adapter OFF)
    logger.info("Computing reference log probs ...")
    lp_ref_flat = []
    with model.disable_adapter():
        for start in range(0, len(all_ids), eval_batch_size):
            lp_ref_flat.append(
                _batch_log_probs(model, all_ids[start:start + eval_batch_size],
                                 all_lbls[start:start + eval_batch_size], device)
            )
    lp_ref = torch.cat(lp_ref_flat).view(N, 4)

    # g_theta [N, K]
    g = beta * (lp_policy - lp_ref)  # [N, 4]
    g_np = g.cpu().float().tolist()

    # Compute per-sample metrics
    top1, exact, tau, ndcg, pw = [], [], [], [], []
    for i, ex in enumerate(held_out_examples):
        gi = g_np[i]
        model_ranking = sorted(range(4), key=lambda k: gi[k], reverse=True)
        top1.append(_top1_acc(ex["true_ranking"], model_ranking))
        exact.append(_exact_match(ex["true_ranking"], model_ranking))
        tau.append(_kendall_tau(ex["true_ranking"], model_ranking))
        ndcg.append(_ndcg(ex["scores"], gi))
        pw.append(_pairwise_acc_k4(ex["true_ranking"], gi))

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    results = {
        "top1_acc":    round(_mean(top1),  4),
        "exact_match": round(_mean(exact), 4),
        "kendall_tau": round(_mean(tau),   4),
        "ndcg":        round(_mean(ndcg),  4),
        "pairwise_acc_k4": round(_mean(pw), 4),
        "n_eval":      N,
    }
    logger.info(
        "ranking_metrics: top1=%.4f exact=%.4f tau=%.4f ndcg=%.4f pairwise_k4=%.4f (n=%d)",
        results["top1_acc"], results["exact_match"], results["kendall_tau"],
        results["ndcg"], results["pairwise_acc_k4"], N,
    )
    model.train()
    return results
