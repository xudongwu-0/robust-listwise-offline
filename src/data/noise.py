"""
Ranking noise injection for controlled corruption experiments.

Two noise types:

  near_tie  — swap the adjacent pair in the ranking with the smallest score gap.
              Simulates annotation noise where raters cannot distinguish
              near-equal responses.

  top_rank  — move a uniformly-random non-top response into the rank-1 slot.
              Simulates systematic error in identifying the best response.

Noise level (noise_prob ∈ [0, 1]) is the per-sample probability of applying
the perturbation.  noise_prob=0 → clean labels; noise_prob=1 → always corrupt.

Usage:
    noise_fn = make_noise_fn("near_tie", noise_prob=0.4, seed=42)
    dataset  = build_listwise_dataset(..., noise_fn=noise_fn)
"""

import random
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Internal noise functions
# ---------------------------------------------------------------------------

def _near_tie_noise(
    ranking: List[int],
    scores: List[float],
    rng: random.Random,
    noise_prob: float,
) -> List[int]:
    """
    Find the adjacent pair in the ranking with the smallest score gap and
    swap them with probability noise_prob.

    ranking : [best_idx, ..., worst_idx]
    scores  : raw overall_score indexed by response index (not by rank)
    """
    if rng.random() >= noise_prob:
        return ranking

    K = len(ranking)
    min_gap = float("inf")
    swap_pos = 0  # position in ranking to swap with swap_pos+1

    for i in range(K - 1):
        gap = scores[ranking[i]] - scores[ranking[i + 1]]  # >= 0 by construction
        if gap < min_gap:
            min_gap = gap
            swap_pos = i

    noisy = list(ranking)
    noisy[swap_pos], noisy[swap_pos + 1] = noisy[swap_pos + 1], noisy[swap_pos]
    return noisy


def _top_rank_noise(
    ranking: List[int],
    scores: List[float],
    rng: random.Random,
    noise_prob: float,
) -> List[int]:
    """
    Swap rank-1 with a uniformly-random other response with probability noise_prob.

    ranking : [best_idx, ..., worst_idx]
    scores  : unused (kept for uniform interface)
    """
    if rng.random() >= noise_prob:
        return ranking

    K = len(ranking)
    swap_pos = rng.randint(1, K - 1)  # pick any position except 0
    noisy = list(ranking)
    noisy[0], noisy[swap_pos] = noisy[swap_pos], noisy[0]
    return noisy


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def make_noise_fn(
    noise_type: str,
    noise_prob: float,
    seed: int = 42,
) -> Optional[Callable[[List[int], List[float]], List[int]]]:
    """
    Return a callable noise_fn(ranking, scores) -> noisy_ranking.

    Returns None if noise_prob == 0 (no noise, avoids unnecessary overhead).

    Args:
        noise_type : "near_tie" | "top_rank"
        noise_prob : probability in [0, 1] of corrupting each sample
        seed       : random seed for reproducibility
    """
    if noise_prob == 0.0:
        return None

    if noise_type not in ("near_tie", "top_rank"):
        raise ValueError(f"Unknown noise_type: {noise_type!r}. Choose 'near_tie' or 'top_rank'.")

    rng = random.Random(seed)

    if noise_type == "near_tie":
        def fn(ranking: List[int], scores: List[float]) -> List[int]:
            return _near_tie_noise(ranking, scores, rng, noise_prob)
    else:
        def fn(ranking: List[int], scores: List[float]) -> List[int]:
            return _top_rank_noise(ranking, scores, rng, noise_prob)

    return fn


# ---------------------------------------------------------------------------
# Sanity checks (importable for use in scripts)
# ---------------------------------------------------------------------------

def verify_noise_functions() -> None:
    """Lightweight CPU sanity checks — call at script startup."""
    import logging
    log = logging.getLogger(__name__)

    # near_tie: adjacent pair with min gap must be swapped
    ranking = [2, 0, 3, 1]           # scores[2]=10, scores[0]=8, scores[3]=8.1, scores[1]=5
    scores  = [8.0, 5.0, 10.0, 8.1]  # gap between pos 1 (idx 0, s=8.0) and pos 2 (idx 3, s=8.1) is 0.1
    # Smallest gap: scores[ranking[1]] - scores[ranking[2]] = 8.0 - 8.1 ... wait let me recalculate
    # ranking[0]=2 → score=10; ranking[1]=3 → score=8.1; ranking[2]=0 → score=8.0; ranking[3]=1 → score=5
    # gaps: [10-8.1=1.9, 8.1-8.0=0.1, 8.0-5=3.0] → min gap at pos 1 (swap pos 1 and 2)
    fn_nt = make_noise_fn("near_tie", noise_prob=1.0, seed=0)
    noisy = fn_nt(ranking, scores)
    assert noisy[1] == ranking[2] and noisy[2] == ranking[1], (
        f"near_tie should swap positions 1 and 2, got {noisy}"
    )
    log.info("noise SANITY: near_tie swap at min-gap position OK")

    # top_rank: rank-1 must change (noise_prob=1.0)
    fn_tr = make_noise_fn("top_rank", noise_prob=1.0, seed=0)
    ranking2 = [3, 0, 1, 2]
    noisy2 = fn_tr(ranking2, [])
    assert noisy2[0] != ranking2[0], f"top_rank should change rank-1, got {noisy2}"
    log.info("noise SANITY: top_rank changes rank-1 OK")

    # noise_prob=0 → no change
    assert make_noise_fn("near_tie", noise_prob=0.0) is None
    log.info("noise SANITY: noise_prob=0 returns None OK")
