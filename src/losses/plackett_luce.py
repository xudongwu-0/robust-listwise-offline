"""
Plackett-Luce losses for nominal and robust listwise DPO.

Mathematical reference: docs/ROBUST_LISTWISE_DPO_MATH.md §4, §5, §6

Score definition (§1):
    g_theta(x, y) = beta * log( pi_theta(y|x) / pi_ref(y|x) )

Nominal listwise PL loss (§4.2):
    ell_PL = - sum_{i=1}^K g[sigma*_i]
             + sum_{i=1}^K log( sum_{j=i}^K exp(g[sigma*_j]) )

Worst-case ranking (§6):
    sigma_wc = argsort(g, ascending=True)
    (highest-scoring candidate is placed LAST)

Robust listwise loss (§5):
    ell_robust = (1-rho) * ell_PL(sigma_obs) + rho * ell_PL(sigma_wc)
"""

import torch


def plackett_luce_loss(scores_ranked: torch.Tensor) -> torch.Tensor:
    """
    Compute the nominal Plackett-Luce loss for a batch.

    Args:
        scores_ranked: FloatTensor of shape [B, K]
            scores_ranked[b, i] = g_theta(x_b, y_{sigma*_b[i]})
            i.e. the scores already reordered by the ground-truth ranking,
            with index 0 being the best-ranked response.

    Returns:
        Scalar: mean PL loss over the batch.

    Implementation follows §4.2 exactly:
        ell_PL = - sum_i scores_ranked[:, i]
                 + sum_i logsumexp(scores_ranked[:, i:], dim=1)

    Sanity check (K=2):
        With scores_ranked = [g_best, g_worst]:
        ell_PL = -(g_best + g_worst)
                 + logsumexp([g_best, g_worst])
                 + g_worst
               = -g_best + log(exp(g_best) + exp(g_worst))
               = log(1 + exp(g_worst - g_best))
               = -log sigma(g_best - g_worst)
               = pairwise DPO loss  ✓
    """
    B, K = scores_ranked.shape

    # suffix_lse[b, i] = log( sum_{j=i}^{K-1} exp(scores_ranked[b, j]) )
    # Corresponds to the denominator at position i in the PL product.
    suffix_lse = torch.stack(
        [torch.logsumexp(scores_ranked[:, i:], dim=1) for i in range(K)],
        dim=1,
    )  # [B, K]

    # ell_PL = - sum_i g[sigma*_i]  +  sum_i log(sum_{j>=i} exp(g[sigma*_j]))
    per_sample_loss = (
        -scores_ranked.sum(dim=1) + suffix_lse.sum(dim=1)
    )  # [B]

    return per_sample_loss.mean()


def worst_case_ranking(g: torch.Tensor) -> torch.Tensor:
    """
    Compute the worst-case ranking under PL loss (§6 of ROBUST_LISTWISE_DPO_MATH.md).

    The worst-case ranking places the candidate with the HIGHEST current score
    at the LAST position, i.e. it is the ascending-score order.

    Args:
        g: FloatTensor [B, K]  — current DPO scores (unordered)

    Returns:
        LongTensor [B, K]  — sigma_wc[b] is a permutation of [0..K-1]
                             sigma_wc[b, 0] = index of the lowest-scored candidate
                             sigma_wc[b, K-1] = index of the highest-scored candidate

    Note: do NOT sort the scores themselves; return the INDEX permutation so the
    caller can use gather() to reorder g consistently with how the nominal
    ranking is used.
    """
    # argsort ascending: position 0 = index of smallest score
    return torch.argsort(g, dim=1, descending=False)  # [B, K]


def robust_pl_loss(
    g: torch.Tensor,
    ranking_obs: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    """
    Compute the robust listwise DPO loss (§5 of ROBUST_LISTWISE_DPO_MATH.md).

        ell_robust = (1-rho) * ell_PL(sigma_obs) + rho * ell_PL(sigma_wc)

    Args:
        g          : FloatTensor [B, K]  — DPO scores g_theta(x, y_k) in original order
        ranking_obs: LongTensor  [B, K]  — observed ranking (rank 0 = best)
        rho        : float in [0, 1]     — robustness coefficient

    Returns:
        Scalar loss.

    Special cases:
        rho=0  → exactly nominal PL loss (no worst-case component)
        rho=1  → entirely worst-case PL loss
    """
    # Reorder by observed ranking
    g_obs = g.gather(1, ranking_obs)           # [B, K]
    loss_nominal = plackett_luce_loss(g_obs)   # scalar

    if rho == 0.0:
        return loss_nominal

    # Worst-case ranking and reordered scores
    sigma_wc = worst_case_ranking(g)           # [B, K]  ascending order
    g_wc = g.gather(1, sigma_wc)              # [B, K]
    loss_wc = plackett_luce_loss(g_wc)        # scalar

    return (1.0 - rho) * loss_nominal + rho * loss_wc
