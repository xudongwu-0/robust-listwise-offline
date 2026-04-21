"""
NominalListwiseTrainer / RobustListwiseTrainer —
trains a PEFT/LoRA model with the nominal or robust
Plackett-Luce listwise DPO objective.

Mathematical reference: docs/ROBUST_LISTWISE_DPO_MATH.md §1, §4, §5, §6

Score (§1):
    g_theta(x, y) = beta * log( pi_theta(y|x) / pi_ref(y|x) )

    where pi_theta is the policy (LoRA-adapted model) and
    pi_ref is the reference model (same base weights, adapter disabled).

Nominal loss (§4.2):
    ell_PL = - sum_i g[sigma*_i]
             + sum_i logsumexp(g[sigma*_i:])

Robust loss (§5):
    ell_robust = (1-rho) * ell_PL(sigma_obs) + rho * ell_PL(sigma_wc)
    sigma_wc = argsort(g, ascending=True)   [§6]

Batched data layout:
    input_ids, attention_mask, labels : [B*K, L]
    ranking                           : [B, K]  (rank 0 = best)
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import Trainer

from losses.plackett_luce import plackett_luce_loss, robust_pl_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: per-sequence log probability
# ---------------------------------------------------------------------------

def compute_per_sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the sum of per-token log probabilities for non-masked positions.

    Args:
        logits: [N, L, V]  — model output logits
        labels: [N, L]     — token IDs; prompt positions are -100 (ignored)

    Returns:
        [N] float tensor — sum of log probs over response tokens
    """
    # Shift: logits[t] predicts label[t+1]
    shift_logits = logits[:, :-1, :].contiguous()    # [N, L-1, V]
    shift_labels = labels[:, 1:].contiguous()         # [N, L-1]

    log_probs = F.log_softmax(shift_logits, dim=-1)

    mask = shift_labels != -100
    # Clamp so we can gather safely on masked positions
    indices = shift_labels.clamp(min=0).unsqueeze(-1)   # [N, L-1, 1]
    token_log_probs = log_probs.gather(-1, indices).squeeze(-1)  # [N, L-1]
    token_log_probs = token_log_probs * mask

    return token_log_probs.sum(dim=1)  # [N]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class NominalListwiseTrainer(Trainer):
    """
    Extends HuggingFace Trainer to compute the nominal PL listwise DPO loss.

    Expects:
      - model: a PEFT PeftModel (LoRA); the reference model is obtained by
               disabling the adapter (model.disable_adapter()).
      - Data collated by ListwiseCollator (see src/data/ultrafeedback_listwise.py).
    """

    def __init__(self, *args, beta: float = 0.1, K: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.K = K

    # ------------------------------------------------------------------
    # Core loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        1. Forward pass through the policy model (adapter ON).
        2. Forward pass through the reference (adapter OFF, no grad).
        3. Compute g_theta scores = beta * (log_pi - log_pi_ref).
        4. Reorder scores by the ground-truth ranking.
        5. Compute PL loss.
        """
        input_ids      = inputs["input_ids"]       # [B*K, L]
        attention_mask = inputs["attention_mask"]  # [B*K, L]
        labels         = inputs["labels"]          # [B*K, L]
        ranking        = inputs["ranking"]         # [B, K]
        B = ranking.size(0)
        K = self.K

        # ---- Policy forward (with gradient) ----
        policy_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        log_probs = compute_per_sequence_log_probs(
            policy_out.logits, labels
        )  # [B*K]

        # ---- Reference forward (adapter disabled, no gradient) ----
        with torch.no_grad():
            with model.disable_adapter():
                ref_out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        ref_log_probs = compute_per_sequence_log_probs(
            ref_out.logits.detach(), labels
        )  # [B*K]

        # ---- g_theta(x, y) = beta * (log_pi_theta - log_pi_ref) ----
        g = self.beta * (log_probs - ref_log_probs)  # [B*K]
        g = g.view(B, K)                             # [B, K]

        # ---- Reorder by ground-truth ranking ----
        # g_ranked[b, i] = g[b, ranking[b, i]]
        # ranking[b, 0] is the index of the best response
        g_ranked = g.gather(1, ranking)  # [B, K]

        # ---- Plackett-Luce loss ----
        loss = plackett_luce_loss(g_ranked)

        if return_outputs:
            return loss, {"g": g, "g_ranked": g_ranked}
        return loss

    # ------------------------------------------------------------------
    # Logging: add reward margin to progress bar
    # ------------------------------------------------------------------

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        # transformers >= 4.47 passes start_time; be forwards-compatible
        try:
            super().log(logs, start_time=start_time)
        except TypeError:
            super().log(logs)


# ---------------------------------------------------------------------------
# RobustListwiseTrainer
# ---------------------------------------------------------------------------

class RobustListwiseTrainer(Trainer):
    """
    Extends HuggingFace Trainer to compute the robust PL listwise DPO loss.

    Loss (§5 of ROBUST_LISTWISE_DPO_MATH.md):
        ell_robust = (1-rho) * ell_PL(sigma_obs) + rho * ell_PL(sigma_wc)

    where sigma_wc is the worst-case ranking = ascending-score order (§6).

    Setting rho=0 recovers the nominal listwise loss exactly.
    Setting rho=1 trains entirely against the worst-case ranking.

    Usage is identical to NominalListwiseTrainer; add rho= to the constructor.
    """

    def __init__(self, *args, beta: float = 0.1, K: int = 4, rho: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.K = K
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho must be in [0, 1], got {rho}")
        self.rho = rho

    def _compute_g(self, model, input_ids, attention_mask, labels, B, K):
        """Shared forward pass producing g_theta scores [B, K]."""
        # Policy forward (gradient flows through here)
        policy_out = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = compute_per_sequence_log_probs(policy_out.logits, labels)

        # Reference forward (adapter OFF, no gradient)
        with torch.no_grad():
            with model.disable_adapter():
                ref_out = model(input_ids=input_ids, attention_mask=attention_mask)
        ref_log_probs = compute_per_sequence_log_probs(ref_out.logits.detach(), labels)

        g = self.beta * (log_probs - ref_log_probs)  # [B*K]
        return g.view(B, K)                           # [B, K]

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute the robust listwise DPO loss.

        Steps:
          1. Single forward pass → g [B, K] (scores in original response order).
          2. Nominal component : reorder g by observed ranking sigma_obs → PL loss.
          3. Worst-case component: reorder g by sigma_wc (ascending) → PL loss.
          4. Combined: (1-rho)*nominal + rho*worst_case.
        """
        input_ids      = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels         = inputs["labels"]
        ranking        = inputs["ranking"]       # [B, K]  sigma_obs
        B = ranking.size(0)
        K = self.K

        g = self._compute_g(model, input_ids, attention_mask, labels, B, K)

        loss = robust_pl_loss(g, ranking, self.rho)

        if return_outputs:
            return loss, {"g": g, "rho": self.rho}
        return loss

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        try:
            super().log(logs, start_time=start_time)
        except TypeError:
            super().log(logs)


# ---------------------------------------------------------------------------
# BTListwiseTrainer  (Bradley-Terry on K=4 data — top-vs-bottom pair)
# ---------------------------------------------------------------------------

class BTListwiseTrainer(Trainer):
    """
    Bradley-Terry (pairwise DPO) trained on K=4 listwise data.

    Extracts only the top-ranked and bottom-ranked responses from the observed
    ranking and trains with the standard pairwise DPO / BT loss:

        ell_BT = -log σ(g[ranking[0]] - g[ranking[K-1]])
               = softplus(g[ranking[K-1]] - g[ranking[0]])

    This makes the BT model directly comparable to PL models in noise sweep
    experiments because all three trainers consume the same K=4 batches
    (including the same noisy ranking signal).

    Data collated by ListwiseCollator (identical to the PL trainers).
    """

    def __init__(self, *args, beta: float = 0.1, K: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.K = K

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **kwargs,
    ):
        input_ids      = inputs["input_ids"]       # [B*K, L]
        attention_mask = inputs["attention_mask"]  # [B*K, L]
        labels         = inputs["labels"]          # [B*K, L]
        ranking        = inputs["ranking"]         # [B, K]
        B = ranking.size(0)
        K = self.K

        # ---- Policy forward ----
        policy_out = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = compute_per_sequence_log_probs(policy_out.logits, labels)

        # ---- Reference forward (adapter OFF, no grad) ----
        with torch.no_grad():
            with model.disable_adapter():
                ref_out = model(input_ids=input_ids, attention_mask=attention_mask)
        ref_log_probs = compute_per_sequence_log_probs(ref_out.logits.detach(), labels)

        # ---- g_theta [B, K] ----
        g_flat = self.beta * (log_probs - ref_log_probs)  # [B*K]
        g = g_flat.view(B, K)                              # [B, K]

        # ---- Extract top and bottom scores ----
        # ranking[:, 0]  = index of best response per batch element
        # ranking[:, -1] = index of worst response per batch element
        g_best  = g.gather(1, ranking[:, 0:1]).squeeze(1)   # [B]
        g_worst = g.gather(1, ranking[:, -1:]).squeeze(1)   # [B]

        # ---- Bradley-Terry / pairwise DPO loss ----
        # loss = -log σ(g_best - g_worst) = softplus(g_worst - g_best)
        loss = F.softplus(g_worst - g_best).mean()

        if return_outputs:
            return loss, {"g": g, "g_best": g_best, "g_worst": g_worst}
        return loss

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        try:
            super().log(logs, start_time=start_time)
        except TypeError:
            super().log(logs)
