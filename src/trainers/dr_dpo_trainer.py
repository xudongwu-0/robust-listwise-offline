"""TV-DR-DPO: Distributionally Robust DPO with TV ambiguity set.

Reference: Mandal, Sasnauskas & Radanovic (2025), Algorithm 3 (DR-DPO).

Design choices for direct comparability with the Nominal BT/DPO baseline
(BTListwiseTrainer):
  * Pair construction: ONE top-vs-bottom pair per K=4 list (identical to
    BTListwiseTrainer). No additional listwise supervision is injected.
  * DRO grouping: Because per-device micro-batch may be 1 (8B), we cannot
    apply TV reweighting within a single micro-batch. Instead we accumulate
    m_dro top-vs-bottom losses across consecutive micro-batches, compute a
    TV worst-case reweighting q* over those m losses, and back-propagate
    the weighted robust loss as ONE optimizer step.

For each minibatch with per-list top-vs-bottom losses {ell_i}_{i=1..m_dro}:

    q* = argmax_{q in Delta^m, TV(q, 1/m) <= rho}  sum_i q_i * ell_i

    L_robust = sum_i q*_i * ell_i

The TV constraint is TV(q, u) = 0.5 * sum_i |q_i - 1/m|.

Implementation:
  Override `training_step` to manage a small buffer of `m_dro` micro-batches.
  - Calls 1 .. m_dro-1: buffer the inputs and return a zero scalar (no
    backward). Configure the Trainer with `gradient_accumulation_steps =
    m_dro` so the optimizer step fires only after the m_dro-th call.
  - Call m_dro: do a 2-pass DRO update:
       Pass 1 (no_grad): forward each buffered micro-batch, compute its
                         top-vs-bottom pair loss; stack the m losses and
                         compute q* (detached).
       Pass 2:           forward each buffered micro-batch with grad,
                         backward q*_i * ell_i incrementally (this avoids
                         holding m forward graphs simultaneously).
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import Trainer

from trainers.listwise_trainer import compute_per_sequence_log_probs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TV worst-case reweighting (closed form)
# ---------------------------------------------------------------------------

@torch.no_grad()
def kl_softmax_weights(losses: torch.Tensor, tau: float) -> torch.Tensor:
    """KLDPO worst-case weights via tempered softmax of centered losses.

    Approximates the worst-case re-weighting under a KL ambiguity set
    around the uniform distribution (Mandal et al. 2025 / Xu et al. 2025
    KLDPO Algorithm 2):

        w_i ∝ exp((l_i - mean(l)) / tau)

    Centering by mean(l) is optional for the math (it cancels out in the
    softmax) but improves numerical stability. We add the standard
    log-sum-exp shift on top of that for full safety against overflow.

    Args:
        losses: 1-D tensor [m] of per-example losses (will be detached).
        tau: temperature, > 0. Large tau → uniform; small tau → puts most
            mass on the largest loss.

    Returns:
        w: 1-D tensor [m], same dtype/device as losses, w.sum()=1, w>=0.
    """
    if losses.dim() != 1:
        raise ValueError(f"losses must be 1-D, got shape {tuple(losses.shape)}")
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    m = losses.numel()
    if m == 1:
        return torch.ones(1, dtype=losses.dtype, device=losses.device)

    losses_d = losses.detach()
    centered = losses_d - losses_d.mean()
    logits = centered / tau
    # Standard numerically-stable softmax.
    logits = logits - logits.max()
    w = torch.softmax(logits, dim=0)
    # Numerical safety.
    w = w.clamp_min(0.0)
    w = w / w.sum()
    return w


@torch.no_grad()
def tv_worst_case_weights(losses: torch.Tensor, rho: float) -> torch.Tensor:
    """Return q* maximizing sum_i q_i * losses_i s.t. q in Delta^m, TV(q,1/m)<=rho.

    Closed-form solution: starting from the uniform 1/m, drain probability
    mass `rho` from the smallest losses (cap u=1/m per coordinate, water-
    fill from the bottom) and place all of that drained mass on the single
    largest loss. This is L1/TV-optimal because the sorted losses are
    monotone and the inner product is linear in q.

    TV(q, 1/m) = 0.5 * sum_i |q_i - 1/m| equals exactly the mass moved
    (`rho` here, when the drain is fully feasible).

    Args:
        losses: 1-D tensor [m] of per-example losses (will be detached).
        rho: TV radius, 0 <= rho <= 1.

    Returns:
        q: 1-D tensor [m], same dtype/device as losses, q.sum()=1, q>=0,
           TV(q, 1/m) <= rho (= rho when m >= 2 and rho < 1).
    """
    if losses.dim() != 1:
        raise ValueError(f"losses must be 1-D, got shape {tuple(losses.shape)}")
    m = losses.numel()
    device = losses.device
    dtype = losses.dtype
    u = 1.0 / m

    if m == 1 or rho <= 0.0:
        return torch.full((m,), u, dtype=dtype, device=device)

    losses_d = losses.detach()
    sorted_losses, sort_idx = torch.sort(losses_d, descending=True)

    # Drain `rho` of mass from the bottom up (each coord caps at u).
    mass_to_move = float(rho)
    drained = torch.zeros(m, dtype=dtype, device=device)
    full_zero_count = int(mass_to_move // u)
    full_zero_count = min(full_zero_count, m - 1)  # keep at least one nonzero
    if full_zero_count > 0:
        drained[m - full_zero_count : m] = u
    remaining = mass_to_move - full_zero_count * u
    leftover_idx = m - full_zero_count - 1
    if remaining > 0 and leftover_idx >= 0:
        drained[leftover_idx] = remaining

    # All drained mass goes to the single largest loss (rank 0 in sorted).
    q_sorted = torch.full((m,), u, dtype=dtype, device=device) - drained
    q_sorted[0] = q_sorted[0] + mass_to_move

    q = torch.empty_like(losses_d)
    q[sort_idx] = q_sorted

    # Numerical safety.
    q = q.clamp_min(0.0)
    q = q / q.sum()
    return q


# ---------------------------------------------------------------------------
# DRDPOTrainer (top-vs-bottom + cross-microbatch TV-DRO)
# ---------------------------------------------------------------------------

class DRDPOTrainer(Trainer):
    """TV-DR-DPO baseline.

    Compatibility with HF Trainer:
        Configure with `gradient_accumulation_steps = m_dro` so that the
        optimizer step fires once per DRO group. Per-device batch size B
        may be > 1; in that case each micro-batch contributes B top-vs-
        bottom losses and the buffer is trimmed to exactly m_dro before
        the worst-case reweighting (typically B=1 is used to keep the
        accounting simple).

    Math:
        For each list b, let sigma_obs[b, 0] be the index of the top
        observed response and sigma_obs[b, K-1] the bottom. The per-list
        BT loss is
            ell_b = softplus(g[b, sigma_obs[b, K-1]] - g[b, sigma_obs[b, 0]])
        with g = beta * (log pi_theta - log pi_ref). Identical to
        BTListwiseTrainer; only the aggregation across lists changes.
    """

    def __init__(
        self,
        *args,
        beta: float = 0.1,
        K: int = 4,
        rho: float = 0.1,
        m_dro: int = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.K = K
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho must be in [0, 1], got {rho}")
        self.rho = rho
        if m_dro < 1:
            raise ValueError(f"m_dro must be >= 1, got {m_dro}")
        self.m_dro = m_dro
        # Buffer of pending micro-batches awaiting DRO aggregation.
        self._dro_inputs_buffer: list = []
        self._last_q_max: float = float("nan")
        self._last_q_min: float = float("nan")

    # ------------------------------------------------------------------
    # DRO-weight hook. Subclasses override this to implement other
    # ambiguity sets (e.g. KL). Default = TV worst-case weights.
    # ------------------------------------------------------------------
    def _dro_weights(self, losses: torch.Tensor) -> torch.Tensor:
        return tv_worst_case_weights(losses, self.rho)

    def _log_extra(self, logs: dict) -> None:
        logs.setdefault("dro/q_max", round(self._last_q_max, 4))
        logs.setdefault("dro/q_min", round(self._last_q_min, 4))
        logs.setdefault("dro/rho", self.rho)
        logs.setdefault("dro/m", self.m_dro)

    # ------------------------------------------------------------------
    # Pair-loss computation for one micro-batch.
    # ------------------------------------------------------------------
    def _pair_losses(self, model, inputs) -> torch.Tensor:
        """Return [B] top-vs-bottom per-list BT losses for one micro-batch."""
        input_ids      = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels         = inputs["labels"]
        ranking        = inputs["ranking"]
        B = ranking.size(0)
        K = self.K

        # Policy forward.
        policy_out = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = compute_per_sequence_log_probs(policy_out.logits, labels)

        # Reference forward (LoRA disabled).
        _base = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            with _base.disable_adapter():
                ref_out = model(input_ids=input_ids, attention_mask=attention_mask)
        ref_log_probs = compute_per_sequence_log_probs(ref_out.logits.detach(), labels)

        g = (self.beta * (log_probs - ref_log_probs)).view(B, K)

        # Top-vs-bottom (identical to BTListwiseTrainer).
        g_best  = g.gather(1, ranking[:, 0:1]).squeeze(1)
        g_worst = g.gather(1, ranking[:, -1:]).squeeze(1)
        ell = F.softplus(g_worst - g_best)  # [B]
        return ell

    # ------------------------------------------------------------------
    # Override training_step to manage the cross-microbatch DRO group.
    # ------------------------------------------------------------------
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        self._dro_inputs_buffer.append(inputs)

        # If the buffer hasn't reached m_dro micro-batches yet, return a
        # zero scalar without invoking backward. The HF Trainer will record
        # this as the loss for the micro-batch and will not call
        # optimizer.step() because gradient_accumulation_steps == m_dro.
        if len(self._dro_inputs_buffer) < self.m_dro:
            return torch.zeros((), device=self.args.device)

        buffered = self._dro_inputs_buffer
        self._dro_inputs_buffer = []

        # ----- Pass 1: compute losses (no_grad) and TV worst-case weights.
        with torch.no_grad():
            ell_chunks = [self._pair_losses(model, inp) for inp in buffered]
        ell_vec = torch.cat(ell_chunks, dim=0)  # [sum_B]
        if ell_vec.numel() > self.m_dro:
            ell_vec = ell_vec[: self.m_dro]
        m_eff = ell_vec.numel()
        q = self._dro_weights(ell_vec)  # [m_eff], detached

        # Map q back to per-microbatch slices (in order).
        q_chunks = []
        offset = 0
        for chunk in ell_chunks:
            n = chunk.numel()
            take = max(0, min(n, m_eff - offset))
            if take == 0:
                q_chunks.append(torch.zeros(n, dtype=q.dtype, device=q.device))
            else:
                pad = n - take
                if pad > 0:
                    q_chunk = torch.cat(
                        [q[offset : offset + take],
                         torch.zeros(pad, dtype=q.dtype, device=q.device)],
                        dim=0,
                    )
                else:
                    q_chunk = q[offset : offset + take]
                q_chunks.append(q_chunk)
                offset += take

        # ----- Pass 2: weighted backward, one micro-batch at a time.
        total_loss = torch.zeros((), device=self.args.device)
        for inp, q_chunk in zip(buffered, q_chunks):
            ell = self._pair_losses(model, inp)            # [B], with grad
            loss_chunk = (q_chunk.detach() * ell).sum()
            self.accelerator.backward(loss_chunk)
            total_loss = total_loss + loss_chunk.detach()

        self._last_q_max = float(q.max().item())
        self._last_q_min = float(q.min().item())

        # `total_loss` already equals the true robust DRO objective
        # sum_i q_i * ell_i (gradients have been back-propagated already by
        # the per-microbatch self.accelerator.backward calls above). The
        # returned scalar is used by HF Trainer for logging only.
        return total_loss

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        if self._last_q_max == self._last_q_max:  # NaN check
            self._log_extra(logs)
        try:
            super().log(logs, start_time=start_time)
        except TypeError:
            super().log(logs)


# ---------------------------------------------------------------------------
# KLDPOTrainer (top-vs-bottom + cross-microbatch KL-DRO via softmax)
# ---------------------------------------------------------------------------

class KLDPOTrainer(DRDPOTrainer):
    """KLDPO baseline: same plumbing as DRDPOTrainer but with softmax
    worst-case re-weighting under a KL ambiguity set.

    For each m_dro-sized buffer of top-vs-bottom DPO/BT losses {ell_i},
    weights are

        w_i = softmax((ell_i - mean(ell)) / tau)

    and the back-propagated loss is sum_i w_i * ell_i.

    `tau` is the KLDPO temperature: large tau → uniform (recovers nominal
    DPO); small tau → emphasize the largest losses.
    """

    def __init__(self, *args, tau: float = 0.1, **kwargs):
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau}")
        # Pass rho=0 (unused) to the parent; we override _dro_weights.
        kwargs.setdefault("rho", 0.0)
        super().__init__(*args, **kwargs)
        self.tau = tau

    def _dro_weights(self, losses: torch.Tensor) -> torch.Tensor:
        return kl_softmax_weights(losses, self.tau)

    def _log_extra(self, logs: dict) -> None:
        logs.setdefault("dro/q_max", round(self._last_q_max, 4))
        logs.setdefault("dro/q_min", round(self._last_q_min, 4))
        logs.setdefault("dro/tau", self.tau)
        logs.setdefault("dro/m", self.m_dro)
