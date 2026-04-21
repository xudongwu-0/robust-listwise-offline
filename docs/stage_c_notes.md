# Stage C — Robust Listwise DPO: Implementation Notes

## 1. Robust loss formula (§5 of ROBUST_LISTWISE_DPO_MATH.md)

```
ell_robust(g, sigma_obs) = (1 - rho) * ell_PL(g, sigma_obs)
                          +      rho  * ell_PL(g, sigma_wc)
```

where:
- `g = [g_1, ..., g_K]` are the current DPO scores for each candidate
- `sigma_obs` is the observed ground-truth ranking (from UltraFeedback `overall_score`)
- `sigma_wc` is the worst-case ranking (see §2 below)
- `rho ∈ [0, 1]` is the robustness coefficient

Setting **rho=0** recovers the nominal listwise loss exactly.

---

## 2. Worst-case ranking (§6)

Per §6 of the math document, the worst-case ranking under the PL loss is:

```
sigma_wc = argsort(g, ascending=True)
```

Interpretation: the candidate with the **highest** current score is placed **last** in
the ranking. This is the permutation that maximises the PL loss given the current scores.

Implementation in `src/losses/plackett_luce.py`:

```python
def worst_case_ranking(g: torch.Tensor) -> torch.Tensor:
    return torch.argsort(g, dim=1, descending=False)  # [B, K]
```

No permutation search is needed. The worst-case is always the ascending-score order.

---

## 3. How the robust loss is computed in the trainer (one training step)

1. **Forward pass (policy)**: run the LoRA model on all `B*K` sequences → logits
2. **Forward pass (reference)**: run the same model with `model.disable_adapter()` → ref logits
3. **Compute g**: `g = beta * (log_pi_theta - log_pi_ref)`, reshape to `[B, K]`
4. **Nominal component**: reorder `g` by `sigma_obs` (ground-truth ranking) → `ell_PL(g_obs)`
5. **Worst-case component**: reorder `g` by ascending order → `ell_PL(g_wc)`
6. **Combine**: `(1-rho) * ell_PL(g_obs) + rho * ell_PL(g_wc)`
7. **Backward + step**: as usual

---

## 4. Sanity checks included in the training script

| # | Check | Expected |
|---|-------|---------|
| 1 | `robust_pl_loss(rho=0)` == `plackett_luce_loss` | delta < 1e-6 |
| 2 | `worst_case_ranking(g)` is ascending-score order | last index = argmax(g) |
| 3 | `robust_pl_loss(rho=1)` == `plackett_luce_loss(sigma_wc)` | delta < 1e-6 |
| 4 | Aligned scores: `nominal_loss < worst_case_loss` | direction check |

---

## 5. Files changed / created in Stage C

| File | Change |
|------|--------|
| `src/losses/plackett_luce.py` | Added `worst_case_ranking()` and `robust_pl_loss()` |
| `src/trainers/listwise_trainer.py` | Added `RobustListwiseTrainer` class |
| `src/scripts/train_robust_listwise.py` | New training script with `--rho` argument |

---

## 6. What still needs to be done (Stage D onward)

### Controlled ranking noise experiment
- Inject controlled label corruption before training:
  - **near_tie**: randomly swap adjacent-score responses (small score difference)
  - **top_rank**: randomly assign a different response to rank-1 position
- Noise levels: `noise=0.0`, `0.4`, `1.0`
- Compare `nominal_listwise` vs `robust_listwise` on **clean** held-out labels

### Evaluation integration
1. **Held-out UltraFeedback**: use `test_prefs` from `HuggingFaceH4/ultrafeedback_binarized`
   - Metric: pairwise accuracy (does the model prefer chosen over rejected?)
2. **RewardBench**: run the trained adapter through the RewardBench evaluation pipeline

### Scale-up
- Once all comparisons are validated on the 1 K-sample subset:
  - increase to 10 K–60 K training samples
  - run multiple seeds
  - produce comparison plots (noise level vs eval accuracy)
