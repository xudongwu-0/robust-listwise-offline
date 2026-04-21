# Stage B — Fixed-List Nominal Listwise DPO: Implementation Notes

## 1. How fixed-list samples are built

Source dataset: `openbmb/UltraFeedback` (train split, ~64 K prompts).

Each prompt in the dataset comes with **exactly 4 completions**, each annotated
with an `overall_score` (float).

A training sample is one (prompt, list-of-4-responses) tuple:

```
prompt      : instruction text
responses   : [y_0, y_1, y_2, y_3]          # original indexing
scores      : [s_0, s_1, s_2, s_3]          # overall_score per completion
ranking     : argsort(scores, descending)    # e.g. [2, 1, 3, 0]
```

`ranking[i]` is the *response index* that occupies rank `i` (rank 0 = best).
This is sigma* from the math document.

---

## 2. How the score g_theta is computed

Per §1 of `ROBUST_LISTWISE_DPO_MATH.md`:

```
g_theta(x, y) = beta * log( pi_theta(y|x) / pi_ref(y|x) )
```

In practice:

```python
log_pi_theta = sum_t log p_theta(y_t | x, y_{<t})   # response tokens only
log_pi_ref   = same, but with LoRA adapter DISABLED  # frozen base model
g = beta * (log_pi_theta - log_pi_ref)
```

The reference model is obtained cheaply by calling
`model.disable_adapter()` (PEFT context manager), which temporarily zeroes
out all LoRA delta weights and runs the frozen base.

---

## 3. How the nominal PL loss is computed

Per §4.2 of `ROBUST_LISTWISE_DPO_MATH.md`:

```
ell_PL = - sum_{i=0}^{K-1} g_ranked[i]
         + sum_{i=0}^{K-1} logsumexp(g_ranked[i:])
```

where `g_ranked[i] = g_theta(x, y_{sigma*[i]})` (score of the response at rank `i`).

The `logsumexp` over the suffix is the log of the Plackett-Luce denominator
at position `i`: it represents the "competition" among all remaining candidates.

**K=2 equivalence check:**

With `g_ranked = [g_best, g_worst]`:
```
ell_PL = -(g_best + g_worst) + logsumexp([g_best, g_worst]) + g_worst
       = -g_best + log(exp(g_best) + exp(g_worst))
       = log(1 + exp(g_worst - g_best))
       = -log sigma(g_best - g_worst)
       = pairwise DPO loss  ✓
```

---

## 4. Sanity checks included in the training script

| # | Check | Where |
|---|-------|-------|
| 1 | K=2 PL loss == pairwise DPO loss | `_sanity_check_pl_k2()` |
| 2 | Score direction: agreeing scores give lower loss | `_sanity_check_score_direction()` |
| 3 | All K responses in a sample share the same prompt (ranking is a valid permutation) | `_sanity_check_grouping()` |
| 4 | Ranking is derived from `overall_score` (descending) | `build_listwise_dataset()` |
| 5 | Training runs end-to-end and loss decreases | training loop |
| 6 | Checkpoint can be saved | `trainer.save_model()` |

---

## 5. What still needs to be done before robust listwise DPO (Stage C)

1. **Worst-case ranking** — implement `sigma_wc = argsort(g, ascending=True)` (§6).
2. **Robust loss** — combine nominal and worst-case PL losses:
   ```
   ell_robust = (1 - rho) * ell_PL(sigma_obs) + rho * ell_PL(sigma_wc)
   ```
3. **Sweep over rho** — compare rho=0 (nominal), rho=0.5, rho=1.0.
4. **Evaluation** — define a held-out metric
   (e.g. pairwise accuracy on test_prefs from `ultrafeedback_binarized`).
5. **Scale up** — increase dataset size and training steps once the
   pipeline is confirmed stable.
