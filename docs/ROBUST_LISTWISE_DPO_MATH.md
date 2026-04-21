# Robust Listwise DPO — Fixed Mathematical Definitions for Implementation

## Purpose

This document exists to prevent implementation drift.

The coding agent must follow the mathematical definitions in this file exactly.
Do not replace the objective with a generic ranking loss.
Do not redesign the score.
Do not change the order of implementation.

---

## 1. Core score definition

For every prompt-response pair (x, y), define the DPO-style implicit score:

g_theta(x, y) := beta * log( pi_theta(y|x) / pi_ref(y|x) )

This is the scalar score used in all pairwise, listwise, and robust-listwise objectives.

### Important
- The score is computed from the current policy and reference policy.
- Do not replace this with a learned linear score or unrelated reward model in the real LLM experiment.
- In the real LLM setting, this is the central score definition.

---

## 2. Pairwise DPO baseline

For a preferred response y+ and a dispreferred response y-, the standard pairwise DPO loss is:

ell_DPO(theta; x,y+,y-) = -log sigma( g_theta(x,y+) - g_theta(x,y-) )

This is the baseline objective and should be implemented first.

---

## 3. Fixed-list listwise setting

For one prompt x, define a fixed candidate list:

Y = {y_1, y_2, ..., y_K}

and a ranking label:

sigma_star in S_K

where sigma_star_i denotes the index of the candidate placed at rank i.

The candidate list is fixed for each sample.
The ranking label is derived from the provided preference information, e.g. by sorting candidates by overall_score.

---

## 4. Nominal listwise DPO

### 4.1 Plackett–Luce ranking probability

Using the scores g_theta(x,y_k), define the Plackett–Luce probability of a full ranking sigma_star:

P_theta(sigma_star | x, y_1:K)
=
prod_{i=1}^K
exp(g_theta(x, y_{sigma_star_i}))
/
sum_{j=i}^K exp(g_theta(x, y_{sigma_star_j}))

### 4.2 Nominal listwise loss

The nominal listwise DPO loss is:

ell_PL(theta; x, y_1:K, sigma_star) := - log P_theta(sigma_star | x, y_1:K)

Equivalent implementation form:

ell_PL
=
- sum_{i=1}^K g_theta(x, y_{sigma_star_i})
+ sum_{i=1}^K log( sum_{j=i}^K exp(g_theta(x, y_{sigma_star_j})) )

This is our nominal listwise DPO objective.

### Important
- Use exactly this PL-based loss.
- Do not substitute a generic listwise ranking loss.
- Do not replace it with pairwise decomposition except for explicit sanity checks.

---

## 5. Robust listwise DPO

For an observed ranking sigma_obs, define the robust listwise objective as:

ell_robust(s, sigma_obs)
=
(1-rho) * ell_PL(s, sigma_obs)
+ rho * ell_PL(s, sigma_wc)

where:
- s=(s_1,...,s_K) is the current candidate score vector
- rho in [0,1] is the robustness coefficient
- sigma_wc is the worst-case ranking under the current scores

This is our robust listwise DPO objective.

---

## 6. Worst-case ranking definition

Under the PL loss, the worst-case ranking is obtained by sorting the current scores in ascending order:

sigma_wc = argsort(scores, ascending=True)

Interpretation:
- the candidate with the highest score is placed last
- the candidate with the lowest score is placed first

### Important
- Do not search over all permutations.
- Do not approximate this with random shuffling.
- Use ascending-score order exactly.

---

## 7. Required implementation order

The coding agent must follow this order strictly.

### Stage A — Pairwise DPO baseline
Implement and verify:
- standard DPO baseline
- TRL + Qwen + LoRA + 4-bit stack works

### Stage B — Nominal listwise DPO
Implement:
- fixed-list data pipeline
- per-candidate score g_theta(x,y)
- PL nominal listwise loss

This stage must be completed before any robust extension.

### Stage C — Robust listwise DPO
Only after nominal listwise DPO runs correctly:
- compute worst-case ranking
- compute robust loss
- compare against nominal listwise and pairwise baselines

---

## 8. Fixed-list training requirement

The nominal listwise trainer must be implemented in a fixed-list setting:

- for each prompt, keep a fixed list of K candidate responses
- compute one score for each candidate
- train using the provided ranking label for the whole list

The agent must not collapse the list into pairwise chosen/rejected examples at this stage.

---

## 9. Required sanity checks

Before moving from nominal listwise to robust listwise, verify:

1. The fixed-list grouping is correct.
2. All K responses in one training example correspond to the same prompt.
3. Ranking labels are derived correctly.
4. Score direction is correct (better response should have higher score).
5. For K=2, nominal listwise behaves consistently with pairwise DPO.
6. Robust listwise with rho=0 is exactly equal to nominal listwise.
7. Worst-case ranking is exactly ascending-score order.

---

## 10. Implementation warning

Do not hallucinate alternative formulations.

The objectives in this file are the source of truth.

Specifically, do not:
- replace PL with another ranking loss,
- replace the DPO score with an unrelated score,
- invent a new robust objective,
- simplify fixed-list training into pairwise training,
- redefine worst-case ranking.

If something is unclear, ask for clarification before changing the mathematics.
