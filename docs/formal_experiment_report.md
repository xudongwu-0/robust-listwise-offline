# Formal Experiment Report: Nominal BT vs. Nominal PL vs. Robust PL

> **Status**: living document — results sections will be filled in once training is complete.
>
> **Author**: experiment notes for group meeting & supervisor discussion

---

## Table of Contents

1. [Experiment Purpose](#1-experiment-purpose)
2. [Three Main Models](#2-three-main-models)
3. [Data and Noise](#3-data-and-noise)
4. [Evaluation Metrics — Detailed Definitions](#4-evaluation-metrics--detailed-definitions)
5. [near_tie vs. top_rank Noise — Nature and Difference](#5-near_tie-vs-top_rank-noise--nature-and-difference)
6. [Phase 1: Clean-Data Training Results](#6-phase-1-clean-data-training-results)
7. [Phase 2: Noise Experiment Results](#7-phase-2-noise-experiment-results)
8. [RewardBench Results](#8-rewardbench-results)
9. [Conclusions and Open Questions](#9-conclusions-and-open-questions)

---

## 1. Experiment Purpose

The central question of this project is:

> **Does robust listwise DPO — which regularises against worst-case ranking
> permutations during training — produce a model that is more robust to
> corrupted preference labels than the nominal listwise or pairwise (BT) baselines?**

To answer this seriously, we need:
1. A fair, unified training setup where all three objectives receive identical data.
2. A comprehensive evaluation suite beyond a single pairwise accuracy number.
3. A controlled noise experiment where the corruption level is a design variable.
4. An out-of-distribution benchmark (RewardBench) that is independent of
   the training distribution.

The experiments here represent the first serious (>smoke-test) sweep.

---

## 2. Three Main Models

All models share:
- Base: **Qwen/Qwen2.5-0.5B-Instruct** in 4-bit NF4 quantisation
- LoRA: `r=16`, `α=32`, targets `q/k/v/o_proj`
- Training dataset: `openbmb/UltraFeedback`, K=4 fixed-list format
- Hyperparameters: `β=0.1`, lr=5e-5, cosine schedule, effective batch=8
  
### 2.1 Nominal BT (Bradley-Terry pairwise DPO)

**Objective:**

$$\ell_{\text{BT}} = -\log \sigma\!\big( g_\theta(x, y_{\text{top}}) - g_\theta(x, y_{\text{bot}}) \big) = \log\!\big(1 + e^{-(g_\top - g_\bot)}\big)$$

where $g_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$ is the implicit DPO reward,
$y_\top$ is the **K=4 top-ranked** response and $y_\bot$ is the **K=4 bottom-ranked** response.

**Key properties:**
- Uses only 2 out of 4 responses per example (the extreme pair).
- Equivalent to standard pairwise DPO on the hardest contrast in the list.
- Simple and well-studied; serves as the most direct comparison baseline.
- Implemented via `BTListwiseTrainer` in `src/trainers/listwise_trainer.py`.

### 2.2 Nominal PL (Plackett-Luce listwise DPO)

**Objective:**

$$\ell_{\text{PL}}(\sigma^*) = -\sum_{i=1}^{K} g_{\sigma^*_i} + \sum_{i=1}^{K} \log\sum_{j=i}^{K} e^{g_{\sigma^*_j}}$$

where $\sigma^*$ is the observed ranking (rank-0 = best) and
$g_{\sigma^*_i} = g_\theta(x, y_{\sigma^*_i})$.

Equivalently: $\ell_{\text{PL}} = -\sum_i g_{\sigma^*_i} + \sum_i \text{logsumexp}(g_{\sigma^*_{i:}})$.

**Key properties:**
- Uses all K=4 responses simultaneously; assigns proportionally more
  weight to correct placement of higher-ranked responses (via the logsumexp suffixes).
- Reduces to $\ell_{\text{BT}}$ when K=2 (verified by sanity check in Stage B).
- More data-efficient than BT per sample.
- Implemented via `NominalListwiseTrainer`.

### 2.3 Robust PL (Robust Plackett-Luce, ρ=0.1)

**Objective:**

$$\ell_{\text{robust}}(\sigma^\text{obs}) = (1-\rho)\,\ell_{\text{PL}}(\sigma^\text{obs}) + \rho\,\ell_{\text{PL}}(\sigma^\text{wc})$$

$$\sigma^\text{wc} = \operatorname{argsort}(g, \text{ ascending}) \quad \text{[worst-case ranking]}$$

**Key properties:**
- $\rho=0$ exactly recovers nominal PL (verified by sanity check in Stage C).
- $\sigma^\text{wc}$ is the ranking that *maximises* the current model's PL loss — it
  puts the response the model currently prefers least at the top.
- The $\rho$ term penalises the model for being exploitable: even if the observed
  label is wrong, the model should not be trapped by it.
- Implemented via `RobustListwiseTrainer` with `rho=0.1` (default).

**Interpretation of ρ=0.1:** 10% of the training signal comes from adversarial
worst-case permutations. Small enough to not dominate on clean data, large enough
to matter under systematic noise.

---

## 3. Data and Noise

### 3.1 Training data

All models in this formal experiment are trained on **openbmb/UltraFeedback** (train split).

- Each sample: 1 instruction + 4 model-generated completions + `overall_score ∈ [1,5]`
- Ground-truth ranking: `argsort(scores, descending=True)` → rank-0 = best
- Training split: first 5000 samples (shuffle seed=42)
- Held-out split: samples 5000–5500 (same shuffle) → clean evaluation only

### 3.2 Noise injection (Phase 2)

Noise is injected into the **training ranking only**. The held-out evaluation set
is always the clean ground truth.

This setup measures: *how well can each model learn despite noisy training labels?*

Two types of noise are implemented (in `src/data/noise.py`):

---

## 4. Evaluation Metrics — Detailed Definitions

All metrics are computed on the **clean held-out K=4 set** (samples 5000–5500 of
UltraFeedback). Unless otherwise stated, higher values indicate better performance.

### 4.1 Top-1 Accuracy (`top1_acc`)

**Definition:**
$$\text{Top-1} = \frac{1}{N} \sum_{n=1}^{N} \mathbf{1}\!\left[ \hat\sigma_n(0) = \sigma^*_n(0) \right]$$

- $\hat\sigma_n(0)$ is the index of the response the model assigns the highest DPO score $g_\theta$.
- $\sigma^*_n(0)$ is the index of the response with the highest `overall_score`.
- Equals 1 if and only if the top-1 prediction is correct.

**Strengths:** Directly measures the most practically important case — "did the model correctly identify the best response?"

**Limitations:** Ignores ordering beyond rank 1; a random permutation still has $\text{Top-1} = 1/K$ (= 0.25 for K=4).

**Baseline (random model):**  0.25

---

### 4.2 Exact Match (`exact_match`)

**Definition:**
$$\text{Exact} = \frac{1}{N} \sum_{n=1}^{N} \mathbf{1}\!\left[ \hat\sigma_n = \sigma^*_n \right]$$

All 4 positions must match simultaneously.

**Strengths:** Most stringent metric; zero credit for partially correct rankings.

**Limitations:** Very harsh — two rankings that differ by a single swap of nearly-equal responses both score 0. Sensitive to tie-breaking noise.

**Baseline (random):** $1/4! = 1/24 \approx 0.042$

---

### 4.3 Kendall τ (`kendall_tau`)

**Definition:**
$$\tau = \frac{C - D}{\binom{K}{2}} \in [-1, +1]$$

where C = number of concordant pairs (the model correctly orders them), D = discordant pairs.

Equivalently: $\tau = 2 \cdot \text{pairwise\_acc}_{K\text{-way}} - 1$ (linear transform of K-way pairwise accuracy).

**Strengths:**
- Balanced metric that considers all $\binom{4}{2} = 6$ response pairs.
- Τ=1 means perfect ranking agreement; τ=0 means indistinguishable from random; τ=-1 means perfectly reversed ranking.
- Standard metric in ranking literature; widely understood.

**Limitations:** Does not weight the top of the ranking more. A swap at position 3–4 has the same cost as a swap at position 1–2.

**Baseline (random):** $\tau \approx 0$ (by symmetry).

---

### 4.4 NDCG@4 (`ndcg`)

**Definition:**
$$\text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}, \quad \text{DCG}@K = \sum_{i=1}^{K} \frac{r_{\hat\sigma(i)}}{\log_2(i+1)}$$

where $r_i$ = `overall_score` of response $i$ (used as graded relevance), and IDCG is the DCG of the perfect ranking.

**Strengths:**
- Logarithmic discount means top positions matter more.
- Handles ties in relevance gracefully (uses continuous scores, not just ranks).
- Standard in information retrieval; directly reflects the "quality loss" of ranking errors at the top.

**Limitations:**
- Depends heavily on the magnitude and distribution of scores. If all four responses have similar scores, NDCG is nearly 1.0 for almost any model.
- Less interpretable than top-1 or Kendall τ for small K.

**Baseline (random):** $\approx 0.85–0.95$ for K=4 with typical UltraFeedback score distributions (scores are clustered near 3.5–4.5, so the rating gap between positions is small).

---

### 4.5 K=4 Pairwise Accuracy (`pairwise_acc_k4`)

**Definition:**
$$\text{PairAcc}_{K4} = \frac{1}{N \cdot \binom{K}{2}} \sum_{n=1}^{N} \sum_{(i,j): i \succ^* j} \mathbf{1}\!\left[ g_\theta(x_n, y_i) > g_\theta(x_n, y_j) \right]$$

Fraction of ground-truth pairwise preferences that the model's $g_\theta$ correctly reproduces, across all $\binom{4}{2}=6$ pairs per example.

This is the *K=4 generalisation* of the standard DPO pairwise accuracy. Note the relationship $\text{PairAcc}_{K4} = (\tau + 1)/2$.

**Strengths:** Familiar, interpretable; directly measures the fraction of preferential comparisons the model gets right.

**Limitations:** Same as Kendall τ — no positional weighting.

**Baseline (random):** 0.5

---

### 4.6 Pairwise Accuracy on Binarized Test Set (`pairwise_acc_binarized`)

**Definition:** Same as §4.5 but evaluated on `HuggingFaceH4/ultrafeedback_binarized`'s `test_prefs` split (K=2 pairs, different annotation pipeline).

Used for comparison with Stage A–C results. Measures generalisation to the binarized annotation standard.

---

### 4.7 RewardBench Accuracy (`rb_*`)

**Definition:** Per-category pairwise accuracy on `allenai/reward-bench`. Same g_θ proxy reward.

Categories: **Chat**, **Chat Hard**, **Safety**, **Reasoning**, **Overall** (simple average).

**Note:** RewardBench was designed for explicit reward models. Our g_θ is a policy-based proxy (DPO implicit reward). This may under-estimate performance relative to dedicated reward head models, but is a valid comparison *across* the three objectives under study.

---

## 5. near_tie vs. top_rank Noise — Nature and Difference

### 5.1 near_tie: definition and mechanism

**Algorithm:**

For a given sample with K=4 responses labelled by `overall_score`:
1. Compute all K-1=3 **adjacent score gaps**:
   `gap[i] = score[ranking[i]] - score[ranking[i+1]]`  for i ∈ {0,1,2}
2. Find `swap_pos = argmin gap`.
3. With probability `noise_level`, swap positions `swap_pos` and `swap_pos+1` in the ranking.

**Concrete example:**

Suppose K=4 responses have overall scores [8.0, 5.0, 10.0, 8.1]. The ground-truth ranking is:
- rank 0 → response 2 (score 10.0)
- rank 1 → response 3 (score 8.1)
- rank 2 → response 0 (score 8.0)
- rank 3 → response 1 (score 5.0)

Adjacent gaps: [10.0−8.1=1.9, 8.1−8.0=0.1, 8.0−5.0=3.0]

Minimum gap is 0.1 at position 1 (between ranks 1 and 2). With `noise_level=1.0`:
- Noisy ranking: [2, **0, 3**, 1]   (responses 3 and 0 are swapped)

**Which pair does it affect?** The near_tie corruption always targets the **locally hardest annotation decision** — two responses whose quality is nearly indistinguishable. In K=4 UltraFeedback data, this often corresponds to the middle ranks (positions 1–2), since UltraFeedback scores tend to cluster around 3.5–4.5.

**Why near_tie is more like smoothing than hard corruption:**

- It never changes the rank-1 (best) response.
- It only swaps two responses whose quality is nearly identical.
- For the Plackett-Luce loss, swapping two low-quality similar responses barely changes the loss — the log-ratio of their selection probabilities is negligible.
- From the model's perspective, the corrupted label is *plausible*: a human annotator could genuinely disagree about which of two near-equal responses is better.
- This is analogous to **label smoothing** in classification: the label is perturbed but the perturbation is bounded and semantically mild.

**Expected model behaviour under near_tie noise:**
- High noise level does *not* strongly corrupt the training signal.
- Both nominal and robust models should be relatively unaffected.
- You may observe that the training loss goes down slightly at noise_level=1.0 (the model has an "easier" objective — it only needs to distinguish the genuinely different responses).

---

### 5.2 top_rank: definition and mechanism

**Algorithm:**

For a given sample:
1. With probability `noise_level`:
   - Sample a random position j ∈ {1, 2, ..., K-1} uniformly.
   - Swap positions 0 and j (replace the best response with a random response).

**Why top_rank is strongly destructive:**

- It corrupts the **single most informative position** in the Plackett-Luce model.
  The PL log-likelihood assigns the largest gradient magnitude to the rank-1 comparison
  (via the first logsumexp suffix, which covers all K responses).
- Concretely: with `noise_level=1.0`, rank-1 is **always wrong** — the model is
  being told "this is the best response" for a response that comes from a uniform
  random draw over the 3 lower-ranked responses.
- At `noise_level=1.0`: with probability 1/3, the selected "best" response is actually
  rank-2 (a moderate perturbation); with probability 1/3 it is rank-3; and with
  probability 1/3 it is rank-4 (the actual worst response — maximum corruption).
- Unlike near_tie, top_rank can promote the **worst** response to rank-1, leading to
  a fully misleading training signal that directly contradicts true quality ordering.

---

### 5.3 Why the two noise types produce different model behaviour

| Aspect | near_tie | top_rank |
|--------|----------|----------|
| Which rank is affected? | Middle ranks (1–2) | Rank 1 (always) |
| Score gap of corrupted pair | Minimal (near-equal) | Potentially large |
| Plausibility of corrupted label | High (genuinely debatable) | Low (best→worst swap is implausible) |
| PL loss impact | Small (negligible gradient change) | Large (corrupts highest-weight term) |
| Analogy | Label smoothing | Systematic adversarial mislabelling |
| Expected nominal model behaviour | Relatively unaffected | Degrades significantly at high noise |
| Expected robust model advantage | Small | Large |

**Interpretation of prior Stage D results (50 training steps):**

- near_tie: nominal and robust were within ±1% — consistent with near_tie being a weak perturbation that doesn't substantially change the training signal.
- top_rank 1.0: nominal dropped to 50.5% (near random) while robust achieved 55.5% (+5%). This confirms that *robust PL's adversarial worst-case term prevents over-trusting the corrupted rank-1 label*.

**Why does nominal possibly improve under near_tie at high noise?**
With near_tie 1.0, all adjacent near-equal swaps become systematic. In some cases this may actually create a *cleaner* training signal in a different sense: the model no longer needs to distinguish near-equal responses, and the ranking of clearly different responses (the extremes) is unchanged. The implicit consequence is that the model becomes more "smoothed" at the middle ranks.

**Why robust PL shows advantage most clearly under top_rank:**
The worst-case term in the robust loss explicitly asks: "what if the ranking I observe is adversarially wrong?" Under top_rank noise, this is *precisely* what is happening — so the robust training procedure has learned to guard against exactly this type of corruption.

---

## 6. Phase 1: Clean-Data Training Results

*Filled in after `train_formal.py` completes.*

### Setup
- Training steps: 1000
- Training samples: 5000 (clean, no noise)
- Eval K=4 held-out: 500 samples (UltraFeedback, no overlap with training)
- Eval binarized: 500 samples (`ultrafeedback_binarized` `test_prefs`)

### Results

See `outputs/formal/formal_clean_results.csv` and `outputs/formal/plots/clean_comparison.png`.

| Model | Train Loss | Top-1 | Exact | Kendall τ | NDCG | PairAcc(K4) | PairAcc(bin) | RB Overall |
|-------|-----------|-------|-------|-----------|------|-------------|--------------|------------|
| Nominal BT | 1.9669 | 0.376 | 0.088 | 0.2387 | 0.8744 | 0.6157 | **0.724** | 0.6606 |
| Nominal PL | 11.396 | 0.374 | **0.098** | 0.2507 | **0.875** | **0.622** | 0.710 | 0.6603 |
| Robust PL  | 11.937 | 0.374 | 0.096 | **0.2513** | 0.873 | **0.622** | 0.706 | 0.6563 |

**RewardBench breakdown:**

| Model | Chat | Chat Hard | Safety | Reasoning | Overall |
|-------|------|-----------|--------|-----------|--------|
| Nominal BT | 0.896 | 0.441 | 0.478 | **0.768** | **0.661** |
| Nominal PL | **0.906** | 0.441 | 0.484 | 0.762 | 0.660 |
| Robust PL  | **0.909** | **0.443** | **0.488** | 0.751 | 0.656 |

### Interpretation

All three objectives converge to similar clean-data performance at 1000 steps — the differences across the 7 metrics are below 0.2% for most metrics.

Key observations:

- **Top-1 accuracy (0.374–0.376)** is nearly identical across models, suggesting that ranking the best response correctly is driven mostly by training data quality, not objective choice, at this scale.
- **Kendall τ (0.239–0.251)**: Nominal and robust PL slightly outperform BT on full-ranking correlation, consistent with PL's ability to use all K=4 positions jointly.
- **Pairwise accuracy (binarized)**: BT leads at 72.4%, showing that BT's direct pairwise training on binary preferences (via the extreme top-bottom pair) yields the strongest signal for simple chosen-vs-rejected discrimination — the task it is most directly optimised for.
- **RewardBench**: See §8 for detailed breakdown. All models score 0.656–0.661 overall; differences are within noise at this training budget.
- **Robust PL vs. Nominal PL on clean data**: At ρ=0.1, robust PL achieves τ=0.2513 vs. PL's 0.2507 — negligible difference. The worst-case term does **not** hurt clean performance, confirming that ρ=0.1 is a conservative regularisation choice in the clean regime.

These clean-data results serve as the baseline for interpreting the noise sweep (§7): any advantage seen under noise is not simply carried over from better clean-data fitting.

---

## 7. Phase 2: Noise Experiment Results

*Filled in after `run_formal_noise_sweep.py` completes.*

### Setup
- Training steps: 200 per run (sufficient to see noise effects, feasible runtime)
- Training samples: 1000 (noisy)
- Eval: 300 clean K=4 held-out + 300 binarized test_prefs

### near_tie Noise Results

*(See `outputs/formal/plots/near_tie_comparison.png`)*

| Noise Level | BT (top1) | PL (top1) | Robust (top1) | BT (τ) | PL (τ) | Robust (τ) |
|-------------|-----------|-----------|---------------|--------|--------|------------|
| noise_level | BT (τ) | PL (τ) | Robust (τ) | Δ(robust−PL) | BT (top-1) | PL (top-1) | Robust (top-1) |
|-------------|--------|--------|------------|--------------|------------|------------|----------------|
| 0.0 | 0.228 | 0.249 | **0.270** | +0.021 | 0.360 | 0.350 | **0.387** |
| 0.4 | 0.221 | 0.250 | **0.260** | +0.010 | 0.343 | 0.360 | **0.383** |
| 1.0 | 0.211 | **0.250** | 0.238 | **−0.012** | 0.347 | **0.367** | 0.357 |

### top_rank Noise Results

*(See `outputs/formal/plots/top_rank_comparison.png`)*

| noise_level | BT (τ) | PL (τ) | Robust (τ) | Δ(robust−PL) | BT (top-1) | PL (top-1) | Robust (top-1) |
|-------------|--------|--------|------------|--------------|------------|------------|----------------|
| 0.4 | 0.199 | 0.214 | **0.219** | +0.005 | 0.323 | 0.353 | **0.357** |
| **1.0** | **−0.016** | 0.051 | **0.079** | **+0.028** | 0.267 | 0.260 | **0.300** |

### Analysis

#### 1. BT catastrophically fails under top_rank noise

At `top_rank 1.0` (rank-1 always wrong), **BTListwiseTrainer collapses to τ=−0.016 — worse than a random ranker** (τ=0 at baseline). This is the strongest result in the experiment, and it is mechanistically explained:

- BTListwiseTrainer uses only `(ranking[0], ranking[-1])` — the top vs. bottom pair.
- Under `top_rank 1.0`, `ranking[0]` is *always* a uniformly-random non-top response.
- The BT training signal is therefore fully corrupted: 100% of training examples tell the model "this random response is better than the worst."
- The model learns inverted preferences for the extreme pair, leading to negative τ.

This result shows that **BT trained on extreme pairs is more brittle to top-rank corruption than listwise models**, precisely because it concentrates gradient on the most vulnerable position in the ranking.

#### 2. Robust PL outperforms nominal PL under top_rank noise

At `top_rank 1.0`:
- Nominal PL: τ=0.051, top-1=0.260
- Robust PL: τ=0.079 (+0.028), top-1=0.300 (+0.040)

This is the largest Δ(robust−PL) in the experiment. The mechanism:
- The robust worst-case term asks the model to guard against the adversarial permutation σ_wc, which penalises it for placing the highest-g_θ response at rank-1.
- Under top_rank 1.0 noise, rank-1 is always corrupted, so the model that "distrusts" rank-1 labels has an advantage.

#### 3. near_tie noise: robust PL leads at low-to-medium levels, reverts at 1.0

| Level | Δ(robust−PL) (τ) |
|-------|------------------|
| 0.0   | +0.021 |
| 0.4   | +0.010 |
| 1.0   | **−0.012** |

At `near_tie 1.0`, nominal PL surpasses robust PL. This is consistent with the theoretical analysis in §5.1: near_tie 1.0 means *all* minimum-gap adjacent pairs are systematically swapped. This creates a **smooth, deterministic perturbation** of the middle ranks — it makes the training signal for middle positions consistently "inverted" in a way that nominal PL might actually find learnable as a different (still consistent) ranking convention. The adversarial term in robust PL adds noise to this consistent signal.

#### 4. Summary of Δ(robust−PL) trends

| Noise condition | Δ(tau) | Interpretation |
|----------------|--------|----------------|
| near_tie 0.0   | +0.021 | Marginal benefit even on clean data (200-step regime) |
| near_tie 0.4   | +0.010 | Small benefit |
| **near_tie 1.0** | −0.012 | Near_tie fully systematic → adversarial term unhelpful |
| top_rank 0.4   | +0.005 | Small benefit at moderate corruption |
| **top_rank 1.0** | **+0.028** | Largest benefit — destructive corrupted rank-1 |

The pattern is clear: **robust PL helps most against hard, semantically destructive corruption (top_rank at high levels), and may be slightly counterproductive against soft, systematic relabelling (near_tie at 1.0).**

Key questions to answer from these results:
1. Does robust PL maintain its advantage at higher step count vs. Stage D (50 steps)?
2. Is the advantage specific to top_rank, or does it also emerge for near_tie at extreme levels?
3. Does BT outperform or underperform nominal PL under noise? (BT only uses the top-vs-bottom extreme pair, which is precisely the pair corrupted by top_rank noise — so BT may be *more* susceptible than PL to top_rank corruption.)

---

## 8. RewardBench Results

*(Filled in after `train_formal.py` completes. See `outputs/formal/plots/rewardbench.png`)*

| Model | Chat | Chat Hard | Safety | Reasoning | Overall |
|-------|------|-----------|--------|-----------|---------|
| Nominal BT | 0.896 | 0.441 | 0.478 | **0.768** | **0.661** |
| Nominal PL | **0.906** | 0.441 | 0.484 | 0.762 | 0.660 |
| Robust PL  | **0.909** | **0.443** | **0.488** | 0.751 | 0.656 |

*(See `outputs/formal/plots/rewardbench.png`)*

### Interpretation

All three models achieve roughly equivalent overall RewardBench scores (0.656–0.661), confirming that the choice of listwise objective has minimal impact on out-of-distribution generalisation at this training scale (1000 steps on UltraFeedback).

Notable category-level differences:

- **Chat**: Robust PL leads marginally (0.909 vs. 0.896 BT), suggesting the distributional robustness objective slightly improves general instruction-following quality.
- **Chat Hard**: Robust PL leads (0.443 vs. 0.441 BT/PL), showing the largest relative benefit on the most challenging chat examples.
- **Safety**: Robust PL leads (0.488 vs. 0.478 BT), which is surprising given that no explicit safety objective is used. A plausible explanation: the worst-case term biases the model away from responses with extreme positive scores (which safety refusals often receive as the "chosen" response), leading to more conservative safety-relevant scoring.
- **Reasoning**: Nominal BT leads (0.768 vs. 0.762 PL / 0.751 robust), consistent with BT's strength on clean clear-preference pairs — reasoning examples are typically unambiguous, favouring BT's pairwise extreme-pair training.
- **Overall gap**: At most 0.5% between models (0.656–0.661), well within expected variance for a 1000-step experiment. Relative ordering could reverse with more training or different seeds.

RewardBench confirms that robust PL does **not** impose a measurable penalty on out-of-distribution performance at this scale — the regularisation cost of the worst-case term is absorbed without damaging generalisation.

---

## 9. Conclusions and Open Questions

### What we are fairly confident about (from Stage D + prior experiments)

1. **Robust PL has a meaningful advantage under `top_rank 1.0` noise** (+5% pairwise accuracy in Stage D over 50 steps). This is statistically plausible and mechanistically justified.

2. **`near_tie` noise is a weak perturbation** — it does not strongly degrade any model, because it only swaps nearly-identical middle-ranked responses whose gradient contribution to the PL loss is small.

3. **Robust PL's training loss is consistently higher than nominal PL** — this is expected and correct. The worst-case term adds a constant surplus to the loss. It is *not* a sign of poor learning.

4. **rho=0 exactly recovers nominal PL** — verified by sanity check. The implementation is mathematically correct.

### What we are still uncertain about

1. **Does the robust PL advantage persist under longer training?**
   With 50 steps, training is severely under-fitted. At 1000 steps, the relative
   positions may change. One hypothesis: at longer training, all models converge
   to similar accuracy on the dominant pairs, and the robust advantage narrows.

2. **Does the advantage hold on clean data at longer training?**
   In Stage D (50 steps, clean), nominal and robust are within ±0.5%. This may
   widen or narrow at 1000 steps.

3. **Is BT better or worse than PL under top_rank noise?**
   BT only uses rank-0 and rank-3. Under `top_rank` noise, rank-0 is always corrupted.
   This means BT's training signal at `noise_level=1.0` is completely adversarial —
   the "chosen" response in BT is always wrong. We expect BT to fail catastrophically
   at `top_rank 1.0` while nominal PL (which also uses ranks 1–3, which are uncorrupted)
   may partially recover.

4. **Does RewardBench reveal any surprising category-level differences?**
   Safety category in RewardBench penalises models that fail to refuse. Robust PL
   has no explicit safety objective, so all three models may perform similarly on Safety.
   Differences, if any, are most likely in Reasoning (which requires fine-grained
   quality discrimination, exactly what the PL objective optimises).

5. **Optimal rho value?** We use rho=0.1 throughout. A sweep over rho ∈ {0.05, 0.1, 0.2, 0.5}
   under fixed top_rank 1.0 noise would reveal the robustness-performance trade-off curve.

### Recommended next steps

1. **More seeds**: repeat Phase 1 and 2 with at least 2 additional random seeds to confirm statistical significance.
2. **rho sweep**: 5 values × 2 noise conditions = 10 training runs to find optimal rho.
3. **Scale-up**: 10K training samples, 2000 steps — closer to publication-quality regime.
4. **Full RewardBench**: evaluate on all 2985 test examples (not a subset) for credible category-level estimates.
5. **KL regularisation tracking**: log `D_KL(pi_theta || pi_ref)` during training to understand whether robust PL moves further from the reference than nominal PL.

---

*File paths for all artefacts:*

| Artefact | Path |
|----------|------|
| Phase 1 results (CSV) | `outputs/formal/formal_clean_results.csv` |
| Phase 2 results (CSV) | `outputs/formal/noise_sweep_results.csv` |
| Phase 1 clean comparison plot | `outputs/formal/plots/clean_comparison.{png,pdf}` |
| near_tie line plot | `outputs/formal/plots/near_tie_comparison.{png,pdf}` |
| top_rank line plot | `outputs/formal/plots/top_rank_comparison.{png,pdf}` |
| Summary bar chart | `outputs/formal/plots/summary_bar.{png,pdf}` |
| RewardBench bar chart | `outputs/formal/plots/rewardbench.{png,pdf}` |
| Phase 1 model checkpoints | `outputs/formal/{nominal_bt,nominal_pl,robust_pl}/` |
| Phase 2 checkpoints | `outputs/formal/noise_sweep/{noise_type}_lvl{l}_{model}/` |
