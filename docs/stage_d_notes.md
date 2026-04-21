# Stage D — Noise Injection Experiment Notes

## Purpose

Measure how pairwise accuracy changes when training labels are corrupted with controlled
ranking noise, comparing the nominal (ρ=0) and robust (ρ=0.1) objectives.

---

## Noise types

### `near_tie`
For each training example, finds the pair of adjacent responses with the **smallest score
gap** and swaps them in the ranking with probability `noise_level`.

Simulates annotation noise where human raters fail to distinguish near-equal responses.
This is a **soft, local** perturbation — the ground truth ranking is only slightly wrong.

### `top_rank`
With probability `noise_level`, replaces the rank-1 response with a uniformly-random
other response.

Simulates a **systematic, hard** error: the model the annotator identified as the best is
actually just random. This is more destructive than near_tie because it corrupts the most
informative position in the Plackett-Luce model (which weighs rank-1 most heavily).

---

## Results (50 training steps, 200 eval examples, Qwen2.5-0.5B-Instruct)

| noise_type | noise_level | nominal acc | robust acc | Δ (robust–nominal) |
|------------|-------------|-------------|------------|---------------------|
| near_tie   | 0.0         | 56.5%       | 56.0%      | **−0.5%**           |
| near_tie   | 0.4         | 58.0%       | 59.0%      | **+1.0%**           |
| near_tie   | 1.0         | 61.0%       | 60.0%      | **−1.0%**           |
| top_rank   | 0.4         | 54.0%       | 53.5%      | **−0.5%**           |
| top_rank   | 1.0         | **50.5%**   | **55.5%**  | **+5.0%**           |

---

## Observations

### 1. near_tie noise: minimal impact from robustness
Differences are within ±1%, which is within the noise of a 50-step / 200-example run.
The near_tie perturbation is benign — swapping near-equal responses barely changes the
PL loss landscape, so the adversarial regularisation in the robust objective provides no
systematic benefit (or harm).

### 2. top_rank noise at level 1.0: clear benefit from robustness
At `top_rank 1.0`, rank-1 is **always** replaced (100% corruption of the most informative
position). Under this condition:
- **Nominal** degrades to 50.5% — barely above random (50%), meaning it essentially
  failed to learn anything useful about what the best response is.
- **Robust** achieves 55.5% — the worst-case regularisation at ρ=0.1 penalises over-
  trusting the observed rank-1 label, which helps when rank-1 is always wrong.

### 3. top_rank at 0.4: robustness not yet beneficial
At 40% corruption, nominal and robust are within 0.5% of each other. The noise is not
severe enough for the ρ=0.1 regularisation to make a visible difference.

### 4. Loss is consistently higher for robust
The robust objective always has higher training loss than nominal (because ρ·ℓ_PL(σ_wc)
is an adversarial surplus on top of the nominal loss). This is expected and correct.

---

## Conclusions

The hypothesis is **partially confirmed**:
- Robustness helps most when noise is **destructive and frequent** (`top_rank 1.0`, +5%).
- Robustness provides no benefit for **soft, local** noise (`near_tie`).
- The benefit is most visible at **high corruption rates**, confirming that ρ acts as a
  floor on the worst-case loss and prevents the model from over-fitting to corrupted labels.

---

## Next steps

- Scale training to ≥200 steps and use ≥500 eval examples for lower-variance estimates.
- Sweep `rho` ∈ {0.05, 0.1, 0.2, 0.5} to find the optimal regularisation strength.
- Evaluate on RewardBench (broader benchmark).
- Reproduce with multiple random seeds to confirm statistical significance.
