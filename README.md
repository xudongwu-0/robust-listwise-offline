# Robust Listwise LLM

Experiments on robust listwise preference optimization with LLMs.

---

## Publish to GitHub (recommended workflow)

This folder is currently a normal directory (no `.git` yet). Use the steps below
to publish safely and avoid uploading large local artifacts by mistake.

### 1. Create the remote repository

Create a new empty repository in your GitHub account (no README, no `.gitignore`)
from your repositories page:

`https://github.com/xudongwu-0?tab=repositories`

Example name: `robust_listwise_llm`

### 2. Initialize Git locally and push

```bash
cd /home/xudong/work/robust_listwise_llm

# initialize repo
git init
git branch -M main

# add your GitHub remote (replace REPO_NAME if needed)
git remote add origin git@github.com:xudongwu-0/REPO_NAME.git
# or HTTPS:
# git remote add origin https://github.com/xudongwu-0/REPO_NAME.git

# first commit
git add .
git commit -m "chore: initial public release"
git push -u origin main
```

### 3. Verify ignored files before commit

By default this project ignores large local artifacts (checkpoints/logs/raw data).
You can verify with:

```bash
git status --short
git check-ignore -v outputs/checkpoints data/raw
```

If you previously tracked large files, untrack them while keeping local copies:

```bash
git rm -r --cached outputs/checkpoints outputs/logs data/raw data/processed
git commit -m "chore: stop tracking local artifacts"
```

### 4. Optional: share model weights with Git LFS

If you later need to publish adapters/checkpoints in Git:

```bash
git lfs install
git lfs track "*.safetensors"
git add .gitattributes
git commit -m "chore: track model files with git-lfs"
```

Without Git LFS, keep model weights out of the repository and release them via
external storage (for example Hugging Face).

---

## Environment setup

```bash
conda create -n robust_listwise_llm python=3.10 -y
conda activate robust_listwise_llm
pip install -r requirements.txt
```

> **PyTorch note**: install with your CUDA version first if needed:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Stage A — DPO baseline sanity check

Verifies that **TRL + Qwen2.5-0.5B-Instruct + LoRA + 4-bit** training stack is functional.

### What it does
- Loads `Qwen/Qwen2.5-0.5B-Instruct` in 4-bit NF4 (~350 MB VRAM)
- Applies LoRA (`r=16`) to attention projection layers
- Trains with TRL `DPOTrainer` on 500 samples from `HuggingFaceH4/ultrafeedback_binarized`
- Runs 50 optimisation steps (< 5 min on a single RTX 4090)

### Run

```bash
conda activate robust_listwise_llm
cd ~/work/robust_listwise_llm
CUDA_VISIBLE_DEVICES=0 python src/scripts/train_dpo_baseline.py
```

### Expected output

```
trainable params: 5,570,560 || all params: 500,194,560 || trainable%: 1.11
...step 5/50  loss: ~0.69
...step 10/50 loss: decreasing
...
checkpoint saved: outputs/checkpoints/dpo_baseline
```

A successful run produces a LoRA adapter checkpoint in `outputs/checkpoints/dpo_baseline/`.

---

## Stage B — Nominal listwise DPO (fixed-list, K=4)

Trains the Plackett-Luce nominal listwise DPO objective on `openbmb/UltraFeedback`, with K=4 candidates per prompt.

### What it does
- Loads `openbmb/UltraFeedback`; uses the `overall_score` annotation to derive a K=4 ranking per prompt
- Each forward pass computes `g_theta(x, y) = beta * log(pi_theta / pi_ref)` for all 4 candidates
- The reference model is the same Qwen base with the LoRA adapter disabled (no extra memory)
- Loss is the Plackett-Luce negative log-likelihood (§4 of `docs/ROBUST_LISTWISE_DPO_MATH.md`)

### Run

```bash
conda activate robust_listwise_llm
cd ~/work/robust_listwise_llm
CUDA_VISIBLE_DEVICES=0 python src/scripts/train_nominal_listwise.py
```

### Expected output

```
SANITY CHECK 1 PASSED: K=2 PL loss = pairwise DPO  (pl=1.0487, pw=1.0487)
SANITY CHECK 2 PASSED: score direction correct  (loss_good=0.0067 < loss_bad=15.07)
SANITY CHECK 3 PASSED: ranking is a valid K=4 permutation in all checked samples
trainable params: 2,162,688 || all params: 496,195,456 || trainable%: 0.44
...step 5/50   loss: decreasing from ~1.6
...step 50/50  loss: lower
checkpoint saved: outputs/checkpoints/nominal_listwise/
```

### Key files

| File | Role |
|------|------|
| [src/data/ultrafeedback_listwise.py](src/data/ultrafeedback_listwise.py) | Dataset builder + `ListwiseCollator` (K=4) |
| [src/losses/plackett_luce.py](src/losses/plackett_luce.py) | PL loss (exact §4.2 formula) |
| [src/trainers/listwise_trainer.py](src/trainers/listwise_trainer.py) | `NominalListwiseTrainer` |
| [src/scripts/train_nominal_listwise.py](src/scripts/train_nominal_listwise.py) | Main training script with inline sanity checks |
| [docs/stage_b_notes.md](docs/stage_b_notes.md) | Implementation notes |

---

## Stage C — Robust listwise DPO

Extends Stage B with the robust objective (§5 of `docs/ROBUST_LISTWISE_DPO_MATH.md`):

```
ell_robust = (1-rho) * ell_PL(sigma_obs) + rho * ell_PL(sigma_wc)
sigma_wc   = argsort(g, ascending=True)   # worst-case = ascending-score order
```

Setting `rho=0` recovers the nominal listwise loss exactly (verified by inline sanity check).

### Run

```bash
conda activate robust_listwise_llm
cd ~/work/robust_listwise_llm

# rho=0 — must be identical to nominal listwise (sanity check)
CUDA_VISIBLE_DEVICES=0 python src/scripts/train_robust_listwise.py --rho 0.0

# rho=0.1 (default)
CUDA_VISIBLE_DEVICES=0 python src/scripts/train_robust_listwise.py --rho 0.1

# rho=0.5
CUDA_VISIBLE_DEVICES=0 python src/scripts/train_robust_listwise.py --rho 0.5
```

Checkpoint is saved to `outputs/checkpoints/robust_listwise_rho{rho:.2f}/`.

### Expected output (rho=0.1)

```
SANITY CHECK 1 PASSED: rho=0 robust == nominal  (nominal=X.XXXXXX, robust0=X.XXXXXX)
SANITY CHECK 2 PASSED: worst-case ranking is ascending-score order
SANITY CHECK 3 PASSED: rho=1 robust == worst-case PL
SANITY CHECK 4 PASSED: aligned scores → nominal < worst-case
...step 5/50   loss: ...
...
checkpoint saved: outputs/checkpoints/robust_listwise_rho0.10/
```

### Key files (Stage C additions)

| File | Change |
|------|--------|
| [src/losses/plackett_luce.py](src/losses/plackett_luce.py) | + `worst_case_ranking()`, `robust_pl_loss()` |
| [src/trainers/listwise_trainer.py](src/trainers/listwise_trainer.py) | + `RobustListwiseTrainer` |
| [src/scripts/train_robust_listwise.py](src/scripts/train_robust_listwise.py) | New training script (`--rho` argument) |
| [docs/stage_c_notes.md](docs/stage_c_notes.md) | Implementation notes |

---

## Stage D — Noise injection experiment sweep

Measures how robust (ρ > 0) vs nominal (ρ = 0) training degrades as injected ranking noise increases.

### Noise types

| Type | Description |
|------|-------------|
| `near_tie` | Swaps the adjacent pair with the smallest score gap (simulates annotation confusion between near-equal responses) |
| `top_rank` | Replaces rank-1 with a uniformly-random other response (simulates systematic error in identifying the best response) |

### Sweep grid

- **noise_type** ∈ {`near_tie`, `top_rank`}
- **noise_level** (per-sample corruption probability) ∈ {0.0, 0.4, 1.0}
- **method** ∈ {`nominal` (ρ=0), `robust` (ρ=0.1)}

### Run

```bash
conda activate robust_listwise_llm
cd ~/work/robust_listwise_llm

# Full sweep (10 configs × ~2 min each ≈ 20 min on single RTX 4090)
CUDA_VISIBLE_DEVICES=1 python src/scripts/run_noise_sweep.py

# Quick smoke test (20 steps × 100 eval examples)
CUDA_VISIBLE_DEVICES=1 python src/scripts/run_noise_sweep.py --quick

# Single config (for debugging)
CUDA_VISIBLE_DEVICES=1 python src/scripts/run_noise_sweep.py \
    --noise_type near_tie --noise_level 0.4 --method robust
```

### Output

Results are saved incrementally (one row per completed run):

```
outputs/noise_sweep/results.csv
outputs/noise_sweep/{noise_type}_lvl{level}_{method}/   # LoRA adapter checkpoint
outputs/noise_sweep/sweep.log
```

### Expected hypothesis

At `noise_level=0`, nominal ≈ robust (identical training signal).
At `noise_level=0.4` or `1.0`, robust should achieve higher pairwise accuracy because the
worst-case regularisation at ρ=0.1 discounts unreliable high rankings.

### Key files (Stage D additions)

| File | Role |
|------|------|
| [src/data/noise.py](src/data/noise.py) | `make_noise_fn()` factory; `verify_noise_functions()` sanity checks |
| [src/data/ultrafeedback_listwise.py](src/data/ultrafeedback_listwise.py) | + `noise_fn` parameter (injected between extract and tokenise steps) |
| [src/eval/pairwise_accuracy.py](src/eval/pairwise_accuracy.py) | DPO pairwise accuracy on `ultrafeedback_binarized test_prefs` |
| [src/scripts/run_noise_sweep.py](src/scripts/run_noise_sweep.py) | Orchestrates full sweep, saves CSV, prints results table |

---

## Project layout

```
src/
  scripts/
    train_dpo_baseline.py        # Stage A — pairwise DPO baseline
    train_nominal_listwise.py    # Stage B — nominal listwise DPO
    train_robust_listwise.py     # Stage C — robust listwise DPO  (--rho)
    run_noise_sweep.py           # Stage D — noise experiment sweep
  data/
    ultrafeedback_listwise.py    # dataset builder + ListwiseCollator (+ noise_fn support)
    noise.py                     # noise factories: near_tie, top_rank
  losses/
    plackett_luce.py             # PL loss, worst_case_ranking, robust_pl_loss
  trainers/
    listwise_trainer.py          # NominalListwiseTrainer, RobustListwiseTrainer
  eval/
    pairwise_accuracy.py         # DPO pairwise accuracy on test_prefs
  models/                        # (future)
data/
  raw/
  processed/
outputs/
  checkpoints/
  logs/
  plots/
docs/
  ROBUST_LISTWISE_DPO_MATH.md    # source-of-truth math definitions
  stage_b_notes.md               # nominal listwise notes
  stage_c_notes.md               # robust listwise notes
```
