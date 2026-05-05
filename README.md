# Robust Listwise Preference Optimization

Offline fixed-list (K=4) preference optimization on UltraFeedback, comparing five methods under synthetic ranking-label noise:

| Method | Key param | Script key |
|--------|-----------|------------|
| Nominal BT/DPO | β=0.1 | `nominal_bt` |
| Nominal PL | β=0.1 | `nominal_pl` |
| Robust PL | ρ=0.05 / 0.10, β=0.1 | `robust_pl_rho050` / `robust_pl` |
| TV-DR-DPO | ρ=0.10, β=0.1 | `tv_dr_dpo_rho010` |
| KLDPO | τ=1.00, β=0.1 | `kldpo_tau100` |

Math derivation: [docs/ROBUST_LISTWISE_DPO_MATH.md](docs/ROBUST_LISTWISE_DPO_MATH.md)

---

## Setup

```bash
git clone https://github.com/xudongwu-0/robust-listwise-offline
cd robust-listwise-offline
conda create -n robust_listwise_llm python=3.10 -y
conda activate robust_listwise_llm
pip install -r requirements.txt
```

> **Requirement**: `trl==0.12.2` (pinned). The reference model is accessed via
> `model.disable_adapter()` — a separate reference copy is not used.

No dataset pre-download needed. `openbmb/UltraFeedback` is streamed and cached
automatically from HuggingFace Hub on first run.

---

## Reproduce Main Table

```bash
# Smoke test — 20 steps, < 5 min
bash scripts/offline/run_quick_test.sh

# Full sweep — 5 noise conditions × 5 methods, ~3 hr on one RTX 4090
bash scripts/offline/run_main_table.sh

# Choose GPU (default: 0)
GPU=2 bash scripts/offline/run_main_table.sh

# Collect and display results
python scripts/offline/collect_results.py
```

Results are written to:
`outputs/qwen3/05b/noise_sweep/qwen3_05b_noise_sweep.csv`

---

## Optional Sweeps

```bash
# Robust PL ρ sweep (clean condition)
bash scripts/offline/run_sweeps.sh rho_sweep

# TV-DR-DPO ρ sweep (clean condition)
bash scripts/offline/run_sweeps.sh tvdrdpo_rho_sweep

# KLDPO τ dev sweep (clean condition)
bash scripts/offline/run_sweeps.sh kldpo_tau_sweep
```

---

## Single Run

```bash
CUDA_VISIBLE_DEVICES=0 python src/scripts/train_qwen3_noise_sweep.py \
    --model_size 05b \
    --noise_type top_rank --noise_level 1.0 \
    --models nominal_bt nominal_pl robust_pl_rho050 tv_dr_dpo_rho010 kldpo_tau100

# Quick mode (20 steps)
CUDA_VISIBLE_DEVICES=0 python src/scripts/train_qwen3_noise_sweep.py \
    --model_size 05b --noise_type clean --quick
```

---

## Hyperparameters

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen3-0.6B` |
| Quantization | 4-bit NF4, bfloat16 compute |
| LoRA r / α | 16 / 32 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj |
| β | 0.1 |
| Learning rate | 5e-5 |
| Effective batch size | 8 (per_device=2 × grad_accum=4) |
| Training steps | 200 |
| List size K | 4 |
| TV-DR-DPO ρ (main table) | 0.10 |
| KLDPO τ (main table) | 1.00 |
| Robust PL ρ (main table) | 0.05 and 0.10 |

---

## Code Structure

```
src/
├── losses/plackett_luce.py          # PL loss, worst-case ranking, robust_pl_loss
├── trainers/
│   ├── listwise_trainer.py          # BT / Nominal PL / Robust PL trainers
│   └── dr_dpo_trainer.py            # TV-DR-DPO and KLDPO trainers
├── data/
│   ├── ultrafeedback_listwise.py    # Dataset builder, ListwiseCollator
│   └── noise.py                     # near_tie / top_rank noise injection
└── eval/
    ├── ranking_metrics.py           # Kendall τ, Top-1, Exact, NDCG
    └── pairwise_accuracy.py         # PairAcc on UF-binarized

scripts/offline/
├── run_main_table.sh    # Sequential main-table sweep
├── run_sweeps.sh        # ρ / τ dev sweeps
├── collect_results.py   # CSV → pivot table
└── run_quick_test.sh    # 20-step smoke test
```
