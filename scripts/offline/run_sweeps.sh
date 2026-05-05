#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/offline/run_sweeps.sh
#
# Optional dev sweeps for hyperparameter selection.
#
# Usage:
#   bash scripts/offline/run_sweeps.sh rho_sweep          # Robust PL ρ grid
#   bash scripts/offline/run_sweeps.sh tvdrdpo_rho_sweep  # TV-DR-DPO ρ grid
#   bash scripts/offline/run_sweeps.sh kldpo_tau_sweep    # KLDPO τ dev sweep
#
# All sweeps run on the clean condition for fast dev selection.
# GPU defaults to 0; override with GPU=<id> env var.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

GPU="${GPU:-0}"
MODE="${1:-}"

case "$MODE" in

  rho_sweep)
    # Robust PL ρ sweep on Qwen3-0.6B clean condition.
    # ρ values: 0.00 (nominal_pl), 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00
    echo "[$(date '+%F %T')] Robust PL rho sweep (clean, 0.6B, GPU=${GPU})"
    CUDA_VISIBLE_DEVICES="${GPU}" python src/scripts/train_qwen3_noise_sweep.py \
        --model_size 05b --noise_type clean \
        --models nominal_pl \
                 robust_pl_rho050 robust_pl robust_pl_rho015 \
                 robust_pl_rho020 robust_pl_rho030 robust_pl_rho50 \
                 robust_pl_rho070 robust_pl_rho100
    echo "Results: outputs/qwen3/05b/noise_sweep/qwen3_05b_noise_sweep.csv"
    ;;

  tvdrdpo_rho_sweep)
    # TV-DR-DPO ρ sweep on Qwen3-0.6B clean condition.
    # ρ values: 0.05, 0.10, 0.20, 0.40, 0.80
    echo "[$(date '+%F %T')] TV-DR-DPO rho sweep (clean, 0.6B, GPU=${GPU})"
    CUDA_VISIBLE_DEVICES="${GPU}" python src/scripts/train_qwen3_noise_sweep.py \
        --model_size 05b --noise_type clean \
        --models tv_dr_dpo_rho005 tv_dr_dpo_rho010 tv_dr_dpo_rho020 \
                 tv_dr_dpo_rho040 tv_dr_dpo_rho080
    echo "Results: outputs/qwen3/05b/noise_sweep/qwen3_05b_noise_sweep.csv"
    ;;

  kldpo_tau_sweep)
    # KLDPO τ dev sweep on Qwen3-0.6B clean condition.
    # τ values: 0.05, 0.10, 0.20, 0.50, 1.00
    # Selected τ for main table: 1.00 (kldpo_tau100)
    echo "[$(date '+%F %T')] KLDPO tau sweep (clean, 0.6B, GPU=${GPU})"
    CUDA_VISIBLE_DEVICES="${GPU}" python src/scripts/train_qwen3_noise_sweep.py \
        --model_size 05b --noise_type clean \
        --models kldpo_tau005 kldpo_tau010 kldpo_tau020 kldpo_tau050 kldpo_tau100
    echo "Results: outputs/qwen3/05b/noise_sweep/qwen3_05b_noise_sweep.csv"
    ;;

  *)
    echo "Usage: $0 {rho_sweep|tvdrdpo_rho_sweep|kldpo_tau_sweep}" >&2
    exit 2
    ;;
esac
