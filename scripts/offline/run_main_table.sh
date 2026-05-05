#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/offline/run_main_table.sh
#
# Runs the offline main-table sweep: all 5 noise conditions × 5 selected
# methods on Qwen3-0.6B (default) or Qwen3-8B.
#
# Selected methods (Table 1):
#   nominal_bt, nominal_pl, robust_pl_rho050, tv_dr_dpo_rho010, kldpo_tau100
#
# Usage:
#   bash scripts/offline/run_main_table.sh            # full sweep, GPU 0
#   bash scripts/offline/run_main_table.sh --quick    # smoke test (20 steps)
#   GPU=2 bash scripts/offline/run_main_table.sh      # choose GPU
#   SIZE=8b bash scripts/offline/run_main_table.sh    # 8B model
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

GPU="${GPU:-0}"
SIZE="${SIZE:-05b}"
QUICK="${1:-}"

MODELS="nominal_bt nominal_pl robust_pl_rho050 tv_dr_dpo_rho010 kldpo_tau100"

# Conditions: (noise_type, noise_level) — empty level means "clean"
declare -a CONDITIONS=(
    "clean "
    "near_tie 0.4"
    "near_tie 1.0"
    "top_rank 0.4"
    "top_rank 1.0"
)

echo "[$(date '+%F %T')] === Offline main-table sweep  size=${SIZE}  GPU=${GPU} ==="

for cond in "${CONDITIONS[@]}"; do
    noise_type=$(echo "$cond" | awk '{print $1}')
    noise_level=$(echo "$cond" | awk '{print $2}')

    if [ "$noise_type" = "clean" ]; then
        noise_args="--noise_type clean"
        cond_label="clean"
    else
        noise_args="--noise_type ${noise_type} --noise_level ${noise_level}"
        cond_label="${noise_type}_${noise_level}"
    fi

    quick_flag=""
    if [ "$QUICK" = "--quick" ]; then
        quick_flag="--quick"
    fi

    echo ""
    echo "[$(date '+%F %T')] --- Condition: ${cond_label} ---"
    CUDA_VISIBLE_DEVICES="${GPU}" python src/scripts/train_qwen3_noise_sweep.py \
        --model_size "${SIZE}" \
        ${noise_args} \
        --models ${MODELS} \
        ${quick_flag}
done

echo ""
echo "[$(date '+%F %T')] === Sweep complete ==="
echo "Results: outputs/qwen3/${SIZE}/noise_sweep/qwen3_${SIZE}_noise_sweep.csv"
