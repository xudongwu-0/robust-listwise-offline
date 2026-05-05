#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/offline/run_quick_test.sh
#
# Lightweight smoke test: trains nominal_pl for 20 steps on the clean
# condition (Qwen3-0.6B) and prints core metrics.
#
# Expected output line:
#   [SMOKE]  nominal_pl / clean  | kendall_tau=0.xxx  top1=0.xxx
#
# Runs in < 5 minutes on a single RTX 4090.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

GPU="${GPU:-0}"
SIZE="05b"

echo "[$(date '+%F %T')] === Offline smoke test (20 steps, ${SIZE}, GPU=${GPU}) ==="

CUDA_VISIBLE_DEVICES="${GPU}" python src/scripts/train_qwen3_noise_sweep.py \
    --model_size "${SIZE}" \
    --noise_type clean \
    --models nominal_pl \
    --quick 2>&1 | tee /tmp/offline_smoke_test.log

# Extract key metric line from the log
echo ""
echo "--- Smoke test result ---"
grep -E "kendall_tau|top1|pair_acc|SMOKE" /tmp/offline_smoke_test.log | tail -5 || true
echo ""
echo "[$(date '+%F %T')] Smoke test complete."
echo "Full log: /tmp/offline_smoke_test.log"
echo "Results CSV: outputs/qwen3/${SIZE}/noise_sweep/qwen3_${SIZE}_noise_sweep.csv"
