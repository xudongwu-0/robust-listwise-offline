#!/usr/bin/env python3
"""
Plot generation for formal experiment results.

Reads:
    outputs/formal/formal_clean_results.csv     — Phase 1: 3 models, clean
    outputs/formal/noise_sweep_results.csv      — Phase 2: noise sweep

Writes (PNG + PDF) to outputs/formal/plots/:
    1. near_tie_comparison.{png,pdf}
       x = noise level, y = Kendall τ (primary) and top-1 acc
       3 lines: nominal_bt / nominal_pl / robust_pl

    2. top_rank_comparison.{png,pdf}
       same structure, top_rank noise type

    3. summary_bar.{png,pdf}
       side-by-side bars at noise_level=1.0 for both noise types
       grouped by model, y = Kendall τ

    4. rewardbench.{png,pdf}   (if rb_overall column is present and not all NaN)
       grouped bar: per-category RewardBench accuracy for 3 models

Usage
-----
conda activate robust_listwise_llm
cd ~/work/robust_listwise_llm
python src/scripts/plot_results.py
python src/scripts/plot_results.py --no_rewardbench
"""

import argparse
import csv
import math
import os
import sys

_FORMAL_DIR = "outputs/formal"
_PLOT_DIR   = os.path.join(_FORMAL_DIR, "plots")

# Consistent colour / marker scheme
MODEL_STYLE = {
    "nominal_bt": dict(color="#2196F3", marker="o",  linestyle="-",  label="Nominal BT"),
    "nominal_pl": dict(color="#4CAF50", marker="s",  linestyle="--", label="Nominal PL"),
    "robust_pl":  dict(color="#F44336", marker="^",  linestyle=":",  label="Robust PL (ρ=0.1)"),
}


def _load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _float(v):
    try:
        x = float(v)
        return x if not math.isnan(x) else None
    except (TypeError, ValueError):
        return None


def _save_fig(fig, name):
    os.makedirs(_PLOT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(_PLOT_DIR, f"{name}.{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved: {_PLOT_DIR}/{name}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# 1.  Noise line plots (near_tie and top_rank)
# ---------------------------------------------------------------------------

def plot_noise_comparison(noise_rows, noise_type: str, metric: str = "kendall_tau",
                          metric2: str = "top1_acc"):
    """
    Two-panel line chart for one noise type.
    Left panel: metric (Kendall τ), right panel: metric2 (top-1 acc)
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    rows = [r for r in noise_rows if r["noise_type"] == noise_type]
    if not rows:
        print(f"  [warn] no rows for noise_type={noise_type}, skipping plot")
        return

    # Extract unique noise levels (sorted)
    levels = sorted(set(float(r["noise_level"]) for r in rows))
    models = list(MODEL_STYLE.keys())

    def _get_series(model, col):
        points = {}
        for r in rows:
            if r["model"] == model:
                lv = float(r["noise_level"])
                v  = _float(r.get(col))
                if v is not None:
                    points[lv] = v
        return [points.get(lv) for lv in levels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    metric_labels = {
        "kendall_tau":   "Kendall τ",
        "top1_acc":      "Top-1 Accuracy",
        "ndcg":          "NDCG@4",
        "pairwise_acc_k4":       "Pairwise Acc (K=4 held-out)",
        "pairwise_acc_binarized":"Pairwise Acc (binarized)",
        "exact_match":   "Exact Match",
    }

    for ax_idx, (col, title) in enumerate(
            [(metric, metric_labels.get(metric, metric)),
             (metric2, metric_labels.get(metric2, metric2))]):
        ax = axes[ax_idx]
        for model in models:
            ys = _get_series(model, col)
            style = MODEL_STYLE[model]
            valid = [(lv, y) for lv, y in zip(levels, ys) if y is not None]
            if valid:
                xs, ys_v = zip(*valid)
                ax.plot(xs, ys_v,
                        color=style["color"], marker=style["marker"],
                        linestyle=style["linestyle"], label=style["label"],
                        linewidth=2, markersize=7)

        ax.set_xlabel("Noise Level (corruption probability)", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_xticks(levels)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.legend(fontsize=9)
        ax.set_title(f"{noise_type.replace('_', ' ').title()} Noise — {title}", fontsize=11)

    fig.suptitle(
        f"Three-Model Comparison under {noise_type.replace('_', ' ').title()} Ranking Noise",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    name = f"{noise_type}_comparison"
    _save_fig(fig, name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2.  Summary bar chart — max noise level comparison
# ---------------------------------------------------------------------------

def plot_summary_bar(noise_rows, noise_level: float = 1.0, metric: str = "kendall_tau"):
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in noise_rows if abs(float(r["noise_level"]) - noise_level) < 1e-6]
    if not rows:
        print(f"  [warn] no rows for noise_level={noise_level}, skipping summary bar")
        return

    noise_types = sorted(set(r["noise_type"] for r in rows))
    models      = list(MODEL_STYLE.keys())
    n_types     = len(noise_types)
    n_models    = len(models)

    values = {}  # (noise_type, model) -> metric value
    for r in rows:
        key = (r["noise_type"], r["model"])
        v   = _float(r.get(metric))
        if v is not None:
            values[key] = v

    width    = 0.25
    x        = np.arange(n_types)
    fig, ax  = plt.subplots(figsize=(7, 4.5))

    for i, model in enumerate(models):
        ys = [values.get((nt, model), 0.0) for nt in noise_types]
        bars = ax.bar(x + i * width, ys, width, label=MODEL_STYLE[model]["label"],
                      color=MODEL_STYLE[model]["color"], alpha=0.85)
        for bar, y in zip(bars, ys):
            if y:
                ax.text(bar.get_x() + bar.get_width() / 2, y + 0.005,
                        f"{y:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([nt.replace("_", " ").title() for nt in noise_types])
    ax.set_ylabel(f"Kendall τ (noise_level={noise_level:.1f})", fontsize=11)
    ax.set_xlabel("Noise Type", fontsize=11)
    ax.set_title(f"Three-Model Comparison at Maximum Noise (level={noise_level:.1f})", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=max(0.0, min(values.values(), default=0) - 0.05) if values else 0.0)

    plt.tight_layout()
    _save_fig(fig, "summary_bar")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3.  RewardBench grouped bar chart
# ---------------------------------------------------------------------------

def plot_rewardbench(clean_rows):
    import matplotlib.pyplot as plt
    import numpy as np

    # Require rb_overall column and at least one non-NaN value
    rb_cols = ["rb_chat", "rb_chat_hard", "rb_safety", "rb_reasoning", "rb_overall"]
    has_rb  = any(_float(r.get("rb_overall")) is not None for r in clean_rows)
    if not has_rb:
        print("  [warn] no RewardBench results; skipping rewardbench plot")
        return

    categories  = ["Chat", "Chat Hard", "Safety", "Reasoning", "Overall"]
    col_keys    = ["rb_chat", "rb_chat_hard", "rb_safety", "rb_reasoning", "rb_overall"]
    models      = [r["model"] for r in clean_rows]

    n_cats   = len(categories)
    n_models = len(models)
    width    = 0.25
    x        = np.arange(n_cats)
    fig, ax  = plt.subplots(figsize=(9, 4.5))

    for i, row in enumerate(clean_rows):
        model = row["model"]
        ys = [_float(row.get(k)) or 0 for k in col_keys]
        bars = ax.bar(x + i * width, ys, width,
                      label=MODEL_STYLE.get(model, {}).get("label", model),
                      color=MODEL_STYLE.get(model, {}).get("color", "grey"),
                      alpha=0.85)
        for bar, y in zip(bars, ys):
            if y:
                ax.text(bar.get_x() + bar.get_width() / 2, y + 0.005,
                        f"{y:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Pairwise Accuracy", fontsize=11)
    ax.set_title("RewardBench Evaluation (allenai/reward-bench)", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6, label="Random baseline")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, "rewardbench")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4.  Phase 1 clean results bar chart
# ---------------------------------------------------------------------------

def plot_clean_comparison(clean_rows):
    """Bar chart comparing 3 models on all 5 ranking metrics (clean eval)."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not clean_rows:
        print("  [warn] no clean results; skipping clean comparison plot")
        return

    metrics = ["top1_acc", "exact_match", "kendall_tau", "ndcg", "pairwise_acc_k4"]
    metric_labels = ["Top-1 Acc", "Exact Match", "Kendall τ", "NDCG@4", "Pairwise Acc (K4)"]
    models  = [r["model"] for r in clean_rows]

    n_metrics = len(metrics)
    n_models  = len(models)
    width     = 0.25
    x         = np.arange(n_metrics)
    fig, ax   = plt.subplots(figsize=(10, 4.5))

    for i, row in enumerate(clean_rows):
        model = row["model"]
        ys = [_float(row.get(m)) or 0 for m in metrics]
        bars = ax.bar(x + i * width, ys, width,
                      label=MODEL_STYLE.get(model, {}).get("label", model),
                      color=MODEL_STYLE.get(model, {}).get("color", "grey"),
                      alpha=0.85)
        for bar, y in zip(bars, ys):
            if y:
                ax.text(bar.get_x() + bar.get_width() / 2, y + 0.005,
                        f"{y:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Three-Model Comparison — Clean Data (1000 steps)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0.0, 1.05)

    plt.tight_layout()
    _save_fig(fig, "clean_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no_rewardbench", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not available; install with: pip install matplotlib")
        sys.exit(1)

    clean_path = os.path.join(_FORMAL_DIR, "formal_clean_results.csv")
    noise_path = os.path.join(_FORMAL_DIR, "noise_sweep_results.csv")

    clean_rows = _load_csv(clean_path)
    noise_rows = _load_csv(noise_path)

    if not clean_rows and not noise_rows:
        print("No result CSVs found. Run train_formal.py and run_formal_noise_sweep.py first.")
        sys.exit(0)

    os.makedirs(_PLOT_DIR, exist_ok=True)

    if clean_rows:
        print(f"Phase 1 clean results: {len(clean_rows)} model(s)")
        plot_clean_comparison(clean_rows)
        if not args.no_rewardbench:
            plot_rewardbench(clean_rows)

    if noise_rows:
        print(f"Phase 2 noise sweep results: {len(noise_rows)} run(s)")
        for nt in ["near_tie", "top_rank"]:
            plot_noise_comparison(noise_rows, noise_type=nt,
                                  metric="kendall_tau", metric2="top1_acc")
        plot_summary_bar(noise_rows, noise_level=1.0, metric="kendall_tau")

    print(f"\nAll plots saved to: {_PLOT_DIR}/")


if __name__ == "__main__":
    main()
