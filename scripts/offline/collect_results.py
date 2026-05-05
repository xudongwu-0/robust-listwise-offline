#!/usr/bin/env python3
"""
scripts/offline/collect_results.py

Read the offline noise-sweep CSV and print a human-readable pivot table.

Usage
-----
  python scripts/offline/collect_results.py               # 0.6B results
  python scripts/offline/collect_results.py --size 8b     # 8B results
  python scripts/offline/collect_results.py --metric kendall_tau top1_accuracy

Output
------
Pivot table: rows = methods, columns = noise conditions,
cells = Kendall τ (or chosen metric).
"""

import argparse
import os
import sys

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required: pip install pandas")


# Canonical display ordering
CONDITION_ORDER = ["clean", "near_tie/0.4", "near_tie/1.0", "top_rank/0.4", "top_rank/1.0"]

METHOD_ORDER = [
    "nominal_bt",
    "nominal_pl",
    "robust_pl_rho050",
    "robust_pl",
    "tv_dr_dpo_rho005",
    "tv_dr_dpo_rho010",
    "kldpo_tau050",
    "kldpo_tau100",
]


def _condition_label(row) -> str:
    nt = str(row.get("noise_type", "")).strip()
    nl = row.get("noise_level", None)
    if nt == "clean" or str(nt) == "nan":
        return "clean"
    if nl is None or str(nl) == "nan":
        return nt
    return f"{nt}/{float(nl):.1f}"


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"Result CSV not found: {path}\nRun the sweep first.")
    df = pd.read_csv(path)
    df["condition"] = df.apply(_condition_label, axis=1)
    return df


def print_pivot(df: pd.DataFrame, metric: str) -> None:
    if metric not in df.columns:
        available = [c for c in df.columns if c not in ("model", "base_model",
                                                          "noise_type", "noise_level",
                                                          "condition", "timestamp")]
        sys.exit(f"Metric '{metric}' not found in CSV. Available: {available}")

    pivot = df.pivot_table(index="model", columns="condition", values=metric, aggfunc="last")

    # Reorder rows / columns where present
    row_order = [m for m in METHOD_ORDER if m in pivot.index] + \
                [m for m in pivot.index if m not in METHOD_ORDER]
    col_order = [c for c in CONDITION_ORDER if c in pivot.columns] + \
                [c for c in pivot.columns if c not in CONDITION_ORDER]
    pivot = pivot.reindex(index=row_order, columns=col_order)

    print(f"\n=== {metric} ===")
    print(pivot.to_string(float_format="{:.4f}".format))
    print()


def main():
    p = argparse.ArgumentParser(description="Collect and display offline sweep results.")
    p.add_argument("--size", default="05b", choices=["05b", "8b"],
                   help="Model size to display (default: 05b)")
    p.add_argument("--metric", nargs="+",
                   default=["kendall_tau", "top1_accuracy", "pair_acc_bin"],
                   help="Metrics to display")
    p.add_argument("--csv", default=None,
                   help="Override CSV path (default: outputs/qwen3/<size>/noise_sweep/...)")
    args = p.parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        csv_path = f"outputs/qwen3/{args.size}/noise_sweep/qwen3_{args.size}_noise_sweep.csv"

    print(f"Loading: {csv_path}")
    df = load_csv(csv_path)
    print(f"Rows: {len(df)}  Methods: {sorted(df['model'].unique())}")

    for metric in args.metric:
        print_pivot(df, metric)


if __name__ == "__main__":
    main()
