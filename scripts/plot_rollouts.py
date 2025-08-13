#!/usr/bin/env python3
"""
Plot utilities for analyzing batch rollout results.

Currently supported:
- Plot distribution (histogram) of SC times from a rollouts.jsonl file

Usage examples:
  python scripts/plot_rollouts.py --run-dir outputs/20250813_020347 \
    --out outputs/20250813_020347/sc_time_hist.png

  python scripts/plot_rollouts.py --rollouts outputs/20250813_020347/rollouts.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

# Use a non-interactive backend in case this runs on a headless server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _resolve_rollouts_path(run_dir: Optional[str], rollouts_path: Optional[str]) -> Path:
    if rollouts_path:
        p = Path(rollouts_path)
        if not p.exists():
            raise FileNotFoundError(f"rollouts file does not exist: {p}")
        return p
    if not run_dir:
        raise ValueError("Either --run-dir or --rollouts must be provided")
    d = Path(run_dir)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"run directory does not exist or is not a directory: {d}")
    p = d / "rollouts.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"rollouts.jsonl not found in run directory: {d}")
    return p


def _read_sc_times(rollouts_file: Path) -> List[float]:
    sc_times: List[float] = []
    with rollouts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            results = rec.get("results")
            if not isinstance(results, dict):
                continue
            sc_time = results.get("sc_time")
            if sc_time is None:
                continue
            try:
                x = float(sc_time)
            except (TypeError, ValueError):
                continue
            if np.isfinite(x):
                sc_times.append(x)
    return sc_times


def plot_sc_time_histogram(sc_times: List[float], out_path: Path, bins: int = 50, title: Optional[str] = None) -> None:
    if len(sc_times) == 0:
        raise ValueError("No finite SC times found to plot")
    data = np.asarray(sc_times, dtype=float)

    plt.figure(figsize=(9, 5.5))
    plt.hist(data, bins=bins, edgecolor="black", alpha=0.75)
    plt.xlabel("SC Time (decimal year)")
    plt.ylabel("Count")
    plt.title(title or "Distribution of SC Times")
    plt.grid(True, axis="y", alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tools for batch rollout results")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=str, default=None, help="Path to a single rollout run directory containing rollouts.jsonl")
    g.add_argument("--rollouts", type=str, default=None, help="Path directly to a rollouts.jsonl file")
    parser.add_argument("--out", type=str, default=None, help="Output image path (PNG). Defaults to <run-dir>/sc_time_hist.png or alongside rollouts.jsonl")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollouts_path = _resolve_rollouts_path(args.run_dir, args.rollouts)

    # Default output path resolution
    if args.out is not None:
        out_path = Path(args.out)
    else:
        default_dir = rollouts_path.parent
        out_path = default_dir / "sc_time_hist.png"

    sc_times = _read_sc_times(rollouts_path)

    # Print basic stats for quick inspection
    if len(sc_times) > 0:
        arr = np.asarray(sc_times, dtype=float)
        q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
        print(f"Loaded {len(arr)} SC times from {rollouts_path}")
        print(f"Min/Median/Max: {arr.min():.3f} / {q50:.3f} / {arr.max():.3f}")
        print(f"P10/P90: {q10:.3f} / {q90:.3f}")

    plot_sc_time_histogram(sc_times, out_path=out_path, bins=int(args.bins))
    print(f"Saved histogram to: {out_path}")


if __name__ == "__main__":
    main()

