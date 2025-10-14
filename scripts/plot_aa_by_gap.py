#!/usr/bin/env python3
"""
Plot AA-time histograms split by the sampled include_gap parameter.

Usage examples:
  python scripts/plot_aa_by_gap.py --run-dir outputs/20251008_201914
  python scripts/plot_aa_by_gap.py --rollouts outputs/20251008_201914/rollouts.jsonl

This script reads rollouts.jsonl, separates rollouts by parameters.include_gap
("gap" vs "no gap"), and saves two histograms of ACD-AI times to the run dir:
  - aa_time_hist_gap.png
  - aa_time_hist_no_gap.png

It reuses the existing plotting utility from plot_rollouts.py.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Ensure we can import sibling module when invoked from repo root or elsewhere
try:
    from plot_rollouts import plot_aa_time_histogram  # type: ignore
except Exception:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plot_rollouts import plot_aa_time_histogram  # type: ignore


def _resolve_rollouts_path(run_dir: Optional[Path], rollouts: Optional[Path]) -> Path:
    if rollouts is not None:
        p = rollouts
    elif run_dir is not None:
        p = run_dir / "rollouts.jsonl"
    else:
        raise ValueError("Provide --run-dir or --rollouts")
    if not p.exists():
        raise FileNotFoundError(f"rollouts.jsonl not found: {p}")
    return p


def _normalize_include_gap(value: object) -> Optional[str]:
    """Return canonical 'gap' or 'no gap' if recognizable, else None."""
    try:
        if isinstance(value, str):
            v = value.strip().lower().replace("_", " ")
            if v in ("gap", "yes", "true", "1"):
                return "gap"
            if v in ("no gap", "no", "false", "0"):
                return "no gap"
        elif isinstance(value, (int, float, bool)):
            return "gap" if bool(value) else "no gap"
    except Exception:
        pass
    return None


def _read_aa_times_by_gap(rollouts_file: Path) -> Tuple[
    Tuple[List[float], int, Optional[float]],
    Tuple[List[float], int, Optional[float]],
]:
    """Read AA times split by include_gap from rollouts.jsonl.

    Returns:
        ((aa_gap, no_sc_gap, sim_end_gap), (aa_nogap, no_sc_nogap, sim_end_nogap))
    """
    aa_gap: List[float] = []
    aa_nogap: List[float] = []
    num_no_sc_gap: int = 0
    num_no_sc_nogap: int = 0
    sim_end_gap: List[float] = []
    sim_end_nogap: List[float] = []

    with rollouts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            params = rec.get("parameters")
            results = rec.get("results")
            if not isinstance(params, dict) or not isinstance(results, dict):
                continue

            inc = _normalize_include_gap(params.get("include_gap"))
            if inc not in ("gap", "no gap"):
                # Unrecognized or missing; skip this record
                continue

            # Track typical simulation end time per subset
            times_array = results.get("times")
            if times_array is not None and isinstance(times_array, list) and len(times_array) > 0:
                try:
                    end_t = float(times_array[-1])
                    if np.isfinite(end_t):
                        if inc == "gap":
                            sim_end_gap.append(end_t)
                        else:
                            sim_end_nogap.append(end_t)
                except Exception:
                    pass

            # Extract AA time
            aa_time = results.get("aa_time")
            try:
                x = float(aa_time) if aa_time is not None else np.nan
            except (TypeError, ValueError):
                x = np.nan

            if np.isfinite(x):
                if inc == "gap":
                    aa_gap.append(x)
                else:
                    aa_nogap.append(x)
            else:
                if inc == "gap":
                    num_no_sc_gap += 1
                else:
                    num_no_sc_nogap += 1

    typical_sim_end_gap = float(np.median(sim_end_gap)) if sim_end_gap else None
    typical_sim_end_nogap = float(np.median(sim_end_nogap)) if sim_end_nogap else None

    return (
        (aa_gap, num_no_sc_gap, typical_sim_end_gap),
        (aa_nogap, num_no_sc_nogap, typical_sim_end_nogap),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot AA-time histograms split by include_gap")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=Path, default=None, help="Output run directory containing rollouts.jsonl")
    g.add_argument("--rollouts", type=Path, default=None, help="Explicit path to rollouts.jsonl")
    p.add_argument("--bins", type=int, default=50, help="Histogram bins")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rollouts_path = _resolve_rollouts_path(args.run_dir, args.rollouts)
    out_dir = rollouts_path.parent

    (aa_gap, no_sc_gap, sim_end_gap), (aa_nogap, no_sc_nogap, sim_end_nogap) = _read_aa_times_by_gap(rollouts_path)

    # Report basic stats
    total_gap = len(aa_gap) + no_sc_gap
    total_nogap = len(aa_nogap) + no_sc_nogap
    print(f"Loaded {total_gap} records for include_gap=gap: {len(aa_gap)} finite AA times, +{no_sc_gap} No ACD-AI")
    print(f"Loaded {total_nogap} records for include_gap=no gap: {len(aa_nogap)} finite AA times, +{no_sc_nogap} No ACD-AI")

    out_gap = out_dir / "aa_time_hist_gap.png"
    out_nogap = out_dir / "aa_time_hist_no_gap.png"

    if total_gap > 0:
        plot_aa_time_histogram(aa_gap, num_no_sc=no_sc_gap, out_path=out_gap, bins=int(args.bins), sim_end=sim_end_gap)
        print(f"Saved {out_gap}")
    else:
        print("Warning: No records for include_gap=gap; skipping plot")

    if total_nogap > 0:
        plot_aa_time_histogram(aa_nogap, num_no_sc=no_sc_nogap, out_path=out_nogap, bins=int(args.bins), sim_end=sim_end_nogap)
        print(f"Saved {out_nogap}")
    else:
        print("Warning: No records for include_gap=no gap; skipping plot")


if __name__ == "__main__":
    main()




