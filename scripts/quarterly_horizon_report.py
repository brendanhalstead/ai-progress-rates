#!/usr/bin/env python3
"""
Generate a formatted quarterly time horizon report from a batch run.

Inputs:
  - --run-dir DIR  (directory containing rollouts.jsonl)
  - --rollouts FILE (explicit path to rollouts.jsonl)

Output: prints formatted text to stdout (suitable for copy/paste),
        optionally writes CSV of the computed per-quarter stats with --out-csv.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


HOUR_MIN = 60.0
# Work-time units (match app plotting ticks):
# - 1 work day  = 8 hours = 480 minutes
# - 1 work week = 5 work days = 2,400 minutes
# - 1 work month ~ 173 hours = 10,380 minutes
# - 1 work year  ~ 2,076 hours = 124,560 minutes
WORK_DAY_MIN = 480.0
WORK_WEEK_MIN = 2400.0
WORK_MONTH_MIN = 10380.0
WORK_YEAR_MIN = 124560.0


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


def _read_trajectories(rollouts_path: Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    times_arr: Optional[np.ndarray] = None
    trajectories: List[np.ndarray] = []
    with rollouts_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            results = rec.get("results") if isinstance(rec, dict) else None
            if not isinstance(results, dict):
                continue
            t = results.get("times")
            h = results.get("horizon_lengths")
            if t is None or h is None:
                continue
            try:
                t_np = np.asarray(t, dtype=float)
                h_np = np.asarray(h, dtype=float)
            except Exception:
                continue
            if t_np.ndim != 1 or h_np.ndim != 1 or t_np.size != h_np.size:
                continue
            if times_arr is None:
                times_arr = t_np
            trajectories.append(h_np)
    if times_arr is None or len(trajectories) == 0:
        raise RuntimeError("No horizon trajectories found in rollouts file")
    return times_arr, trajectories


def _clean_and_stack(trajectories: List[np.ndarray]) -> np.ndarray:
    min_horizon_minutes = 0.001
    max_work_year_minutes = float(120000 * 52 * 40 * 60)
    stacked = np.vstack([
        np.clip(np.where(np.isfinite(arr) & (arr > 0), arr, np.nan), min_horizon_minutes, max_work_year_minutes).astype(float)
        for arr in trajectories
    ])
    return stacked


def _decimal_year_to_datetime(decimal_year: float) -> datetime:
    year = int(np.floor(decimal_year))
    frac = float(decimal_year - year)
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    total_seconds = (end - start).total_seconds()
    return start + timedelta(seconds=frac * total_seconds)


def _datetime_to_decimal_year(dt: datetime) -> float:
    start = datetime(dt.year, 1, 1)
    end = datetime(dt.year + 1, 1, 1)
    frac = (dt - start).total_seconds() / (end - start).total_seconds()
    return float(dt.year + frac)


def _quarter_start(dt: datetime) -> datetime:
    month = ((dt.month - 1) // 3) * 3 + 1
    return datetime(dt.year, month, 1)


def _quarter_end_from_start(qs: datetime) -> datetime:
    if qs.month <= 9:
        nxt = datetime(qs.year, qs.month + 3, 1)
    else:
        nxt = datetime(qs.year + 1, 1, 1)
    return nxt - timedelta(days=1)


def _list_quarters(times_arr: np.ndarray) -> List[datetime]:
    dt_min = _decimal_year_to_datetime(float(np.min(times_arr)))
    dt_max = _decimal_year_to_datetime(float(np.max(times_arr)))
    cur = _quarter_start(dt_min)
    quarters: List[datetime] = []
    while cur <= dt_max:
        quarters.append(cur)
        if cur.month <= 9:
            cur = datetime(cur.year, cur.month + 3, 1)
        else:
            cur = datetime(cur.year + 1, 1, 1)
    return quarters


def _fmt_minutes(m: float) -> str:
    if not (isinstance(m, (int, float)) and math.isfinite(m)):
        return "NaN"
    m = max(0.0, float(m))
    if m < 1.0:
        return f"{m * 60.0:.0f}s"
    if m < 60.0:
        return f"{m:.1f}min" if m < 10 else f"{m:.0f}min"
    # Hours (< 1 work day)
    if m < WORK_DAY_MIN:
        val = m / HOUR_MIN
        return f"{val:.1f}hr" if val < 10 else f"{val:.0f}hr"
    # Work days (< 1 work week)
    if m < WORK_WEEK_MIN:
        val = m / WORK_DAY_MIN
        return f"{val:.1f}d" if val < 10 else f"{val:.0f}d"
    # Work weeks (< 1 work month)
    if m < WORK_MONTH_MIN:
        val = m / WORK_WEEK_MIN
        return f"{val:.1f}wk" if val < 10 else f"{val:.1f}wk"
    # Work months (< 1 work year)
    if m < WORK_YEAR_MIN:
        val = m / WORK_MONTH_MIN
        return f"{val:.1f}mo" if val < 10 else f"{val:.1f}mo"
    # Work years and above
    val = m / WORK_YEAR_MIN
    return f"{val:.1f}yr" if val < 10 else f"{val:.1f}yr"


@dataclass
class QuarterStats:
    label: str
    year: int
    qnum: int
    end_date: date
    mean: float
    median: float
    p5: float
    p95: float


def compute_quarter_stats(times_arr: np.ndarray, stacked: np.ndarray) -> List[QuarterStats]:
    quarters = _list_quarters(times_arr)
    rows: List[QuarterStats] = []
    for qs in quarters:
        # Evaluate at quarter END to align with the displayed end date
        qe = _quarter_end_from_start(qs)
        q_dec = _datetime_to_decimal_year(qe)
        idx = int(np.argmin(np.abs(times_arr - q_dec)))
        col = stacked[:, idx]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            mean = median = p5 = p95 = float("nan")
        else:
            mean = float(np.nanmean(finite))
            median = float(np.nanmedian(finite))
            p5 = float(np.nanpercentile(finite, 5))
            p95 = float(np.nanpercentile(finite, 95))
        qnum = (qs.month - 1) // 3 + 1
        rows.append(QuarterStats(
            label=f"{qs.year}Q{qnum}",
            year=qs.year,
            qnum=qnum,
            end_date=qe.date(),
            mean=mean,
            median=median,
            p5=p5,
            p95=p95,
        ))
    return rows


def print_report(rows: List[QuarterStats], baseline_label: str, baseline_date: date) -> None:
    print("METR Task Horizon Predictions (50% reliability) - PUBLIC MODELS")
    print("=" * 70)
    print(f"Baseline: {baseline_label}")
    print("=" * 70)
    print()
    print(f"{'Quarter':<8}  {'End Date':<12}  {'Median':<9}  {'90% CI':<24}  {'Days':>6}")
    print("-" * 85)

    prev_year: Optional[int] = None
    for r in rows:
        if prev_year is not None and r.year != prev_year:
            print()
        prev_year = r.year
        med_s = _fmt_minutes(r.median)
        lo_s = _fmt_minutes(r.p5)
        hi_s = _fmt_minutes(r.p95)
        ci_str = f"[{lo_s} - {hi_s}]"
        days = (r.end_date - baseline_date).days
        print(f"{r.label:<8}  {r.end_date.isoformat():<12}  {med_s:<9}  {ci_str:<24}  {days:>6}")

    print()
    print("=" * 70)
    print("Key Milestones (Median Estimates) - PUBLIC MODELS")
    print("=" * 70)

    thresholds = [
        ("2 hours", 2 * HOUR_MIN),
        ("4 hours", 4 * HOUR_MIN),
        ("1 day", 1 * WORK_DAY_MIN),
        ("2 days", 2 * WORK_DAY_MIN),
        ("3 days", 3 * WORK_DAY_MIN),
        ("1 week", 1 * WORK_WEEK_MIN),
        ("1 month", 1 * WORK_MONTH_MIN),
    ]

    # Build per-quarter finite arrays to compute coverage at the milestone quarter
    # Note: We recompute the index for each row to maintain consistency
    # with compute_quarter_stats
    def rows_to_finite_arrays() -> List[np.ndarray]:
        # This function requires closure vars: times_arr and stacked; handled in main where needed
        return []  # placeholder; replaced in main


def write_csv(rows: List[QuarterStats], out_csv: Path) -> None:
    import csv
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quarter", "end_date", "mean_minutes", "median_minutes", "ci5_minutes", "ci95_minutes"])
        for r in rows:
            w.writerow([
                r.label,
                r.end_date.isoformat(),
                f"{r.mean:.6f}",
                f"{r.median:.6f}",
                f"{r.p5:.6f}",
                f"{r.p95:.6f}",
            ])


def main() -> None:
    p = argparse.ArgumentParser(description="Formatted quarterly horizon report")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=Path, default=None)
    g.add_argument("--rollouts", type=Path, default=None)
    p.add_argument("--baseline-label", type=str, default="GPT5 at 2.3hr on 2025-08-07")
    p.add_argument("--baseline-date", type=str, default="2025-08-07", help="YYYY-MM-DD used for 'Days' column")
    p.add_argument("--out-csv", type=Path, default=None, help="Optional CSV path for per-quarter stats")
    args = p.parse_args()

    rollouts_path = _resolve_rollouts_path(args.run_dir, args.rollouts)
    times_arr, trajectories = _read_trajectories(rollouts_path)
    stacked = _clean_and_stack(trajectories)

    rows = compute_quarter_stats(times_arr, stacked)

    # Print main table
    baseline_dt = datetime.strptime(args.baseline_date, "%Y-%m-%d").date()
    print_report(rows, args.baseline_label, baseline_dt)

    # Print milestones with coverage
    # Build per-quarter finite values for coverage calculation
    quarters = _list_quarters(times_arr)
    per_q_finite: List[np.ndarray] = []
    for qs in quarters:
        q_dec = _datetime_to_decimal_year(qs)
        idx = int(np.argmin(np.abs(times_arr - q_dec)))
        col = stacked[:, idx]
        finite = col[np.isfinite(col)]
        per_q_finite.append(finite)

    thresholds = [
        ("2 hours", 2 * HOUR_MIN),
        ("4 hours", 4 * HOUR_MIN),
        ("1 day", 1 * WORK_DAY_MIN),
        ("2 days", 2 * WORK_DAY_MIN),
        ("3 days", 3 * WORK_DAY_MIN),
        ("1 week", 1 * WORK_WEEK_MIN),
        ("1 month", 1 * WORK_MONTH_MIN),
    ]

    for name, thr in thresholds:
        hit_label: Optional[str] = None
        hit_date: Optional[str] = None
        hit_idx: Optional[int] = None
        for i, r in enumerate(rows):
            med = r.median
            if not (isinstance(med, (int, float)) and math.isfinite(med)):
                continue
            if med >= thr:
                hit_label = r.label
                hit_date = r.end_date.isoformat()
                hit_idx = i
                break
        if hit_label is None or hit_idx is None:
            continue
        vals = per_q_finite[hit_idx]
        pct = 0.0 if vals.size == 0 else 100.0 * float(np.sum(vals >= thr)) / float(vals.size)
        dt_fmt = datetime.strptime(hit_date, "%Y-%m-%d").strftime("%B %d, %Y")
        print(f"{name:<12} -> {hit_label} ({dt_fmt}, {pct:.0f}% of samples)")

    if args.out_csv is not None:
        write_csv(rows, args.out_csv)


if __name__ == "__main__":
    main()
