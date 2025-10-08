#!/usr/bin/env python3
"""
Generate a table showing the probability of achieving ACD-AI by each calendar quarter.

Inputs:
  - --run-dir DIR: directory containing rollouts.jsonl (outputs/<timestamp>/)
  - --rollouts FILE: explicit path to rollouts.jsonl

Outputs (written into the run directory):
  - sc_by_quarter.csv
  - sc_by_quarter.html (Bootstrap-styled table fragment)

Notes:
  - Probabilities are computed over ALL rollouts, including those with no ACD-AI.
  - "P(by end)" is cumulative up to the end of the quarter.
  - A final "No ACD-AI" row is included showing probability of no ACD-AI by end of data.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from datetime import datetime, timedelta


def _resolve_rollouts_path(run_dir: Optional[Path], rollouts: Optional[Path]) -> Path:
    if rollouts is not None:
        rp = rollouts
    elif run_dir is not None:
        rp = run_dir / "rollouts.jsonl"
    else:
        raise ValueError("Provide --run-dir or --rollouts")
    if not rp.exists():
        raise FileNotFoundError(f"rollouts.jsonl not found: {rp}")
    return rp


def _decimal_year_to_datetime(decimal_year: float) -> datetime:
    year = int(np.floor(decimal_year))
    frac = float(decimal_year - year)
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    total_seconds = (end - start).total_seconds()
    dt = start + timedelta(seconds=frac * total_seconds)
    return dt


def _quarter_index(dt: datetime) -> int:
    # Unique index across years: (year * 4) + (quarter - 1)
    q = (dt.month - 1) // 3 + 1
    return dt.year * 4 + (q - 1)


def _quarter_bounds_from_index(qidx: int) -> Tuple[datetime, datetime]:
    year = qidx // 4
    q = (qidx % 4) + 1
    start_month = (q - 1) * 3 + 1
    start = datetime(year, start_month, 1)
    if q == 4:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, start_month + 3, 1)
    return start, end


@dataclass
class QuarterRow:
    label: str
    start: datetime
    end: datetime
    count_in_quarter: int
    p_in_quarter: float
    p_by_end: float


def _read_aa_times(rollouts_path: Path) -> Tuple[List[float], int, int]:
    aa_times: List[float] = []
    num_no_sc = 0
    total = 0
    with rollouts_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            res = obj.get("results") if isinstance(obj, dict) else None
            if not isinstance(res, dict):
                continue
            val = res.get("aa_time")
            try:
                v = float(val) if val is not None else np.nan
            except Exception:
                v = np.nan
            if np.isfinite(v):
                aa_times.append(float(v))
            else:
                num_no_sc += 1
    # Recompute total from counts in case of parsing issues
    total = len(aa_times) + num_no_sc
    return aa_times, num_no_sc, total


def compute_quarter_table(aa_times: List[float], num_no_sc: int) -> Tuple[List[QuarterRow], float]:
    total = len(aa_times) + num_no_sc
    if total == 0 or len(aa_times) == 0:
        return [], 0.0

    dts = [_decimal_year_to_datetime(v) for v in aa_times]
    qidx_counts: Dict[int, int] = {}
    for dt in dts:
        qi = _quarter_index(dt)
        qidx_counts[qi] = qidx_counts.get(qi, 0) + 1

    all_qidx = sorted(qidx_counts.keys())
    if not all_qidx:
        return [], 0.0

    first = all_qidx[0]
    last = all_qidx[-1]
    rows: List[QuarterRow] = []
    cum = 0
    for qi in range(first, last + 1):
        start, end = _quarter_bounds_from_index(qi)
        label = f"{start.year} Q{((qi % 4) + 1)}"
        cnt = qidx_counts.get(qi, 0)
        cum += cnt
        p_in = cnt / total
        p_by_end = cum / total
        rows.append(QuarterRow(label=label, start=start, end=end, count_in_quarter=cnt, p_in_quarter=p_in, p_by_end=p_by_end))

    return rows, (num_no_sc / total)


def write_csv(rows: List[QuarterRow], p_no_sc: float, out_dir: Path) -> Path:
    path = out_dir / "sc_by_quarter.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quarter", "start_date", "end_date", "count", "p_in_quarter", "p_by_end", "p_no_sc"])
        for r in rows:
            w.writerow([
                r.label,
                r.start.date().isoformat(),
                (r.end - timedelta(days=1)).date().isoformat(),
                r.count_in_quarter,
                f"{r.p_in_quarter:.6f}",
                f"{r.p_by_end:.6f}",
                f"{p_no_sc:.6f}",
            ])
    return path


def write_html(rows: List[QuarterRow], p_no_sc: float, out_dir: Path) -> Path:
    path = out_dir / "sc_by_quarter.html"
    if not rows:
        html = (
            '<div class="text-muted small">No SC data available to compute quarter probabilities.</div>'
        )
        path.write_text(html, encoding="utf-8")
        return path

    def fmt_pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    # Standalone fragment styled to match Bootstrap tables in the app
    parts: List[str] = []
    parts.append('<div class="table-responsive">')
    parts.append('<table class="table table-sm align-middle">')
    parts.append('<thead><tr>\n'
                 '<th>Quarter</th>\n'
                 '<th>Range</th>\n'
                 '<th class="text-end">Count</th>\n'
                 '<th class="text-end">P(in quarter)</th>\n'
                 '<th class="text-end">P(by end)</th>\n'
                 '</tr></thead>')
    parts.append('<tbody>')
    for r in rows:
        rng = f"{r.start.date().isoformat()} – {(r.end - timedelta(days=1)).date().isoformat()}"
        parts.append(
            '<tr>'
            f'<td>{r.label}</td>'
            f'<td>{rng}</td>'
            f'<td class="text-end">{r.count_in_quarter}</td>'
            f'<td class="text-end">{fmt_pct(r.p_in_quarter)}</td>'
            f'<td class="text-end">{fmt_pct(r.p_by_end)}</td>'
            '</tr>'
        )
    # Footer row for No SC probability
    parts.append(
        '<tr class="table-light">'
        '<td colspan="3"><span class="text-muted small">No SC (ever)</span></td>'
        f'<td class="text-end" colspan="2"><strong>{fmt_pct(p_no_sc)}</strong></td>'
        '</tr>'
    )
    parts.append('</tbody></table></div>')
    html = "".join(parts)
    path.write_text(html, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ACD-AI-by-quarter probability table")
    p.add_argument("--run-dir", type=Path, default=None, help="Output run directory containing rollouts.jsonl")
    p.add_argument("--rollouts", type=Path, default=None, help="Explicit path to rollouts.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rollouts_path = _resolve_rollouts_path(args.run_dir, args.rollouts)
    out_dir = rollouts_path.parent

    aa_times, num_no_sc, total = _read_aa_times(rollouts_path)
    print(f"Loaded {len(aa_times)} finite ACD-AI times (+{num_no_sc} No ACD-AI) from {rollouts_path}")
    rows, p_no_sc = compute_quarter_table(aa_times, num_no_sc)
    csv_path = write_csv(rows, p_no_sc, out_dir)
    html_path = write_html(rows, p_no_sc, out_dir)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {html_path}")


if __name__ == "__main__":
    main()


