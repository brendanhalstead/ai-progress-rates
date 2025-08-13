#!/usr/bin/env python3
"""
Plot utilities for analyzing batch rollout results.

Currently supported:
- Plot distribution (histogram) of SC times from a rollouts.jsonl file
 - Plot distribution (histogram) of horizon length at SC across rollouts

Usage examples:
  python scripts/plot_rollouts.py --run-dir outputs/20250813_020347 \
    --out outputs/20250813_020347/sc_time_hist.png

  python scripts/plot_rollouts.py --rollouts outputs/20250813_020347/rollouts.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Use a non-interactive backend in case this runs on a headless server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from datetime import datetime, timedelta
import yaml


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


def _decimal_year_to_date_string(decimal_year: float) -> str:
    """Convert a decimal year (e.g., 2031.5) to calendar date string YYYY-MM-DD.

    Handles leap years by interpolating between Jan 1 of the year and Jan 1 of the next year.
    """
    year = int(np.floor(decimal_year))
    frac = float(decimal_year - year)
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    total_seconds = (end - start).total_seconds()
    dt = start + timedelta(seconds=frac * total_seconds)
    return dt.date().isoformat()


def _now_decimal_year() -> float:
    """Return the current time as a decimal year (UTC)."""
    now = datetime.utcnow()
    start = datetime(now.year, 1, 1)
    end = datetime(now.year + 1, 1, 1)
    frac = (now - start).total_seconds() / (end - start).total_seconds()
    return float(now.year + frac)


def _format_time_duration(minutes: float) -> str:
    """Pretty format a duration given in minutes."""
    if not np.isfinite(minutes):
        return "NaN"
    if minutes < 1.0:
        seconds = minutes * 60.0
        return f"{seconds:.0f} sec"
    if minutes < 60.0:
        return f"{minutes:.0f} min"
    hours = minutes / 60.0
    if hours < 24.0:
        return f"{hours:.0f} hrs"
    days = hours / 24.0
    if days < 7:
        return f"{days:.0f} days"
    weeks = days / 7.0
    if weeks < 4:
        return f"{weeks:.0f} weeks"
    months = days / 30.44
    if months < 12:
        return f"{months:.0f} months"
    years = days / 365.25
    if years < 1000:
        return f"{years:.0f} years"
    # Work-years not distinguished in label; show with comma grouping
    return f"{years:,.0f} years"


def _read_horizon_trajectories(rollouts_file: Path) -> Tuple[np.ndarray, List[np.ndarray], List[Optional[float]]]:
    """Read horizon length time series from a rollouts.jsonl file.

    Returns:
        times: common time array in decimal years
        trajectories: list of horizon length arrays (one per rollout)
    """
    trajectories: List[np.ndarray] = []
    sc_times: List[Optional[float]] = []
    common_times: Optional[np.ndarray] = None
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
            times = results.get("times")
            horizon = results.get("horizon_lengths")
            sc_time_val = results.get("sc_time")
            if times is None or horizon is None:
                continue
            try:
                times_arr = np.asarray(times, dtype=float)
                horizon_arr = np.asarray(horizon, dtype=float)
            except Exception:
                continue
            if times_arr.ndim != 1 or horizon_arr.ndim != 1 or times_arr.size != horizon_arr.size:
                continue
            if common_times is None:
                common_times = times_arr
            trajectories.append(horizon_arr)
            try:
                sc_times.append(float(sc_time_val) if sc_time_val is not None and np.isfinite(float(sc_time_val)) else None)
            except Exception:
                sc_times.append(None)
    if common_times is None or len(trajectories) == 0:
        raise ValueError("No horizon trajectories found in rollouts file")
    return common_times, trajectories, sc_times


def _get_time_tick_values_and_labels() -> Tuple[List[float], List[str]]:
    """Tick values and labels using work-time units (minutes, log scale).

    Matches get_time_tick_values_and_labels() in app.py
    (e.g., 1 week = 5 work days = 2,400 minutes; 1 year = work-year minutes).
    """
    tick_values = [
        0.033333,   # 2 sec
        0.5,        # 30 sec
        2,          # 2 min
        8,          # 8 min
        30,         # 30 min
        120,        # 2 hrs
        480,        # 8 hrs
        2400,       # 1 week (work week)
        10380,      # 1 month (work month)
        41520,      # 4 months (work months)
        124560,     # 1 year (work year)
        622800,     # 5 years (work years)
        2491200,    # 20 years (work years)
        12456000,   # 100 years (work years)
        49824000,   # 400 years (work years)
        199296000,  # 1,600 years (work years)
        797184000,  # 6,400 years (work years)
        3188736000, # 25,600 years (work years)
        14947200000,# 120,000 years (work years)
    ]
    tick_labels = [
        "2 sec",
        "30 sec",
        "2 min",
        "8 min",
        "30 min",
        "2 hrs",
        "8 hrs",
        "1 week",
        "1 month",
        "4 months",
        "1 year",
        "5 years",
        "20 years",
        "100 years",
        "400 years",
        "1,600 years",
        "6,400 years",
        "25,600 years",
        "120,000 years",
    ]
    return tick_values, tick_labels


def _load_metr_p80_points() -> Optional[List[Tuple[float, float]]]:
    """Load SOTA METR p80 horizon points as (decimal_year, p80_minutes).

    Returns None if file missing or malformed.
    """
    try:
        with open("benchmark_results.yaml", "r") as f:
            bench = yaml.safe_load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None
    results = bench.get("results") if isinstance(bench, dict) else None
    if not isinstance(results, dict):
        return None
    points: List[Tuple[float, float]] = []
    for _model_name, model_info in results.items():
        try:
            release = model_info.get("release_date")
            if isinstance(release, str):
                d = datetime.strptime(release, "%Y-%m-%d").date()
            else:
                # If already a date-like, try year/month/day attrs
                d = release
            decimal_year = float(d.year + (d.timetuple().tm_yday - 1) / 365.25)
        except Exception:
            continue
        agents = model_info.get("agents", {})
        if not isinstance(agents, dict):
            continue
        # Only include points marked SOTA
        is_sota = False
        p80_est = None
        for _agent_name, agent_data in agents.items():
            if not isinstance(agent_data, dict):
                continue
            if agent_data.get("is_sota"):
                is_sota = True
            p80 = agent_data.get("p80_horizon_length", {})
            if isinstance(p80, dict) and p80.get("estimate") is not None:
                p80_est = float(p80["estimate"])
        if is_sota and p80_est is not None and p80_est > 0:
            points.append((decimal_year, float(p80_est)))
    return points if points else None


def plot_horizon_trajectories(
    times: np.ndarray,
    trajectories: List[np.ndarray],
    out_path: Path,
    current_horizon_minutes: float = 15.0,
    alpha: float = 0.08,
    max_trajectories: int = 2000,
    overlay_metr: bool = True,
    title: Optional[str] = None,
    stop_at_sc: bool = False,
    sc_times: Optional[List[Optional[float]]] = None,
) -> None:
    """Render horizon length trajectories similar to the reference figure.

    Assumes times are decimal years and horizons are in minutes.
    """
    if len(trajectories) == 0:
        raise ValueError("No trajectories provided")

    # Clip extremely large values to avoid distorting the log scale (cap at 1e6 minutes)
    min_horizon_minutes = 0.001
    # Cap at 120,000 work-years = 120000 * 52 weeks * 40 hours -> minutes
    max_work_year_minutes = float(120000 * 52 * 40 * 60)
    # Replace non-positive and non-finite values with NaN so they don't draw lines to zero
    cleaned: List[np.ndarray] = []
    for idx, t in enumerate(trajectories):
        arr = t.astype(float)
        arr[~np.isfinite(arr)] = np.nan
        arr[arr <= 0] = np.nan
        arr = np.clip(arr, min_horizon_minutes, max_work_year_minutes)
        # If requested, mask values after this rollout's sc_time
        if stop_at_sc and sc_times is not None and idx < len(sc_times) and sc_times[idx] is not None:
            sc = sc_times[idx]
            if sc is not None:
                arr = arr.copy()
                arr[times > float(sc)] = np.nan
        cleaned.append(arr)

    # Compute median trajectory across rollouts (align by index; the batch process uses a shared grid)
    stacked = np.vstack([t for t in cleaned[:max_trajectories]])
    median_traj = np.nanmedian(stacked, axis=0)

    plt.figure(figsize=(14, 8))

    # Draw all trajectories in faint colors
    num_plot = min(len(cleaned), max_trajectories)
    for i in range(num_plot):
        plt.plot(times, cleaned[i], color=(0.2, 0.5, 0.7, alpha), linewidth=1.0)

    # Central trajectory
    plt.plot(times, median_traj, color="tab:green", linestyle="--", linewidth=2.0, label="Central Trajectory")

    # Horizontal line for current horizon
    plt.axhline(current_horizon_minutes, color="red", linewidth=2.0, label=f"Current Horizon ({int(current_horizon_minutes)} min)")

    # Vertical line for current time
    now_year = _now_decimal_year()
    plt.axvline(now_year, color="tab:blue", linestyle="--", linewidth=1.75, label="Current Time")

    # Optional METR p80 scatter
    if overlay_metr:
        points = _load_metr_p80_points()
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.scatter(xs, ys, color="black", s=18, label="External Benchmarks (p80)", zorder=10)

    # Axes and formatting
    plt.yscale("log")
    tick_values, tick_labels = _get_time_tick_values_and_labels()
    plt.yticks(tick_values, tick_labels)
    # Expand y-limit to cover data range
    finite_max = np.nanmax(stacked)
    if np.isfinite(finite_max):
        ymin = min(tick_values)
        ymax = min(max(finite_max, max(tick_values)), max_work_year_minutes)
        plt.ylim(ymin, ymax)
    plt.xlabel("Year")
    plt.ylabel("Time Horizon")
    plt.title(title or "Complete Time Horizon Extension Trajectories\n(Historical development and future projections)")
    plt.grid(True, which="both", axis="y", alpha=0.25)
    plt.legend(loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _read_horizon_at_sc(rollouts_file: Path) -> List[float]:
    """Compute horizon length at SC time for each rollout where available."""
    values: List[float] = []
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
            times = results.get("times")
            horizon = results.get("horizon_lengths")
            sc_time = results.get("sc_time")
            if times is None or horizon is None or sc_time is None:
                continue
            try:
                times_arr = np.asarray(times, dtype=float)
                horizon_arr = np.asarray(horizon, dtype=float)
                sc_t = float(sc_time)
            except Exception:
                continue
            if not (np.isfinite(sc_t) and times_arr.ndim == 1 and horizon_arr.ndim == 1 and times_arr.size == horizon_arr.size):
                continue
            if np.any(~np.isfinite(horizon_arr)):
                # Mask non-finite values for interpolation
                valid = np.isfinite(horizon_arr)
                if valid.sum() < 2:
                    continue
                times_arr = times_arr[valid]
                horizon_arr = horizon_arr[valid]
            # Clip to plotting cap to avoid absurd tails
            cap = float(120000 * 52 * 40 * 60)
            horizon_arr = np.clip(horizon_arr, 0.001, cap)
            # Interpolate horizon at sc time
            if sc_t <= times_arr.min():
                val = horizon_arr[0]
            elif sc_t >= times_arr.max():
                val = horizon_arr[-1]
            else:
                val = float(np.interp(sc_t, times_arr, horizon_arr))
            if np.isfinite(val) and val > 0:
                values.append(val)
    return values


def plot_horizon_at_sc_histogram(values: List[float], out_path: Path, bins: int = 50, title: Optional[str] = None) -> None:
    if len(values) == 0:
        raise ValueError("No horizon_at_sc values found to plot")
    data = np.asarray(values, dtype=float)
    # Define log-spaced bins between min and cap
    cap = float(120000 * 52 * 40 * 60)
    data = np.clip(data, 0.001, cap)
    xmin = max(data[data > 0].min(), 0.001)
    xmax = max(data.max(), xmin * 1.01)
    bin_edges = np.logspace(np.log10(xmin), np.log10(xmax), int(bins))

    plt.figure(figsize=(10, 6))
    counts, _, _ = plt.hist(data, bins=bin_edges, edgecolor="black", alpha=0.6, label="Histogram")

    # Percentiles and annotations
    q10, q50, q90 = np.quantile(data, [0.1, 0.5, 0.9])
    ymax = float(np.max(counts) if counts.size else 1.0)
    y_annot = ymax * 0.95
    plt.axvline(q10, color="tab:gray", linestyle="--", linewidth=1.5, label="P10")
    plt.axvline(q50, color="tab:green", linestyle="-", linewidth=1.75, label="Median")
    plt.axvline(q90, color="tab:gray", linestyle="--", linewidth=1.5, label="P90")
    plt.text(q10, y_annot, f"P10: {_format_time_duration(q10)}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))
    plt.text(q50, y_annot, f"Median: {_format_time_duration(q50)}", rotation=90, va="top", ha="right", color="tab:green", fontsize=9, backgroundcolor=(1,1,1,0.6))
    plt.text(q90, y_annot, f"P90: {_format_time_duration(q90)}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))

    # Axis formatting: log x with custom ticks
    plt.xscale("log")
    ticks, labels = _get_time_tick_values_and_labels()
    plt.xticks(ticks, labels, rotation=0)
    plt.xlabel("Horizon at SC (minutes)")
    plt.ylabel("Count")
    plt.title(title or "Distribution of Horizon Length at SC")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_sc_time_histogram(sc_times: List[float], out_path: Path, bins: int = 50, title: Optional[str] = None) -> None:
    if len(sc_times) == 0:
        raise ValueError("No finite SC times found to plot")
    data = np.asarray(sc_times, dtype=float)

    plt.figure(figsize=(10, 6))

    counts, bin_edges, _ = plt.hist(
        data,
        bins=bins,
        edgecolor="black",
        alpha=0.6,
        label="Histogram",
    )

    # KDE overlay, scaled to count space
    xs = np.linspace(data.min(), data.max(), 512)
    kde = gaussian_kde(data)
    # Assume uniform bins when bins is an int
    bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
    kde_counts = kde(xs) * len(data) * bin_width
    plt.plot(xs, kde_counts, color="tab:orange", linewidth=2.25, label="Gaussian KDE")

    # Percentiles and annotations
    q10, q50, q90 = np.quantile(data, [0.1, 0.5, 0.9])
    ymax = float(max(np.max(counts) if counts.size else 0.0, np.max(kde_counts) if kde_counts.size else 0.0))
    y_annot = ymax * 0.95 if ymax > 0 else 1.0

    # Vertical lines
    plt.axvline(q10, color="tab:gray", linestyle="--", linewidth=1.5, label="P10")
    plt.axvline(q50, color="tab:green", linestyle="-", linewidth=1.75, label="Median")
    plt.axvline(q90, color="tab:gray", linestyle="--", linewidth=1.5, label="P90")

    # Text labels showing arrival dates
    plt.text(q10, y_annot, f"P10: {_decimal_year_to_date_string(q10)}", rotation=90,
             va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))
    plt.text(q50, y_annot, f"Median: {_decimal_year_to_date_string(q50)}", rotation=90,
             va="top", ha="right", color="tab:green", fontsize=9, backgroundcolor=(1,1,1,0.6))
    plt.text(q90, y_annot, f"P90: {_decimal_year_to_date_string(q90)}", rotation=90,
             va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))

    plt.xlabel("SC Time (decimal year)")
    plt.ylabel("Count")
    plt.title(title or "Distribution of SC Times")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tools for batch rollout results")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=str, default=None, help="Path to a single rollout run directory containing rollouts.jsonl")
    g.add_argument("--rollouts", type=str, default=None, help="Path directly to a rollouts.jsonl file")
    parser.add_argument("--out", type=str, default=None, help="Output image path (PNG). Defaults vary by --mode")
    parser.add_argument("--mode", type=str, choices=["sc_hist", "horizon_trajectories", "horizon_at_sc_hist"], default="sc_hist", help="Which plot to generate")
    # Histogram options
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins for sc_hist mode")
    # Horizon trajectories options
    parser.add_argument("--current-horizon-minutes", type=float, default=15.0, help="Horizontal reference line for current horizon in minutes")
    parser.add_argument("--alpha", type=float, default=0.08, help="Transparency for individual trajectories")
    parser.add_argument("--max-trajectories", type=int, default=2000, help="Maximum number of trajectories to draw")
    parser.add_argument("--no-metr", action="store_true", help="Disable overlay of METR p80 benchmark points")
    parser.add_argument("--stop-at-sc", action="store_true", help="Mask each trajectory after its own sc_time")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollouts_path = _resolve_rollouts_path(args.run_dir, args.rollouts)

    default_dir = rollouts_path.parent
    if args.out is not None:
        out_path = Path(args.out)
    else:
        if args.mode == "sc_hist":
            out_path = default_dir / "sc_time_hist.png"
        elif args.mode == "horizon_at_sc_hist":
            out_path = default_dir / "horizon_at_sc_hist.png"
        else:
            out_path = default_dir / ("horizon_trajectories_stop_at_sc.png" if args.stop_at_sc else "horizon_trajectories.png")

    if args.mode == "sc_hist":
        sc_times = _read_sc_times(rollouts_path)
        if len(sc_times) > 0:
            arr = np.asarray(sc_times, dtype=float)
            q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
            print(f"Loaded {len(arr)} SC times from {rollouts_path}")
            print(f"Min/Median/Max: {arr.min():.3f} / {q50:.3f} / {arr.max():.3f}")
            print(f"P10/P90: {q10:.3f} / {q90:.3f}")
        plot_sc_time_histogram(sc_times, out_path=out_path, bins=int(args.bins))
        print(f"Saved histogram to: {out_path}")
        return

    if args.mode == "horizon_at_sc_hist":
        values = _read_horizon_at_sc(rollouts_path)
        if len(values) > 0:
            arr = np.asarray(values, dtype=float)
            q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
            print(f"Loaded {len(arr)} horizon_at_sc values from {rollouts_path}")
            print(f"P10/Median/P90: {_format_time_duration(q10)} / {_format_time_duration(q50)} / {_format_time_duration(q90)}")
        plot_horizon_at_sc_histogram(values, out_path=out_path, bins=int(args.bins))
        print(f"Saved histogram to: {out_path}")
        return

    # horizon_trajectories mode
    times, trajectories, sc_times = _read_horizon_trajectories(rollouts_path)
    print(f"Loaded {len(trajectories)} trajectories with {len(times)} time points each from {rollouts_path}")
    plot_horizon_trajectories(
        times,
        trajectories,
        out_path=out_path,
        current_horizon_minutes=float(args.current_horizon_minutes),
        alpha=float(args.alpha),
        max_trajectories=int(args.max_trajectories),
        overlay_metr=(not args.no_metr),
        stop_at_sc=bool(args.stop_at_sc),
        sc_times=sc_times,
    )
    print(f"Saved horizon trajectories to: {out_path}")


if __name__ == "__main__":
    main()

