#!/usr/bin/env python3
"""
Plot utilities for analyzing batch rollout results.

Currently supported:
- Plot distribution (histogram) of SC times from a rollouts.jsonl file
- Plot distribution (histogram) of horizon length at SC across rollouts
- Plot distribution (histogram) of arrival time for arbitrary milestones

Usage examples:
  python scripts/plot_rollouts.py --run-dir outputs/20250813_020347 \
    --out outputs/20250813_020347/sc_time_hist.png

  python scripts/plot_rollouts.py --rollouts outputs/20250813_020347/rollouts.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np

# Use a non-interactive backend in case this runs on a headless server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from datetime import datetime, timedelta
import yaml

# Use monospace font for all text elements
matplotlib.rcParams["font.family"] = "monospace"


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


def _read_sc_times(rollouts_file: Path) -> Tuple[List[float], int]:
    sc_times: List[float] = []
    num_no_sc: int = 0
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
            # Count rollouts with missing/non-finite sc_time as "No SC"
            try:
                x = float(sc_time) if sc_time is not None else np.nan
            except (TypeError, ValueError):
                x = np.nan
            if np.isfinite(x):
                sc_times.append(x)
            else:
                num_no_sc += 1
    return sc_times, num_no_sc


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


def _list_milestone_names(rollouts_file: Path) -> List[str]:
    names = set()
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
            milestones = results.get("milestones")
            if isinstance(milestones, dict):
                for k in milestones.keys():
                    names.add(str(k))
    return sorted(names)


def _read_milestone_times(rollouts_file: Path, milestone_name: str) -> Tuple[List[float], int]:
    times: List[float] = []
    num_not_achieved: int = 0
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
            milestones = results.get("milestones")
            if not isinstance(milestones, dict):
                continue
            info = milestones.get(milestone_name)
            if not isinstance(info, dict):
                continue
            t = info.get("time")
            try:
                x = float(t) if t is not None else np.nan
            except (TypeError, ValueError):
                x = np.nan
            if np.isfinite(x):
                times.append(x)
            else:
                num_not_achieved += 1
    return times, num_not_achieved


def _parse_milestone_pairs(pairs_arg: Optional[str]) -> List[Tuple[str, str]]:
    """Parse a pairs string like "SC:SAR,SAR:SIAR" into [("SC","SAR"), ...]."""
    if not pairs_arg:
        return []
    pairs: List[Tuple[str, str]] = []
    for chunk in pairs_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            continue
        left, right = chunk.split(":", 1)
        left = left.strip()
        right = right.strip()
        if left and right:
            pairs.append((left, right))
    return pairs


def _read_milestone_transition_durations(
    rollouts_file: Path,
    pairs: List[Tuple[str, str]],
    filter_milestone: Optional[str] = None,
    filter_by_year: Optional[float] = None,
) -> Tuple[List[str], List[List[float]], List[int], int]:
    """For each pair (A,B), compute finite durations B.time - A.time.

    Returns:
        labels: ["A to B", ...]
        durations_per_pair: list of finite durations arrays (years)
        num_not_achieved_per_pair: count of rollouts where duration is undefined or infinite
        total_rollouts: number of rollout records processed
    """
    labels: List[str] = [f"{a} to {b}" for a, b in pairs]
    durations_per_pair: List[List[float]] = [[] for _ in pairs]
    num_not_achieved: List[int] = [0 for _ in pairs]
    total = 0

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
            milestones = results.get("milestones")
            if not isinstance(milestones, dict):
                continue
            # Filter by milestone-by-year if requested
            if filter_milestone is not None and filter_by_year is not None:
                m = milestones.get(filter_milestone)
                pass_filter = False
                if isinstance(m, dict) and m.get("time") is not None:
                    try:
                        t_m = float(m["time"])  # type: ignore[index]
                        if np.isfinite(t_m) and t_m <= float(filter_by_year):
                            pass_filter = True
                    except Exception:
                        pass_filter = False
                if not pass_filter:
                    continue

            total += 1

            # Extract milestone times for this rollout once
            times_map: Dict[str, Optional[float]] = {}
            for name, info in milestones.items():
                try:
                    if isinstance(info, dict) and info.get("time") is not None:
                        t = float(info["time"])  # type: ignore[index]
                        if np.isfinite(t):
                            times_map[str(name)] = float(t)
                        else:
                            times_map[str(name)] = None
                    else:
                        times_map[str(name)] = None
                except Exception:
                    times_map[str(name)] = None

            for idx, (a, b) in enumerate(pairs):
                ta = times_map.get(a)
                tb = times_map.get(b)
                if ta is None or tb is None:
                    num_not_achieved[idx] += 1
                    continue
                dur = float(tb - ta)
                if not np.isfinite(dur) or dur <= 0.0:
                    # Treat non-positive or non-finite as not achieved/undefined for this purpose
                    num_not_achieved[idx] += 1
                    continue
                durations_per_pair[idx].append(dur)

    return labels, durations_per_pair, num_not_achieved, total


def _read_milestone_scatter_data(
    rollouts_file: Path,
    from_name: str,
    to_name: str,
    include_inf: bool = True,
    inf_years_cap: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays (x_from_time, y_duration_years) for a milestone scatter.

    If include_inf is True, rollouts missing either milestone contribute a point at
    (from_time if available else NaN, duration=inf_years_cap) only when from_time exists.
    If include_inf is False, only finite durations with both times are included.
    """
    xs: List[float] = []
    ys: List[float] = []
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
            milestones = results.get("milestones")
            if not isinstance(milestones, dict):
                continue
            t_from = None
            t_to = None
            try:
                mf = milestones.get(from_name)
                if isinstance(mf, dict) and mf.get("time") is not None:
                    t_from = float(mf["time"])  # type: ignore[index]
                    if not np.isfinite(t_from):
                        t_from = None
            except Exception:
                t_from = None
            try:
                mt = milestones.get(to_name)
                if isinstance(mt, dict) and mt.get("time") is not None:
                    t_to = float(mt["time"])  # type: ignore[index]
                    if not np.isfinite(t_to):
                        t_to = None
            except Exception:
                t_to = None

            if t_from is None:
                # Without an x value, we cannot plot this row
                continue
            if t_to is None:
                if include_inf:
                    xs.append(float(t_from))
                    ys.append(float(inf_years_cap))
                continue
            dur = float(t_to - t_from)
            if np.isfinite(dur) and dur > 0.0:
                xs.append(float(t_from))
                ys.append(float(dur))
            else:
                if include_inf:
                    xs.append(float(t_from))
                    ys.append(float(inf_years_cap))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def plot_milestone_scatter(
    xs: np.ndarray,
    ys: np.ndarray,
    out_path: Path,
    title: Optional[str] = None,
    kind: str = "hex",
    gridsize: int = 50,
    point_size: float = 8.0,
    scatter_overlay: bool = True,
    ymin_years: Optional[float] = None,
    ymax_years: Optional[float] = None,
    condition_text: Optional[str] = None,
) -> None:
    if xs.size == 0 or ys.size == 0:
        raise ValueError("No data to plot for milestone scatter")

    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    if kind == "hex":
        hb = ax.hexbin(xs, ys, gridsize=int(gridsize), xscale='linear', yscale='log', cmap='viridis', mincnt=1)
        cb = plt.colorbar(hb)
        cb.set_label('Count')
    elif kind == "hist2d":
        h, xedges, yedges, img = ax.hist2d(xs, ys, bins=int(gridsize), cmap='viridis', norm=None)
        ax.set_yscale('log')
        cb = plt.colorbar(img)
        cb.set_label('Count')
    else:
        ax.set_yscale('log')
        ax.scatter(xs, ys, s=float(point_size), alpha=0.6, color='tab:blue')

    if kind in ("hex", "hist2d") and scatter_overlay:
        ax.scatter(xs, ys, s=float(point_size), alpha=0.3, color='black')

    # Y-axis bounds and ticks similar to other plots
    ymin = float(ymin_years) if ymin_years is not None else max(ys[ys > 0].min(), 1e-3)
    ymax = float(ymax_years) if ymax_years is not None else max(ys.max(), ymin * 10)
    ax.set_ylim(ymin, ymax)
    ticks, labels = _get_year_tick_values_and_labels(ymin, ymax)
    ax.set_yscale('log')
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

    ax.set_xlabel("From milestone date (decimal year)")
    ax.set_ylabel("Duration to target milestone (years, log scale)")
    ax.grid(True, which='both', axis='y', alpha=0.25)
    ax.set_title(title or "Milestone date vs transition duration")

    if condition_text:
        ax_inset = plt.gcf().add_axes([0.72, 0.12, 0.26, 0.2])
        ax_inset.axis('off')
        ax_inset.text(0.0, 1.0, f"Condition: {condition_text}", va='top', ha='left', fontsize=12, family='monospace', bbox=dict(facecolor=(1,1,1,0.7), edgecolor='0.7'))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _format_years_value(y: float) -> str:
    if not np.isfinite(y):
        return "inf"
    if y < 0.01:
        return f"{y:.3f}"
    if y < 0.1:
        return f"{y:.2f}"
    if y < 10:
        return f"{y:.2f}".rstrip("0").rstrip(".")
    if y < 1000:
        return f"{y:.2f}".rstrip("0").rstrip(".")
    return f"{y:,.0f}"


def _get_year_tick_values_and_labels(ymin: float, ymax: float) -> Tuple[List[float], List[str]]:
    if ymin <= 0:
        ymin = 1e-3
    lo = int(np.floor(np.log10(ymin)))
    hi = int(np.ceil(np.log10(ymax)))
    values: List[float] = []
    labels: List[str] = []
    for p in range(lo, hi + 1):
        v = float(10 ** p)
        values.append(v)
        if v >= 1000:
            labels.append(f"{int(v):,}")
        elif v >= 1:
            labels.append(str(int(v)))
        else:
            labels.append(f"{v:g}")
    return values, labels


def plot_milestone_transition_boxplot(
    labels: List[str],
    durations_per_pair: List[List[float]],
    num_not_achieved_per_pair: List[int],
    out_path: Path,
    title: Optional[str] = None,
    ymin_years: Optional[float] = None,
    ymax_years: Optional[float] = None,
    exclude_inf_from_stats: bool = False,
    inf_years_cap: float = 100.0,
    condition_text: Optional[str] = None,
) -> None:
    if len(labels) == 0:
        raise ValueError("No milestone pairs provided")
    if len(labels) != len(durations_per_pair) or len(labels) != len(num_not_achieved_per_pair):
        raise ValueError("Mismatched inputs for boxplot")

    # Prepare data groups. Finite groups are the true finite durations.
    # plot_groups are what will be fed to the boxplot: include capped
    # 'Not achieved' points unless excluded.
    finite_groups: List[np.ndarray] = []
    plot_groups: List[np.ndarray] = []
    global_min = np.inf
    global_max = 0.0
    for arr, ninf in zip(durations_per_pair, num_not_achieved_per_pair):
        a = np.asarray(arr, dtype=float)
        if a.size:
            a = a[np.isfinite(a) & (a > 0)]
        finite_groups.append(a)
        if exclude_inf_from_stats:
            group = a
        else:
            extra = np.full(int(ninf), float(inf_years_cap)) if int(ninf) > 0 else np.asarray([], dtype=float)
            group = np.concatenate([a, extra]) if a.size or extra.size else np.asarray([], dtype=float)
        plot_groups.append(group)
        if group.size:
            global_min = min(global_min, float(group.min()))
            global_max = max(global_max, float(group.max()))

    if not np.isfinite(global_min):
        raise ValueError("No finite durations found to plot")

    # Axis limits
    ymin = float(ymin_years) if ymin_years is not None else 10 ** np.floor(np.log10(global_min)) / 10.0
    ymax_candidate = float(ymax_years) if ymax_years is not None else 10 ** np.ceil(np.log10(max(global_max, global_min * 10)))
    # Ensure y-axis includes the cap for plotted 'Not achieved' points
    ymax_candidate = max(ymax_candidate, float(inf_years_cap))
    ymax = max(ymax_candidate, ymin * 10.0)

    plt.figure(figsize=(16, 8))
    ax = plt.gca()

    # Boxplot
    bp = plt.boxplot(
        plot_groups,
        labels=labels,
        showfliers=True,
        patch_artist=True,
        widths=0.6,
        whis=1.5,
    )

    # Styling
    for patch in bp['boxes']:
        patch.set(facecolor=(0.5, 0.8, 0.5, 0.7), edgecolor='black')
    for element in ['whiskers', 'caps', 'medians', 'fliers']:
        for line in bp[element]:
            line.set(color='black', linewidth=1.2)

    plt.yscale('log')
    ticks, tick_labels = _get_year_tick_values_and_labels(ymin, ymax)
    plt.yticks(ticks, tick_labels)
    plt.ylim(ymin, ymax)
    plt.xlabel("Milestone Transition")
    plt.ylabel("Calendar Years (log scale)")
    plt.grid(True, which="both", axis="y", alpha=0.25)

    plt.title(title or "Time Spent in Each Milestone Transition (calendar years)")

    # If excluded from stats/plot, we still show points at the cap as a visual cue
    if exclude_inf_from_stats:
        x_positions = np.arange(1, len(labels) + 1, dtype=float)
        y_points: List[float] = []
        x_points: List[float] = []
        sizes: List[float] = []
        for xi, ninf in zip(x_positions, num_not_achieved_per_pair):
            if int(ninf) <= 0:
                continue
            count = int(ninf)
            y_points.extend([float(inf_years_cap)] * count)
            x_points.extend([float(xi)] * count)
            sizes.extend([12.0] * count)
        if x_points:
            plt.scatter(x_points, y_points, s=sizes, color='black', alpha=0.5, zorder=3, label='Not achieved (capped)')
            plt.legend(loc='upper left')

    # Stats panel
    x_text = 1.03
    y_text = 0.98
    panel_lines: List[str] = ["Statistics (years):", ""]
    if condition_text:
        panel_lines.append(f"Condition: {condition_text}")
        panel_lines.append("")
    for lbl, arr, ninf, grp in zip(labels, finite_groups, num_not_achieved_per_pair, plot_groups):
        panel_lines.append(lbl)
        # Use the same data that feeds the boxplot for consistency
        with_inf = arr if exclude_inf_from_stats else grp
        if with_inf.size == 0:
            panel_lines.append("  10th: n/a")
            panel_lines.append("  50th: n/a")
            panel_lines.append("  90th: n/a")
            panel_lines.append("")
            continue
        q10, q50, q90 = np.quantile(with_inf, [0.1, 0.5, 0.9])
        panel_lines.append(f"  10th: {_format_years_value(float(q10))}")
        panel_lines.append(f"  50th: {_format_years_value(float(q50))}")
        panel_lines.append(f"  90th: {_format_years_value(float(q90))}")
        panel_lines.append("")

    txt = "\n".join(panel_lines)
    ax_inset = plt.gcf().add_axes([0.72, 0.12, 0.26, 0.76])
    ax_inset.axis('off')
    ax_inset.text(0.0, 1.0, txt, va='top', ha='left', fontsize=12, family='monospace', bbox=dict(facecolor=(1,1,1,0.7), edgecolor='0.7'))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_milestone_time_histogram(times: List[float], num_not_achieved: int, out_path: Path, bins: int = 50, title: Optional[str] = None, milestone_label: Optional[str] = None) -> None:
    total_n = int(len(times) + max(0, num_not_achieved))
    if total_n == 0:
        raise ValueError("No rollouts found to plot")
    data = np.asarray(times, dtype=float) if len(times) > 0 else np.asarray([], dtype=float)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    kde_counts = np.asarray([], dtype=float)
    bin_edges = np.asarray([0.0, 1.0], dtype=float)
    counts = np.asarray([], dtype=float)
    bin_width = 1.0
    no_x = None

    if len(data) > 0:
        counts, bin_edges, _ = plt.hist(
            data,
            bins=bins,
            edgecolor="black",
            alpha=0.6,
            label="Histogram",
        )
        bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
        if len(data) >= 2:
            xs = np.linspace(data.min(), data.max(), 512)
            try:
                kde = gaussian_kde(data)
                kde_counts = kde(xs) * max(len(data), 1) * bin_width
                plt.plot(xs, kde_counts, color="tab:orange", linewidth=2.25, label="Gaussian KDE")
            except Exception:
                kde_counts = np.asarray([], dtype=float)

        # Place the Not Achieved bar just to the right of the last numeric bin
        no_x = float(bin_edges[-1] + bin_width / 2.0)
    else:
        # Only Not Achieved data: choose an arbitrary position for the single bar
        no_x = 1.0

    # Draw the Not Achieved bar if needed
    if num_not_achieved > 0:
        ax.bar(no_x, num_not_achieved, width=bin_width, edgecolor="black", alpha=0.6, label="Not achieved")
        left = float(bin_edges[0]) if len(data) > 0 else 0.0
        right = float(no_x + bin_width)
        ax.set_xlim(left, right)

    # Percentiles and annotations (including Not Achieved as +inf)
    if total_n > 0:
        with_inf = data
        if num_not_achieved > 0:
            with_inf = np.concatenate([data, np.full(int(num_not_achieved), np.inf)])
        q10, q50, q90 = np.quantile(with_inf, [0.1, 0.5, 0.9])
        ymax_hist = float(np.max(counts) if counts.size else 0.0)
        ymax_kde = float(np.max(kde_counts) if kde_counts.size else 0.0)
        ymax = max(ymax_hist, ymax_kde, float(num_not_achieved), 1.0)
        y_annot = ymax * 0.95

        def _pos_and_label(qv: float) -> Tuple[float, str]:
            if np.isfinite(qv):
                return float(qv), _decimal_year_to_date_string(float(qv))
            return float(no_x), "Not achieved"

        x10, lbl10 = _pos_and_label(q10)
        x50, lbl50 = _pos_and_label(q50)
        x90, lbl90 = _pos_and_label(q90)

        plt.axvline(x10, color="tab:gray", linestyle="--", linewidth=1.5, label="P10")
        plt.axvline(x50, color="tab:green", linestyle="-", linewidth=1.75, label="Median")
        plt.axvline(x90, color="tab:gray", linestyle="--", linewidth=1.5, label="P90")

        plt.text(x10, y_annot, f"P10: {lbl10}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))
        plt.text(x50, y_annot, f"Median: {lbl50}", rotation=90, va="top", ha="right", color="tab:green", fontsize=9, backgroundcolor=(1,1,1,0.6))
        plt.text(x90, y_annot, f"P90: {lbl90}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))

    plt.xlabel("Arrival Time (decimal year)")
    plt.ylabel("Count")
    ttl = title or (f"Distribution of Arrival Time: {milestone_label}" if milestone_label else "Distribution of Milestone Arrival Time")
    plt.title(ttl)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


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


def plot_sc_time_histogram(sc_times: List[float], num_no_sc: int, out_path: Path, bins: int = 50, title: Optional[str] = None) -> None:
    total_n = int(len(sc_times) + max(0, num_no_sc))
    if total_n == 0:
        raise ValueError("No rollouts found to plot")
    data = np.asarray(sc_times, dtype=float) if len(sc_times) > 0 else np.asarray([], dtype=float)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    kde_counts = np.asarray([], dtype=float)
    bin_edges = np.asarray([0.0, 1.0], dtype=float)
    counts = np.asarray([], dtype=float)
    bin_width = 1.0
    nosc_x = None

    if len(data) > 0:
        counts, bin_edges, _ = plt.hist(
            data,
            bins=bins,
            edgecolor="black",
            alpha=0.6,
            label="Histogram",
        )
        bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
        # KDE overlay on finite data only
        if len(data) >= 2:
            xs = np.linspace(data.min(), data.max(), 512)
            try:
                kde = gaussian_kde(data)
                kde_counts = kde(xs) * max(len(data), 1) * bin_width
                plt.plot(xs, kde_counts, color="tab:orange", linewidth=2.25, label="Gaussian KDE")
            except Exception:
                kde_counts = np.asarray([], dtype=float)

        # Place the No SC bar just to the right of the last numeric bin
        nosc_x = float(bin_edges[-1] + bin_width / 2.0)
    else:
        # Only No SC data: choose an arbitrary position for the single bar
        nosc_x = 1.0

    # Draw the No SC bar if needed
    if num_no_sc > 0:
        ax.bar(nosc_x, num_no_sc, width=bin_width, edgecolor="black", alpha=0.6, label="No SC")
        # Ensure x-limits include the No SC bar, but keep default tick locator/formatter
        left = float(bin_edges[0]) if len(data) > 0 else 0.0
        right = float(nosc_x + bin_width)
        ax.set_xlim(left, right)

    # Percentiles and annotations (including No SC as +inf)
    if total_n > 0:
        with_inf = data
        if num_no_sc > 0:
            with_inf = np.concatenate([data, np.full(int(num_no_sc), np.inf)])
        q10, q50, q90 = np.quantile(with_inf, [0.1, 0.5, 0.9])
        ymax_hist = float(np.max(counts) if counts.size else 0.0)
        ymax_kde = float(np.max(kde_counts) if kde_counts.size else 0.0)
        ymax = max(ymax_hist, ymax_kde, float(num_no_sc), 1.0)
        y_annot = ymax * 0.95

        def _pos_and_label(qv: float) -> Tuple[float, str]:
            if np.isfinite(qv):
                return float(qv), _decimal_year_to_date_string(float(qv))
            return float(nosc_x), "No SC"

        x10, lbl10 = _pos_and_label(q10)
        x50, lbl50 = _pos_and_label(q50)
        x90, lbl90 = _pos_and_label(q90)

        plt.axvline(x10, color="tab:gray", linestyle="--", linewidth=1.5, label="P10")
        plt.axvline(x50, color="tab:green", linestyle="-", linewidth=1.75, label="Median")
        plt.axvline(x90, color="tab:gray", linestyle="--", linewidth=1.5, label="P90")

        plt.text(x10, y_annot, f"P10: {lbl10}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))
        plt.text(x50, y_annot, f"Median: {lbl50}", rotation=90, va="top", ha="right", color="tab:green", fontsize=9, backgroundcolor=(1,1,1,0.6))
        plt.text(x90, y_annot, f"P90: {lbl90}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))

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
    parser.add_argument("--mode", type=str, choices=["sc_hist", "horizon_trajectories", "horizon_at_sc_hist", "milestone_time_hist", "milestone_transition_box", "milestone_scatter"], default="sc_hist", help="Which plot to generate")
    # Histogram options
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins for sc_hist mode")
    # Milestone options
    parser.add_argument("--milestone", type=str, default=None, help="Milestone name for milestone_time_hist mode (e.g., 'ACD-AI')")
    parser.add_argument("--list-milestones", action="store_true", help="List milestone names found in the rollouts file and exit (works for milestone_time_hist and milestone_transition_box)")
    # Milestone transition boxplot options
    parser.add_argument("--pairs", type=str, default=None, help="Comma-separated milestone pairs for transition durations, formatted as FROM:TO (e.g., 'SC:SAR,SAR:SIAR,SIAR:ASI')")
    parser.add_argument("--ymin-years", type=float, default=None, help="Minimum y-axis (years, log scale) for transition boxplot")
    parser.add_argument("--ymax-years", type=float, default=None, help="Maximum y-axis (years, log scale) for transition boxplot")
    parser.add_argument("--exclude-inf-from-stats", action="store_true", help="Exclude 'not achieved' (treated as +inf) from the stats panel and the plot")
    parser.add_argument("--inf-years", type=float, default=100.0, help="Where to plot 'not achieved' as points on the y-axis (years)")
    parser.add_argument("--filter-milestone", type=str, default=None, help="Only include rollouts where this milestone was achieved by --filter-by-year (e.g., '5x-AIR')")
    parser.add_argument("--filter-by-year", type=float, default=None, help="Decimal year cutoff for --filter-milestone (e.g., 2029.5)")
    # Milestone scatter/heatmap options
    parser.add_argument("--scatter-pair", type=str, default=None, help="Pair for scatter/heatmap FROM:TO (e.g., '5x-AIR:2000x-AIR')")
    parser.add_argument("--scatter-kind", type=str, choices=["hex", "hist2d", "scatter"], default="hex", help="Density visualization type")
    parser.add_argument("--gridsize", type=int, default=50, help="Grid size for hex/hist2d density")
    parser.add_argument("--point-size", type=float, default=8.0, help="Point size for scatter overlay")
    parser.add_argument("--no-scatter-overlay", action="store_true", help="Disable scatter overlay when using hex/hist2d")
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
        elif args.mode == "milestone_time_hist":
            name = args.milestone or "milestone"
            safe = "".join([c if c.isalnum() or c in ("_", "-") else "_" for c in name])
            out_path = default_dir / f"milestone_{safe}_hist.png"
        elif args.mode == "milestone_scatter":
            out_path = default_dir / "milestone_scatter.png"
        else:
            if args.mode == "milestone_transition_box":
                out_path = default_dir / "milestone_transition_box.png"
            else:
                out_path = default_dir / ("horizon_trajectories_stop_at_sc.png" if args.stop_at_sc else "horizon_trajectories.png")

    if args.mode == "sc_hist":
        sc_times, num_no_sc = _read_sc_times(rollouts_path)
        if (len(sc_times) + num_no_sc) > 0 and len(sc_times) > 0:
            arr = np.asarray(sc_times, dtype=float)
            print(f"Loaded {len(arr)} finite SC times (+{num_no_sc} No SC) from {rollouts_path}")
            print(f"Finite Min/Median/Max: {arr.min():.3f} / {np.median(arr):.3f} / {arr.max():.3f}")
            with_inf = np.concatenate([arr, np.full(int(num_no_sc), np.inf)]) if num_no_sc > 0 else arr
            q10, q50, q90 = np.quantile(with_inf, [0.1, 0.5, 0.9])
            def _fmt_q(qv: float) -> str:
                return f"{qv:.3f}" if np.isfinite(qv) else "No SC"
            print(f"P10/Median/P90 (incl. No SC): {_fmt_q(q10)} / {_fmt_q(q50)} / {_fmt_q(q90)}")
        elif (len(sc_times) + num_no_sc) > 0:
            print(f"Loaded 0 finite SC times (+{num_no_sc} No SC) from {rollouts_path}")
        plot_sc_time_histogram(sc_times, num_no_sc=num_no_sc, out_path=out_path, bins=int(args.bins))
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

    if args.mode == "milestone_time_hist":
        if args.list_milestones:
            names = _list_milestone_names(rollouts_path)
            if names:
                print("Milestones found:")
                for n in names:
                    print(f" - {n}")
            else:
                print("No milestones found in file.")
            return
        if not args.milestone:
            raise ValueError("--milestone is required for milestone_time_hist mode (use --list-milestones to inspect names)")
        times, num_na = _read_milestone_times(rollouts_path, args.milestone)
        if (len(times) + num_na) > 0 and len(times) > 0:
            arr = np.asarray(times, dtype=float)
            print(f"Loaded {len(arr)} finite '{args.milestone}' arrival times (+{num_na} Not achieved) from {rollouts_path}")
            print(f"Finite Min/Median/Max: {arr.min():.3f} / {np.median(arr):.3f} / {arr.max():.3f}")
            with_inf = np.concatenate([arr, np.full(int(num_na), np.inf)]) if num_na > 0 else arr
            q10, q50, q90 = np.quantile(with_inf, [0.1, 0.5, 0.9])
            def _fmt_q(qv: float) -> str:
                return f"{qv:.3f}" if np.isfinite(qv) else "Not achieved"
            print(f"P10/Median/P90 (incl. Not achieved): {_fmt_q(q10)} / {_fmt_q(q50)} / {_fmt_q(q90)}")
        elif (len(times) + num_na) > 0:
            print(f"Loaded 0 finite '{args.milestone}' arrival times (+{num_na} Not achieved) from {rollouts_path}")
        plot_milestone_time_histogram(times, num_not_achieved=num_na, out_path=out_path, bins=int(args.bins), milestone_label=args.milestone)
        print(f"Saved histogram to: {out_path}")
        return

    if args.mode == "milestone_transition_box":
        if args.list_milestones:
            names = _list_milestone_names(rollouts_path)
            if names:
                print("Milestones found:")
                for n in names:
                    print(f" - {n}")
            else:
                print("No milestones found in file.")
            return
        pairs = _parse_milestone_pairs(args.pairs)
        if not pairs:
            raise ValueError("--pairs is required for milestone_transition_box mode (use --list-milestones to inspect names)")
        labels, durations, num_na_per_pair, total = _read_milestone_transition_durations(
            rollouts_path,
            pairs,
            filter_milestone=(args.filter_milestone if args.filter_milestone else None),
            filter_by_year=(float(args.filter_by_year) if args.filter_by_year is not None else None),
        )
        print(f"Processed {total} rollouts from {rollouts_path}")
        for lbl, arr, ninf in zip(labels, durations, num_na_per_pair):
            arr_np = np.asarray(arr, dtype=float)
            if arr_np.size:
                q10, q50, q90 = np.quantile(arr_np, [0.1, 0.5, 0.9])
                print(f"{lbl}: n={arr_np.size} finite, +{ninf} Not achieved | P10/Median/P90 (years): {q10:.3f} / {q50:.3f} / {q90:.3f}")
            else:
                print(f"{lbl}: n=0 finite, +{ninf} Not achieved")
        plot_milestone_transition_boxplot(
            labels,
            durations,
            num_na_per_pair,
            out_path=out_path,
            title=None,
            ymin_years=(float(args.ymin_years) if args.ymin_years is not None else None),
            ymax_years=(float(args.ymax_years) if args.ymax_years is not None else None),
            exclude_inf_from_stats=bool(args.exclude_inf_from_stats),
            inf_years_cap=float(args.inf_years),
            condition_text=(f"{args.filter_milestone} achieved by {args.filter_by_year}" if args.filter_milestone and args.filter_by_year is not None else None),
        )
        print(f"Saved transition boxplot to: {out_path}")
        return

    if args.mode == "milestone_scatter":
        if not args.scatter_pair:
            raise ValueError("--scatter-pair is required for milestone_scatter mode (e.g., '5x-AIR:2000x-AIR')")
        if ":" not in args.scatter_pair:
            raise ValueError("--scatter-pair must be formatted as FROM:TO")
        from_name, to_name = [s.strip() for s in args.scatter_pair.split(":", 1)]
        xs, ys = _read_milestone_scatter_data(
            rollouts_path,
            from_name,
            to_name,
            include_inf=(not args.exclude_inf_from_stats),
            inf_years_cap=float(args.inf_years),
        )
        print(f"Loaded {xs.size} points for scatter {from_name} -> {to_name} from {rollouts_path}")
        plot_milestone_scatter(
            xs,
            ys,
            out_path=out_path,
            kind=str(args.scatter_kind),
            gridsize=int(args.gridsize),
            point_size=float(args.point_size),
            scatter_overlay=(not args.no_scatter_overlay),
            ymin_years=(float(args.ymin_years) if args.ymin_years is not None else None),
            ymax_years=(float(args.ymax_years) if args.ymax_years is not None else None),
            condition_text=None,
        )
        print(f"Saved milestone scatter to: {out_path}")
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

