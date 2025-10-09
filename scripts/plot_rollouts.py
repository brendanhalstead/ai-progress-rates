#!/usr/bin/env python3
"""
Plot utilities for analyzing batch rollout results.

Currently supported:
- Plot distribution (histogram) of ACD-AI times from a rollouts.jsonl file
- Plot distribution (histogram) of horizon length at ACD-AI across rollouts
- Plot distribution (histogram) of arrival time for arbitrary milestones

Usage examples:
  python scripts/plot_rollouts.py --run-dir outputs/20250813_020347 \
    --out outputs/20250813_020347/aa_time_hist.png

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


def _read_aa_times(rollouts_file: Path) -> Tuple[List[float], int, Optional[float]]:
    """Read SC arrival times from rollouts.

    Returns:
        aa_times: list of SC arrival times (only for rollouts that achieved SC)
        num_no_sc: count of rollouts where SC was not achieved
        typical_sim_end: typical simulation end time (for display purposes), or None
    """
    aa_times: List[float] = []
    num_no_sc: int = 0
    sim_end_times: List[float] = []

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

            # Track simulation end time
            times_array = results.get("times")
            if times_array is not None and len(times_array) > 0:
                try:
                    sim_end_times.append(float(times_array[-1]))
                except Exception:
                    pass

            aa_time = results.get("aa_time")
            try:
                x = float(aa_time) if aa_time is not None else np.nan
            except (TypeError, ValueError):
                x = np.nan
            if np.isfinite(x):
                aa_times.append(x)
            else:
                num_no_sc += 1

    # Use median simulation end time for display
    typical_sim_end = float(np.median(sim_end_times)) if sim_end_times else None
    return aa_times, num_no_sc, typical_sim_end


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
    aa_times: List[Optional[float]] = []
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
            aa_time_val = results.get("aa_time")
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
                aa_times.append(float(aa_time_val) if aa_time_val is not None and np.isfinite(float(aa_time_val)) else None)
            except Exception:
                aa_times.append(None)
    if common_times is None or len(trajectories) == 0:
        raise ValueError("No horizon trajectories found in rollouts file")
    return common_times, trajectories, aa_times


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
    aa_times: Optional[List[Optional[float]]] = None,
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
        # If requested, mask values after this rollout's aa_time
        if stop_at_sc and aa_times is not None and idx < len(aa_times) and aa_times[idx] is not None:
            sc = aa_times[idx]
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
    """Compute horizon length at ACD-AI time for each rollout where available."""
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
            aa_time = results.get("aa_time")
            if times is None or horizon is None or aa_time is None:
                continue
            try:
                times_arr = np.asarray(times, dtype=float)
                horizon_arr = np.asarray(horizon, dtype=float)
                sc_t = float(aa_time)
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
            # Interpolate horizon at ACD-AI time
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


def _read_milestone_times(rollouts_file: Path, milestone_name: str) -> Tuple[List[float], int, Optional[float]]:
    """Read milestone arrival times from rollouts.

    Returns:
        times: list of arrival times (only for rollouts that achieved the milestone)
        num_not_achieved: count of rollouts where milestone was not achieved
        typical_sim_end: typical simulation end time (for display purposes), or None
    """
    times: List[float] = []
    num_not_achieved: int = 0
    sim_end_times: List[float] = []

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

            # Track simulation end time
            times_array = results.get("times")
            if times_array is not None and len(times_array) > 0:
                try:
                    sim_end_times.append(float(times_array[-1]))
                except Exception:
                    pass

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

    # Use median simulation end time for display
    typical_sim_end = float(np.median(sim_end_times)) if sim_end_times else None
    return times, num_not_achieved, typical_sim_end


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
) -> Tuple[List[str], List[List[float]], List[List[float]], List[int], List[int], List[int], Optional[float]]:
    """For each pair (A,B), compute finite durations B.time - A.time.

    Returns:
        labels: ["A to B", ...]
        durations_per_pair: list of finite durations arrays (years, only where both achieved in order)
        durations_with_censored_per_pair: list of durations including censored (B not achieved uses sim_end)
        num_b_not_achieved_per_pair: count where A achieved but B not achieved
        num_b_before_a_per_pair: count where both achieved but B came before A
        total_a_achieved_per_pair: number of rollouts where A was achieved for each pair
        typical_max_duration: typical maximum possible duration (sim_end - earliest A time)
    """
    labels: List[str] = [f"{a} to {b}" for a, b in pairs]
    durations_per_pair: List[List[float]] = [[] for _ in pairs]
    durations_with_censored_per_pair: List[List[float]] = [[] for _ in pairs]
    num_b_not_achieved: List[int] = [0 for _ in pairs]
    num_b_before_a: List[int] = [0 for _ in pairs]
    total_a_achieved: List[int] = [0 for _ in pairs]
    max_durations: List[float] = []

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

            # Get simulation end time
            times_array = results.get("times")
            if times_array is not None and len(times_array) > 0:
                try:
                    simulation_end = float(times_array[-1])
                except Exception:
                    simulation_end = None
            else:
                simulation_end = None

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

                if ta is None:
                    # If first milestone not achieved, skip entirely (don't count)
                    continue

                # Count this rollout for this pair (A was achieved)
                total_a_achieved[idx] += 1

                # Track max possible duration for this transition
                if simulation_end is not None and simulation_end > ta:
                    max_durations.append(float(simulation_end - ta))

                if tb is None:
                    # Second milestone not achieved
                    num_b_not_achieved[idx] += 1
                    # For censored data, use simulation end time
                    if simulation_end is not None and simulation_end > ta:
                        censored_dur = float(simulation_end - ta)
                        durations_with_censored_per_pair[idx].append(censored_dur)
                    continue

                dur = float(tb - ta)
                if not np.isfinite(dur):
                    # Treat non-finite as B not achieved
                    num_b_not_achieved[idx] += 1
                    # For censored data, use simulation end time
                    if simulation_end is not None and simulation_end > ta:
                        censored_dur = float(simulation_end - ta)
                        durations_with_censored_per_pair[idx].append(censored_dur)
                    continue

                if dur <= 0.0:
                    # B came before or at same time as A
                    num_b_before_a[idx] += 1
                    continue

                durations_per_pair[idx].append(dur)
                durations_with_censored_per_pair[idx].append(dur)

    typical_max_duration = float(np.median(max_durations)) if max_durations else None
    return labels, durations_per_pair, durations_with_censored_per_pair, num_b_not_achieved, num_b_before_a, total_a_achieved, typical_max_duration


def _compute_x_years_in_1_year(
    times: np.ndarray,
    progress: np.ndarray,
    current_time_idx: int = 0,
) -> float:
    """Compute maximum 'X years in 1 year' metric for a trajectory.

    Finds the 1-year window with the maximum OOMs crossed and compares it to
    the OOMs crossed in the present year (starting at current_time_idx).

    The metric answers: "If the max year had X times more progress than the
    present year, what is X?" Result should always be >= 1.

    Args:
        times: array of time points (decimal years)
        progress: array of progress values (in OOMs or log10 scale)
        current_time_idx: index representing the "present" time

    Returns:
        X value such that "X years in 1 year" happened (ratio of max to present)
    """
    if len(times) < 2 or len(progress) < 2:
        return np.nan

    # Find OOMs crossed in each rolling 1-year window (looking forward from each point)
    max_ooms_per_year = 0.0

    for i in range(len(times)):
        # Find the point approximately 1 year after times[i]
        target_time = times[i] + 1.0

        # Find the index closest to target_time
        end_idx = None
        for j in range(i + 1, len(times)):
            if times[j] >= target_time:
                end_idx = j
                break

        if end_idx is None:
            # Use the last point if we don't reach 1 year ahead
            if i < len(times) - 1:
                end_idx = len(times) - 1
            else:
                continue

        # Calculate OOMs crossed in this window
        ooms_in_window = progress[end_idx] - progress[i]
        max_ooms_per_year = max(max_ooms_per_year, float(ooms_in_window))

    # Get current year OOMs (1 year window starting from current_time_idx)
    if current_time_idx >= len(times) - 1:
        return np.nan

    target_time = times[current_time_idx] + 1.0
    current_end_idx = None
    for j in range(current_time_idx + 1, len(times)):
        if times[j] >= target_time:
            current_end_idx = j
            break

    if current_end_idx is None:
        current_end_idx = len(times) - 1

    current_ooms = progress[current_end_idx] - progress[current_time_idx]

    if current_ooms <= 0:
        return np.nan

    # This ratio should always be >= 1 since max includes current as a candidate
    return float(max_ooms_per_year / current_ooms)


def _read_x_years_in_1_year(rollouts_file: Path) -> List[float]:
    """Read progress trajectories and compute 'X years in 1 year' metric for each rollout.

    Only includes rollouts where ACD-AI (aa_time) is achieved.

    Returns:
        List of X values (one per rollout that has valid progress data and achieved ACD-AI)
    """
    x_values: List[float] = []

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

            # Check if ACD-AI was achieved
            aa_time = results.get("aa_time")
            try:
                aa_t = float(aa_time) if aa_time is not None else np.nan
            except (TypeError, ValueError):
                aa_t = np.nan
            if not np.isfinite(aa_t):
                # Skip rollouts where ACD-AI was not achieved
                continue

            times = results.get("times")
            progress = results.get("progress")

            if times is None or progress is None:
                continue

            try:
                times_arr = np.asarray(times, dtype=float)
                progress_arr = np.asarray(progress, dtype=float)
            except Exception:
                continue

            if times_arr.ndim != 1 or progress_arr.ndim != 1 or times_arr.size != progress_arr.size:
                continue

            if times_arr.size < 2:
                continue

            # Assume index 0 is the "present" time
            x_val = _compute_x_years_in_1_year(times_arr, progress_arr, current_time_idx=0)

            if np.isfinite(x_val) and x_val > 0:
                x_values.append(x_val)

    return x_values


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
    durations_with_censored_per_pair: List[List[float]],
    num_b_not_achieved_per_pair: List[int],
    num_b_before_a_per_pair: List[int],
    total_per_pair: List[int],
    out_path: Path,
    title: Optional[str] = None,
    ymin_years: Optional[float] = None,
    ymax_years: Optional[float] = None,
    exclude_inf_from_stats: bool = False,
    inf_years_cap: Optional[float] = None,
    inf_years_display: float = 100.0,
    condition_text: Optional[str] = None,
) -> None:
    if len(labels) == 0:
        raise ValueError("No milestone pairs provided")
    if len(labels) != len(durations_per_pair) or len(labels) != len(num_b_not_achieved_per_pair) or len(labels) != len(num_b_before_a_per_pair):
        raise ValueError("Mismatched inputs for boxplot")

    # Prepare data groups. Finite groups are the true finite durations.
    # Censored groups include both achieved and censored (not achieved -> sim_end)
    finite_groups: List[np.ndarray] = []
    censored_groups: List[np.ndarray] = []
    global_min = np.inf
    global_max = 0.0

    for arr, arr_censored in zip(durations_per_pair, durations_with_censored_per_pair):
        # Finite only
        a = np.asarray(arr, dtype=float)
        if a.size:
            a = a[np.isfinite(a) & (a > 0)]
        finite_groups.append(a)
        if a.size:
            global_min = min(global_min, float(a.min()))
            global_max = max(global_max, float(a.max()))

        # Including censored
        ac = np.asarray(arr_censored, dtype=float)
        if ac.size:
            ac = ac[np.isfinite(ac) & (ac > 0)]
        censored_groups.append(ac)
        if ac.size:
            global_min = min(global_min, float(ac.min()))
            global_max = max(global_max, float(ac.max()))

    if not np.isfinite(global_min):
        raise ValueError("No finite durations found to plot")

    # Axis limits
    ymin = float(ymin_years) if ymin_years is not None else 10 ** np.floor(np.log10(global_min)) / 10.0
    ymax_candidate = float(ymax_years) if ymax_years is not None else 10 ** np.ceil(np.log10(max(global_max, global_min * 10)))
    # Ensure y-axis includes the display point for 'Not achieved' points
    ymax_candidate = max(ymax_candidate, float(inf_years_display))
    ymax = max(ymax_candidate, ymin * 10.0)

    plt.figure(figsize=(18, 8))
    ax = plt.gca()

    # Adjust plot area to leave room for stats panel on right
    plt.subplots_adjust(right=0.64)

    # Interleave the two sets of boxes: achieved only and including censored
    # For each pair, we'll have two boxes side by side
    all_groups = []
    all_positions = []
    all_labels = []
    colors = []

    for i, (label, fg, cg) in enumerate(zip(labels, finite_groups, censored_groups)):
        # Position boxes with spacing: pairs at 3*i and 3*i+1, with gap to next pair
        pos_achieved = 3 * i + 0.6
        pos_censored = 3 * i + 1.4

        all_groups.append(fg)
        all_positions.append(pos_achieved)
        colors.append((0.5, 0.8, 0.5, 0.7))  # Green for achieved only

        all_groups.append(cg)
        all_positions.append(pos_censored)
        colors.append((0.5, 0.5, 0.8, 0.7))  # Blue for including censored

        # Label in the middle of the two boxes
        all_labels.append(label if i == 0 else "")
        all_labels.append("")

    # Create boxplot with custom positions
    bp = plt.boxplot(
        all_groups,
        positions=all_positions,
        showfliers=True,
        patch_artist=True,
        widths=0.7,
        whis=1.5,
    )

    # Styling with different colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set(facecolor=color, edgecolor='black')
    for element in ['whiskers', 'caps', 'medians', 'fliers']:
        for line in bp[element]:
            line.set(color='black', linewidth=1.2)

    # Set custom x-tick labels at the center of each pair
    tick_positions = [3 * i + 1.0 for i in range(len(labels))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0.5, 0.8, 0.5, 0.7), edgecolor='black', label='Both achieved'),
        Patch(facecolor=(0.5, 0.5, 0.8, 0.7), edgecolor='black', label='Assuming achieved at cutoff')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.yscale('log')
    ticks, tick_labels = _get_year_tick_values_and_labels(ymin, ymax)
    plt.yticks(ticks, tick_labels)
    plt.ylim(ymin, ymax)
    plt.xlabel("Milestone Transition")
    plt.ylabel("Calendar Years (log scale)")
    plt.grid(True, which="both", axis="y", alpha=0.25)

    plt.title(title or "Time Spent in Each Milestone Transition (calendar years)")

    # Stats panel
    x_text = 1.03
    y_text = 0.98
    panel_lines: List[str] = ["Statistics (years):", ""]
    if condition_text:
        panel_lines.append(f"Condition: {condition_text}")
        panel_lines.append("")

    for lbl, arr, arr_c, n_not_achieved, n_before, total_a in zip(labels, finite_groups, censored_groups, num_b_not_achieved_per_pair, num_b_before_a_per_pair, total_per_pair):
        panel_lines.append(lbl)

        # Stats for achieved only
        panel_lines.append("  Both achieved:")
        if arr.size == 0:
            panel_lines.append("    (none)")
        else:
            q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
            panel_lines.append(f"    10/50/90: {_format_years_value(float(q10))}/{_format_years_value(float(q50))}/{_format_years_value(float(q90))}")

        # Show not achieved count before "Assuming..." section
        if n_not_achieved > 0:
            milestone_b = lbl.split(" to ")[1] if " to " in lbl else "second"
            pct_not_achieved = 100.0 * n_not_achieved / total_a if total_a > 0 else 0.0
            panel_lines.append(f"  ({pct_not_achieved:.1f}% {milestone_b} not achieved)")

        # Stats for including censored
        milestone_b = lbl.split(" to ")[1] if " to " in lbl else "second"
        panel_lines.append(f"  Assuming {milestone_b} achieved at simulation cutoff:")
        if arr_c.size == 0:
            panel_lines.append("    (none)")
        else:
            q10_c, q50_c, q90_c = np.quantile(arr_c, [0.1, 0.5, 0.9])
            panel_lines.append(f"    10/50/90: {_format_years_value(float(q10_c))}/{_format_years_value(float(q50_c))}/{_format_years_value(float(q90_c))}")

        # Show out of order count at the end
        if n_before > 0:
            milestone_b = lbl.split(" to ")[1] if " to " in lbl else "second"
            milestone_a = lbl.split(" to ")[0] if " to " in lbl else "first"
            pct_before = 100.0 * n_before / total_a if total_a > 0 else 0.0
            panel_lines.append(f"  ({pct_before:.1f}% {milestone_b} before {milestone_a})")
        panel_lines.append("")

    txt = "\n".join(panel_lines)
    ax_inset = plt.gcf().add_axes([0.66, 0.12, 0.32, 0.76])
    ax_inset.axis('off')
    ax_inset.text(0.0, 1.0, txt, va='top', ha='left', fontsize=12, family='monospace', bbox=dict(facecolor=(1,1,1,0.7), edgecolor='0.7'))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def plot_milestone_time_histogram(times: List[float], num_not_achieved: int, out_path: Path, bins: int = 50, title: Optional[str] = None, milestone_label: Optional[str] = None, sim_end: Optional[float] = None) -> None:
    """Plot histogram of milestone arrival times.

    Args:
        times: list of arrival times (only for rollouts that achieved the milestone)
        num_not_achieved: count of rollouts where milestone was not achieved
        milestone_label: name of the milestone for labeling
        sim_end: typical simulation end time for labeling "not achieved" bar
    """
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
                kde_counts = kde(xs) * len(data) * bin_width
                plt.plot(xs, kde_counts, color="tab:orange", linewidth=2.25, label="Gaussian KDE")
            except Exception:
                kde_counts = np.asarray([], dtype=float)

        # Place the Not Achieved bar at the simulation end time if available
        if sim_end is not None:
            no_x = float(sim_end)
        else:
            no_x = float(bin_edges[-1] + bin_width / 2.0)
    else:
        # Only Not Achieved data: use sim_end if available, else arbitrary position
        if sim_end is not None:
            no_x = float(sim_end)
            bin_width = 1.0
        else:
            no_x = 1.0
            bin_width = 1.0

    # Calculate ymax for annotations
    ymax_hist = float(np.max(counts) if counts.size else 0.0)
    ymax_kde = float(np.max(kde_counts) if kde_counts.size else 0.0)
    ymax = max(ymax_hist, ymax_kde, float(num_not_achieved), 1.0)

    # Draw the Not Achieved bar if needed
    sim_end_year = int(np.round(sim_end)) if sim_end is not None else None
    if num_not_achieved > 0:
        ax.bar(no_x, num_not_achieved, width=bin_width, edgecolor="black", alpha=0.6, color="tab:red")
        left = float(bin_edges[0]) if len(data) > 0 else (no_x - bin_width)
        right = float(no_x + bin_width / 2.0)
        ax.set_xlim(left, right)
    elif len(data) > 0:
        # No "not achieved" bar, but we have data - set xlim based on bins
        left = float(bin_edges[0])
        right = float(bin_edges[-1])
        ax.set_xlim(left, right)

    # Percentiles and annotations (including not achieved as simulation end)
    if total_n > 0:
        y_annot = ymax * 0.95

        # Create combined data including "not achieved" at simulation end
        if num_not_achieved > 0 and sim_end is not None:
            combined_data = np.concatenate([data, np.full(num_not_achieved, sim_end)])
        else:
            combined_data = data

        if len(combined_data) > 0:
            q10, q50, q90 = np.quantile(combined_data, [0.1, 0.5, 0.9])

            # Determine position and label for each percentile
            def _percentile_pos_label(qv: float) -> Tuple[float, str]:
                # If percentile is at or beyond simulation end, it's "not achieved"
                if sim_end is not None and qv >= sim_end - 0.01:  # small tolerance
                    sim_end_year = int(np.round(sim_end))
                    return float(no_x), f"Not achieved by {sim_end_year}"
                return float(qv), _decimal_year_to_date_string(float(qv))

            x10, lbl10 = _percentile_pos_label(q10)
            x50, lbl50 = _percentile_pos_label(q50)
            x90, lbl90 = _percentile_pos_label(q90)

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
    plt.xlabel("Horizon at ACD-AI (minutes)")
    plt.ylabel("Count")
    plt.title(title or "Distribution of Horizon Length at ACD-AI")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_aa_time_histogram(aa_times: List[float], num_no_sc: int, out_path: Path, bins: int = 50, title: Optional[str] = None, sim_end: Optional[float] = None) -> None:
    """Plot histogram of ACD-AI arrival times.

    Args:
        aa_times: list of SC arrival times (only for rollouts that achieved SC)
        num_no_sc: count of rollouts where SC was not achieved
        sim_end: typical simulation end time for labeling "not achieved" bar
    """
    total_n = int(len(aa_times) + max(0, num_no_sc))
    if total_n == 0:
        raise ValueError("No rollouts found to plot")
    data = np.asarray(aa_times, dtype=float) if len(aa_times) > 0 else np.asarray([], dtype=float)

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
        if len(data) >= 2:
            xs = np.linspace(data.min(), data.max(), 512)
            try:
                kde = gaussian_kde(data)
                kde_counts = kde(xs) * len(data) * bin_width
                plt.plot(xs, kde_counts, color="tab:orange", linewidth=2.25, label="Gaussian KDE")
            except Exception:
                kde_counts = np.asarray([], dtype=float)

        # Place the No SC bar at the simulation end time if available
        if sim_end is not None:
            nosc_x = float(sim_end)
        else:
            nosc_x = float(bin_edges[-1] + bin_width / 2.0)
    else:
        # Only No SC data: use sim_end if available, else arbitrary position
        if sim_end is not None:
            nosc_x = float(sim_end)
            bin_width = 1.0
        else:
            nosc_x = 1.0
            bin_width = 1.0

    # Calculate ymax for annotations
    ymax_hist = float(np.max(counts) if counts.size else 0.0)
    ymax_kde = float(np.max(kde_counts) if kde_counts.size else 0.0)
    ymax = max(ymax_hist, ymax_kde, float(num_no_sc), 1.0)

    # Draw the No SC bar if needed
    sim_end_year = int(np.round(sim_end)) if sim_end is not None else None
    if num_no_sc > 0:
        ax.bar(nosc_x, num_no_sc, width=bin_width, edgecolor="black", alpha=0.6, color="tab:red")
        left = float(bin_edges[0]) if len(data) > 0 else (nosc_x - bin_width)
        right = float(nosc_x + bin_width / 2.0)
        ax.set_xlim(left, right)
    elif len(data) > 0:
        # No "no SC" bar, but we have data - set xlim based on bins
        left = float(bin_edges[0])
        right = float(bin_edges[-1])
        ax.set_xlim(left, right)

    # Percentiles and annotations (including not achieved as simulation end)
    if total_n > 0:
        y_annot = ymax * 0.95

        # Create combined data including "not achieved" at simulation end
        if num_no_sc > 0 and sim_end is not None:
            combined_data = np.concatenate([data, np.full(num_no_sc, sim_end)])
        else:
            combined_data = data

        if len(combined_data) > 0:
            q10, q50, q90 = np.quantile(combined_data, [0.1, 0.5, 0.9])

            # Determine position and label for each percentile
            def _percentile_pos_label(qv: float) -> Tuple[float, str]:
                # If percentile is at or beyond simulation end, it's "not achieved"
                if sim_end is not None and qv >= sim_end - 0.01:  # small tolerance
                    sim_end_year = int(np.round(sim_end))
                    return float(nosc_x), f"Not achieved by {sim_end_year}"
                return float(qv), _decimal_year_to_date_string(float(qv))

            x10, lbl10 = _percentile_pos_label(q10)
            x50, lbl50 = _percentile_pos_label(q50)
            x90, lbl90 = _percentile_pos_label(q90)

            plt.axvline(x10, color="tab:gray", linestyle="--", linewidth=1.5, label="P10")
            plt.axvline(x50, color="tab:green", linestyle="-", linewidth=1.75, label="Median")
            plt.axvline(x90, color="tab:gray", linestyle="--", linewidth=1.5, label="P90")

            plt.text(x10, y_annot, f"P10: {lbl10}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))
            plt.text(x50, y_annot, f"Median: {lbl50}", rotation=90, va="top", ha="right", color="tab:green", fontsize=9, backgroundcolor=(1,1,1,0.6))
            plt.text(x90, y_annot, f"P90: {lbl90}", rotation=90, va="top", ha="right", color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))

    plt.xlabel("ACD-AI Time (decimal year)")
    plt.ylabel("Count")
    plt.title(title or "Distribution of ACD-AI Times")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_x_years_in_1_year_histogram(x_values: List[float], out_path: Path, bins: int = 50, title: Optional[str] = None) -> None:
    """Plot histogram of 'X years in 1 year' values.

    Args:
        x_values: list of X values (ratio of max OOMs/year to current OOMs/year)
        out_path: output file path
        bins: number of histogram bins
        title: optional plot title
    """
    if len(x_values) == 0:
        raise ValueError("No X values found to plot")

    data = np.asarray(x_values, dtype=float)

    # Separate values ≤50 and >50
    data_in_range = data[data <= 50]
    num_above_50 = int(np.sum(data > 50))

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Create log-spaced bins from min to 50
    xmin = max(data_in_range[data_in_range > 0].min(), 1.0) if len(data_in_range) > 0 else 1.0
    xmax = 50.0
    bin_edges = np.logspace(np.log10(xmin), np.log10(xmax), int(bins))

    # Create histogram for data ≤50 with log-spaced bins
    counts, bin_edges, _ = ax.hist(
        data_in_range,
        bins=bin_edges,
        edgecolor="black",
        alpha=0.6,
        label="Histogram",
    )

    # Add a special bar for >50 values
    if num_above_50 > 0:
        # Position the >50 bar at x=70 (visually separated from the 50 mark)
        bar_x = 70.0
        bar_width = 20.0  # Make it visually distinct
        ax.bar(bar_x, num_above_50, width=bar_width, edgecolor="black",
               alpha=0.6, color="tab:red", label=f">50 ({num_above_50})")

    # Set x-axis to log scale
    plt.xscale("log")

    # Add KDE if we have enough data (only for data ≤50)
    if len(data_in_range) >= 2:
        xs = np.linspace(data_in_range.min(), data_in_range.max(), 512)
        try:
            kde = gaussian_kde(data_in_range)
            bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
            kde_counts = kde(xs) * len(data_in_range) * bin_width
            plt.plot(xs, kde_counts, color="tab:orange", linewidth=2.25, label="Gaussian KDE")
        except Exception:
            pass

    # Percentiles and annotations (using original uncapped data)
    q10, q50, q90 = np.quantile(data, [0.1, 0.5, 0.9])
    ymax = float(np.max(counts) if counts.size else 1.0)
    y_annot = ymax * 0.95

    # Only show percentile lines if they're <= 50
    if q10 <= 50:
        plt.axvline(q10, color="tab:gray", linestyle="--", linewidth=1.5, label="P10")
        plt.text(q10, y_annot, f"P10: {q10:.1f}x", rotation=90, va="top", ha="right",
                 color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))

    if q50 <= 50:
        plt.axvline(q50, color="tab:green", linestyle="-", linewidth=1.75, label="Median")
        plt.text(q50, y_annot, f"Median: {q50:.1f}x", rotation=90, va="top", ha="right",
                 color="tab:green", fontsize=9, backgroundcolor=(1,1,1,0.6))

    if q90 <= 50:
        plt.axvline(q90, color="tab:gray", linestyle="--", linewidth=1.5, label="P90")
        plt.text(q90, y_annot, f"P90: {q90:.1f}x", rotation=90, va="top", ha="right",
                 color="tab:gray", fontsize=9, backgroundcolor=(1,1,1,0.6))

    plt.xlabel("Speedup factor (ratio of annual progress between fastest and current year)")
    plt.ylabel("Count")
    plt.title(title or "Distribution of Maximum 'X Years in 1 Year'\n(Only simulations reaching ACD-AI)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_milestone_pdfs_overlay(rollouts_file: Path, milestone_names: List[str], out_path: Path, title: Optional[str] = None) -> None:
    """Plot overlaid PDFs for multiple milestones."""
    plt.figure(figsize=(12, 7))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

    global_min = np.inf
    global_max = -np.inf
    global_sim_end = None

    stats_lines = []

    for idx, milestone_name in enumerate(milestone_names):
        times, num_not_achieved, sim_end = _read_milestone_times(rollouts_file, milestone_name)

        if len(times) < 2:
            print(f"Warning: Not enough data for {milestone_name} to create KDE, skipping")
            continue

        data = np.asarray(times, dtype=float)
        global_min = min(global_min, float(data.min()))
        global_max = max(global_max, float(data.max()))

        # Track simulation end for "not achieved" marker
        if sim_end is not None:
            global_sim_end = sim_end if global_sim_end is None else max(global_sim_end, sim_end)

        # Calculate percentiles (including not achieved as sim_end)
        if num_not_achieved > 0 and sim_end is not None:
            combined_data = np.concatenate([data, np.full(num_not_achieved, sim_end)])
        else:
            combined_data = data

        q10, q50 = np.quantile(combined_data, [0.1, 0.5])

        # Calculate achievement probability
        total_runs = len(times) + num_not_achieved
        prob_achieved = len(times) / total_runs if total_runs > 0 else 0.0

        # Create KDE
        try:
            kde = gaussian_kde(data)
            xs = np.linspace(data.min(), data.max(), 512)
            # Scale PDF by probability of achievement
            pdf_values = kde(xs) * prob_achieved

            color = colors[idx % len(colors)]
            plt.plot(xs, pdf_values, linewidth=2.5, label=milestone_name, color=color)

            # Find mode (peak of KDE)
            mode_idx = np.argmax(pdf_values)
            mode = xs[mode_idx]

            # Store stats for display
            pct_achieved = prob_achieved * 100

            # For ACD-AI, AI2027-SC, SAR-level, and SIAR-level, also show stats for achieved-only runs
            if milestone_name in ["ACD-AI", "AI2027-SC", "SAR-level-experiment-selection-skill", "SIAR-level-experiment-selection-skill"] and num_not_achieved > 0:
                # Calculate percentiles using only achieved runs
                q10_achieved, q50_achieved = np.quantile(data, [0.1, 0.5])
                stats_lines.append(f"{milestone_name}: Mode={mode:.1f}, P10={q10:.1f}, P50={q50:.1f}, {pct_achieved:.0f}% achieved")
                stats_lines.append(f"  (filtering for achieved: P10={q10_achieved:.1f}, P50={q50_achieved:.1f})")
            else:
                stats_lines.append(f"{milestone_name}: Mode={mode:.1f}, P10={q10:.1f}, P50={q50:.1f}, {pct_achieved:.0f}% achieved")
        except Exception as e:
            print(f"Warning: Could not create KDE for {milestone_name}: {e}")
            continue

    plt.xlabel("Arrival Time (decimal year)")
    plt.ylabel("Probability Density")
    plt.title(title or "Milestone Arrival Time Distributions")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')

    # Add statistics text in top right
    stats_text = "\n".join(stats_lines)
    plt.text(0.98, 0.98, stats_text,
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
             family='monospace')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def batch_plot_all(rollouts_file: Path, output_dir: Path) -> None:
    """Generate all standard plots for a batch rollout.

    Creates:
    - Milestone time histograms for key milestones
    - Milestone transition boxplot for key transitions
    - Overlaid PDF plot showing arrival distributions for key milestones
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Milestone time histograms
    milestones = [
        "ACD-AI",
        "AIR-5x",
        "AI2027-SC",
        "SAR-level-experiment-selection-skill",
        "AIR-25x",
        "SIAR-level-experiment-selection-skill",
        "AIR-250x"
    ]

    for milestone in milestones:
        # Create safe filename from milestone name
        safe_name = milestone.replace("(Expensive, threshold only considers taste) ", "").replace("-", "_").replace("x", "x")
        out_path = output_dir / f"milestone_time_hist_{safe_name}.png"

        times, num_not_achieved, sim_end = _read_milestone_times(rollouts_file, milestone)

        if not times and num_not_achieved == 0:
            print(f"Warning: No data found for milestone '{milestone}', skipping histogram")
            continue

        plot_milestone_time_histogram(
            times,
            num_not_achieved,
            out_path,
            bins=50,
            title=f"Distribution of {milestone} Times",
            sim_end=sim_end
        )
        print(f"Saved {out_path}")

    # Milestone transition boxplot
    pairs_str = "ACD-AI:AI2027-SC,ACD-AI:SAR-level-experiment-selection-skill,SAR-level-experiment-selection-skill:SIAR-level-experiment-selection-skill"
    pairs = _parse_milestone_pairs(pairs_str)
    out_path = output_dir / "milestone_transition_box.png"

    labels, durations, durations_censored, num_b_not_achieved, num_b_before_a, total_per_pair, typical_max = _read_milestone_transition_durations(
        rollouts_file,
        pairs,
        filter_milestone=None,
        filter_by_year=None
    )

    if labels:
        plot_milestone_transition_boxplot(
            labels,
            durations,
            durations_censored,
            num_b_not_achieved,
            num_b_before_a,
            total_per_pair,
            out_path,
            ymin_years=None,
            ymax_years=None,
            exclude_inf_from_stats=False,
            inf_years_cap=typical_max,
            inf_years_display=100.0,
            title="Milestone Transition Durations"
        )
        print(f"Saved {out_path}")
    else:
        print("Warning: No transition data found, skipping boxplot")

    # Overlaid PDF plot for key milestones
    overlay_milestones = [
        "ACD-AI",
        "AI2027-SC",
        "SAR-level-experiment-selection-skill",
        "SIAR-level-experiment-selection-skill"
    ]
    out_path = output_dir / "milestone_pdfs_overlay.png"
    plot_milestone_pdfs_overlay(
        rollouts_file,
        overlay_milestones,
        out_path,
        title="Milestone Arrival Time Distributions"
    )
    print(f"Saved {out_path}")

    # Parameter sensitivity analysis for milestone transitions
    try:
        import sys
        sys.path.insert(0, str(rollouts_file.parent.parent / "scripts"))
        from sensitivity_analysis import analyze_milestone_transitions

        # Sensitivity analysis: ACD-AI to SAR-level
        print("Running parameter sensitivity analysis for ACD-AI to SAR-level-experiment-selection-skill (both achieved only)...")
        analyze_milestone_transitions(
            rollouts_file,
            output_dir,
            transition_pair=("ACD-AI", "SAR-level-experiment-selection-skill"),
            include_censored=False
        )

        print("Running parameter sensitivity analysis for ACD-AI to SAR-level-experiment-selection-skill (including censored)...")
        analyze_milestone_transitions(
            rollouts_file,
            output_dir,
            transition_pair=("ACD-AI", "SAR-level-experiment-selection-skill"),
            include_censored=True
        )

        # Sensitivity analysis: SAR-level to SIAR-level
        print("Running parameter sensitivity analysis for SAR-level-experiment-selection-skill to SIAR-level-experiment-selection-skill (both achieved only)...")
        analyze_milestone_transitions(
            rollouts_file,
            output_dir,
            transition_pair=("SAR-level-experiment-selection-skill", "SIAR-level-experiment-selection-skill"),
            include_censored=False
        )

        print("Running parameter sensitivity analysis for SAR-level-experiment-selection-skill to SIAR-level-experiment-selection-skill (including censored)...")
        analyze_milestone_transitions(
            rollouts_file,
            output_dir,
            transition_pair=("SAR-level-experiment-selection-skill", "SIAR-level-experiment-selection-skill"),
            include_censored=True
        )

    except Exception as e:
        print(f"Warning: Could not run parameter sensitivity analysis: {e}")

    # X years in 1 year distribution
    out_path = output_dir / "x_years_in_1_year_hist.png"
    x_values = _read_x_years_in_1_year(rollouts_file)

    if len(x_values) > 0:
        plot_x_years_in_1_year_histogram(
            x_values,
            out_path,
            bins=50
        )
        # Print statistics
        arr = np.asarray(x_values, dtype=float)
        q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
        print(f"Saved {out_path}")
        print(f"  X years in 1 year - P10/Median/P90: {q10:.1f}x / {q50:.1f}x / {q90:.1f}x")
    else:
        print(f"Warning: No valid X values found, skipping X years in 1 year plot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tools for batch rollout results")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=str, default=None, help="Path to a single rollout run directory containing rollouts.jsonl")
    g.add_argument("--rollouts", type=str, default=None, help="Path directly to a rollouts.jsonl file")
    parser.add_argument("--out", type=str, default=None, help="Output image path (PNG). Defaults vary by --mode")
    parser.add_argument("--mode", type=str, choices=["sc_hist", "horizon_trajectories", "horizon_at_sc_hist", "milestone_time_hist", "milestone_transition_box", "milestone_scatter"], default="sc_hist", help="Which plot to generate")
    parser.add_argument("--batch-all", action="store_true", help="Generate all standard plots (ignores --mode and --out)")
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
    parser.add_argument("--filter-milestone", type=str, default=None, help="Only include rollouts where this milestone was achieved by --filter-by-year (e.g., 'AIR-5x')")
    parser.add_argument("--filter-by-year", type=float, default=None, help="Decimal year cutoff for --filter-milestone (e.g., 2029.5)")
    # Milestone scatter/heatmap options
    parser.add_argument("--scatter-pair", type=str, default=None, help="Pair for scatter/heatmap FROM:TO (e.g., 'AIR-5x:AIR-2000x')")
    parser.add_argument("--scatter-kind", type=str, choices=["hex", "hist2d", "scatter"], default="hex", help="Density visualization type")
    parser.add_argument("--gridsize", type=int, default=50, help="Grid size for hex/hist2d density")
    parser.add_argument("--point-size", type=float, default=8.0, help="Point size for scatter overlay")
    parser.add_argument("--no-scatter-overlay", action="store_true", help="Disable scatter overlay when using hex/hist2d")
    # Horizon trajectories options
    parser.add_argument("--current-horizon-minutes", type=float, default=15.0, help="Horizontal reference line for current horizon in minutes")
    parser.add_argument("--alpha", type=float, default=0.08, help="Transparency for individual trajectories")
    parser.add_argument("--max-trajectories", type=int, default=2000, help="Maximum number of trajectories to draw")
    parser.add_argument("--no-metr", action="store_true", help="Disable overlay of METR p80 benchmark points")
    parser.add_argument("--stop-at-sc", action="store_true", help="Mask each trajectory after its own aa_time")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollouts_path = _resolve_rollouts_path(args.run_dir, args.rollouts)

    default_dir = rollouts_path.parent

    # Handle batch-all mode
    if args.batch_all:
        print(f"Generating all standard plots from {rollouts_path}")
        batch_plot_all(rollouts_path, default_dir)
        print("Batch plotting complete.")
        return

    if args.out is not None:
        out_path = Path(args.out)
    else:
        if args.mode == "sc_hist":
            out_path = default_dir / "aa_time_hist.png"
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
        aa_times, num_no_sc, sim_end = _read_aa_times(rollouts_path)
        if (len(aa_times) + num_no_sc) > 0 and len(aa_times) > 0:
            arr = np.asarray(aa_times, dtype=float)
            print(f"Loaded {len(arr)} finite ACD-AI times (+{num_no_sc} No ACD-AI) from {rollouts_path}")
            print(f"Finite Min/Median/Max: {arr.min():.3f} / {np.median(arr):.3f} / {arr.max():.3f}")
            with_inf = np.concatenate([arr, np.full(int(num_no_sc), np.inf)]) if num_no_sc > 0 else arr
            q10, q50, q90 = np.quantile(with_inf, [0.1, 0.5, 0.9])
            def _fmt_q(qv: float) -> str:
                return f"{qv:.3f}" if np.isfinite(qv) else "No ACD-AI"
            print(f"P10/Median/P90 (incl. No ACD-AI): {_fmt_q(q10)} / {_fmt_q(q50)} / {_fmt_q(q90)}")
        elif (len(aa_times) + num_no_sc) > 0:
            print(f"Loaded 0 finite ACD-AI times (+{num_no_sc} No ACD-AI) from {rollouts_path}")
        plot_aa_time_histogram(aa_times, num_no_sc=num_no_sc, out_path=out_path, bins=int(args.bins), sim_end=sim_end)
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
        times, num_na, sim_end = _read_milestone_times(rollouts_path, args.milestone)
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
        plot_milestone_time_histogram(times, num_not_achieved=num_na, out_path=out_path, bins=int(args.bins), milestone_label=args.milestone, sim_end=sim_end)
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
        labels, durations, durations_censored, num_b_not_achieved, num_b_before_a, total_per_pair, typical_max = _read_milestone_transition_durations(
            rollouts_path,
            pairs,
            filter_milestone=(args.filter_milestone if args.filter_milestone else None),
            filter_by_year=(float(args.filter_by_year) if args.filter_by_year is not None else None),
        )
        for lbl, arr, n_not_achieved, n_before, total_a in zip(labels, durations, num_b_not_achieved, num_b_before_a, total_per_pair):
            arr_np = np.asarray(arr, dtype=float)
            if arr_np.size:
                q10, q50, q90 = np.quantile(arr_np, [0.1, 0.5, 0.9])
                print(f"{lbl}: n={arr_np.size} achieved in order | P10/Median/P90 (years): {q10:.3f} / {q50:.3f} / {q90:.3f}")
                if n_not_achieved > 0:
                    milestone_b = lbl.split(" to ")[1] if " to " in lbl else "B"
                    print(f"  +{n_not_achieved}/{total_a} {milestone_b} not achieved")
                if n_before > 0:
                    print(f"  +{n_before}/{total_a} out of order")
            else:
                print(f"{lbl}: n=0 achieved in order (total where A achieved: {total_a})")
        plot_milestone_transition_boxplot(
            labels,
            durations,
            durations_censored,
            num_b_not_achieved,
            num_b_before_a,
            total_per_pair,
            out_path=out_path,
            title=None,
            ymin_years=(float(args.ymin_years) if args.ymin_years is not None else None),
            ymax_years=(float(args.ymax_years) if args.ymax_years is not None else None),
            exclude_inf_from_stats=bool(args.exclude_inf_from_stats),
            inf_years_cap=typical_max,
            inf_years_display=float(args.inf_years),
            condition_text=(f"{args.filter_milestone} achieved by {args.filter_by_year}" if args.filter_milestone and args.filter_by_year is not None else None),
        )
        print(f"Saved transition boxplot to: {out_path}")
        return

    if args.mode == "milestone_scatter":
        if not args.scatter_pair:
            raise ValueError("--scatter-pair is required for milestone_scatter mode (e.g., 'AIR-5x:AIR-2000x')")
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
    times, trajectories, aa_times = _read_horizon_trajectories(rollouts_path)
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
        aa_times=aa_times,
    )
    print(f"Saved horizon trajectories to: {out_path}")


if __name__ == "__main__":
    main()

