#!/usr/bin/env python3
"""
Generate a summary table for key milestones showing:
- Median arrival date
- Difference between median arrival dates
- Median transition duration (assuming achieved at simulation cutoff)
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


def extract_milestone_data(
    rollouts_file: Path,
    milestone_names: List[str]
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Extract arrival times and transition durations for given milestones.

    Returns:
        - arrival_times: dict mapping milestone name to list of arrival times (only achieved)
        - arrival_times_with_censored: dict mapping milestone name to list of arrival times (including censored at sim end)
        - transitions_with_censored: dict mapping (from, to) pair to list of durations (including censored)
    """
    arrival_times: Dict[str, List[float]] = {name: [] for name in milestone_names}
    arrival_times_with_censored: Dict[str, List[float]] = {name: [] for name in milestone_names}

    # For transitions, we'll track consecutive pairs
    transitions_with_censored: Dict[Tuple[str, str], List[float]] = {}
    for i in range(len(milestone_names) - 1):
        transitions_with_censored[(milestone_names[i], milestone_names[i+1])] = []

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

            # Get simulation end time for censored data
            times_array = results.get("times")
            if times_array is not None and len(times_array) > 0:
                try:
                    simulation_end = float(times_array[-1])
                except Exception:
                    simulation_end = None
            else:
                simulation_end = None

            # Extract arrival times for each milestone
            milestone_times: Dict[str, Optional[float]] = {}
            for name in milestone_names:
                m = milestones.get(name)
                if isinstance(m, dict) and m.get("time") is not None:
                    try:
                        t = float(m["time"])
                        if np.isfinite(t):
                            milestone_times[name] = t
                            arrival_times[name].append(t)
                            arrival_times_with_censored[name].append(t)
                        else:
                            milestone_times[name] = None
                            # Not achieved - use simulation end
                            if simulation_end is not None:
                                arrival_times_with_censored[name].append(simulation_end)
                    except Exception:
                        milestone_times[name] = None
                        # Not achieved - use simulation end
                        if simulation_end is not None:
                            arrival_times_with_censored[name].append(simulation_end)
                else:
                    milestone_times[name] = None
                    # Not achieved - use simulation end
                    if simulation_end is not None:
                        arrival_times_with_censored[name].append(simulation_end)

            # Calculate transition durations (with censored data)
            for i in range(len(milestone_names) - 1):
                from_name = milestone_names[i]
                to_name = milestone_names[i + 1]

                t_from = milestone_times.get(from_name)
                t_to = milestone_times.get(to_name)

                if t_from is None:
                    # If first milestone not achieved, skip
                    continue

                if t_to is None:
                    # Second milestone not achieved - use simulation end as censored value
                    if simulation_end is not None and simulation_end > t_from:
                        duration = simulation_end - t_from
                        transitions_with_censored[(from_name, to_name)].append(duration)
                elif t_to > t_from:
                    # Both achieved, normal case
                    duration = t_to - t_from
                    transitions_with_censored[(from_name, to_name)].append(duration)

    return arrival_times, arrival_times_with_censored, transitions_with_censored


def format_year(value: float) -> str:
    """Format a year value."""
    return f"{value:.1f}"


def format_duration(value: float) -> str:
    """Format a duration in years."""
    return f"{value:.1f}"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate milestone summary table")
    parser.add_argument("rollouts_file", type=Path, help="Path to rollouts.jsonl file")
    parser.add_argument("-o", "--output", type=Path, help="Output CSV file path (default: same directory as rollouts file)")
    args = parser.parse_args()

    if not args.rollouts_file.exists():
        print(f"Error: File not found: {args.rollouts_file}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.rollouts_file.parent / "milestone_summary.csv"

    # Milestones to analyze
    milestone_names = [
        "ACD-AI",
        "SAR-level-experiment-selection-skill",
        "SIAR-level-experiment-selection-skill"
    ]

    print("Extracting data from rollouts...")
    arrival_times, arrival_times_with_censored, transitions_with_censored = extract_milestone_data(
        args.rollouts_file,
        milestone_names
    )

    # Prepare data for CSV
    rows = []
    prev_median_arrival = None

    for i, name in enumerate(milestone_names):
        times = arrival_times[name]
        times_censored = arrival_times_with_censored[name]

        # Median arrival date (including censored data)
        if len(times_censored) > 0:
            median_arrival = np.median(times_censored)
            median_arrival_str = format_year(median_arrival)
        else:
            median_arrival = None
            median_arrival_str = "N/A"

        # Difference from previous median arrival
        if i == 0 or prev_median_arrival is None or median_arrival is None:
            delta_str = "-"
        else:
            delta = median_arrival - prev_median_arrival
            delta_str = format_duration(delta)

        # Median transition from previous (with censored data)
        if i == 0:
            median_transition_str = "-"
        else:
            prev_name = milestone_names[i - 1]
            trans_durations = transitions_with_censored.get((prev_name, name), [])
            if len(trans_durations) > 0:
                median_transition = np.median(trans_durations)
                median_transition_str = format_duration(median_transition)
            else:
                median_transition_str = "N/A"

        # Achievement statistics
        total_runs = len(arrival_times["ACD-AI"]) if "ACD-AI" in arrival_times else len(times)
        achieved_count = len(times)
        achieved_pct = 100.0 * achieved_count / total_runs if total_runs > 0 else 0.0

        rows.append({
            'Milestone': name,
            'Median Arrival': median_arrival_str,
            'Delta from Previous': delta_str,
            'Median Transition': median_transition_str,
            'Achieved Count': achieved_count,
            'Total Runs': total_runs,
            'Achievement Rate (%)': f"{achieved_pct:.1f}"
        })

        prev_median_arrival = median_arrival

    # Write to CSV
    with output_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = ['Milestone', 'Median Arrival', 'Delta from Previous', 'Median Transition',
                      'Achieved Count', 'Total Runs', 'Achievement Rate (%)']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTable written to: {output_path}")
    print("\nSummary:")
    for row in rows:
        print(f"  {row['Milestone']}: Median arrival {row['Median Arrival']}, "
              f"Δ={row['Delta from Previous']}, "
              f"Transition={row['Median Transition']}, "
              f"Achieved {row['Achievement Rate (%)']}%")


if __name__ == "__main__":
    main()
