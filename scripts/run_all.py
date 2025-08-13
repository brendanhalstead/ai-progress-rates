#!/usr/bin/env python3
"""
Run the full pipeline with one command (no arguments):
  1) Run batch_rollout.py using config/sampling_config.yaml
  2) Generate plots using plot_rollouts.py for that run
  3) Run sensitivity_analysis.py on that run (including plots)

All artifacts are saved inside the newly created outputs/<timestamp>/ directory.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Set


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
OUTPUTS_DIR = REPO_ROOT / "outputs"
CONFIG_PATH = REPO_ROOT / "config" / "sampling_config.yaml"


def _list_output_dirs() -> Set[Path]:
    if not OUTPUTS_DIR.exists():
        return set()
    return {p for p in OUTPUTS_DIR.iterdir() if p.is_dir()}


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    # Stream output live to inherit the progress bar rendering
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=True)


def _detect_new_run_dir(before: Set[Path], after: Set[Path], stdout: str) -> Optional[Path]:
    # Prefer exact diff
    diff = sorted(after - before)
    if len(diff) == 1:
        return diff[0]
    # Try to parse from stdout line: "Run complete. Artifacts in: <path>"
    for line in stdout.splitlines():
        line = line.strip()
        marker = "Artifacts in: "
        if marker in line:
            path_str = line.split(marker, 1)[1].strip()
            p = Path(path_str)
            if p.exists() and p.is_dir():
                return p
    # Fallback to latest mtime
    if after:
        return max(after, key=lambda p: p.stat().st_mtime)
    return None


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    before_dirs = _list_output_dirs()

    # 1) Batch rollout
    batch_cmd = [sys.executable, str(SCRIPTS_DIR / "batch_rollout.py"), "--config", str(CONFIG_PATH)]
    batch_res = _run(batch_cmd)

    after_dirs = _list_output_dirs()
    # We no longer capture stdout, so detect by diff/mtime
    run_dir = _detect_new_run_dir(before_dirs, after_dirs, stdout="")
    if run_dir is None:
        raise RuntimeError("Failed to determine output run directory")
    print(f"Resolved run directory: {run_dir}")

    # 2) Plots
    # 2a) SC time histogram
    _run([sys.executable, str(SCRIPTS_DIR / "plot_rollouts.py"), "--run-dir", str(run_dir), "--mode", "sc_hist"]) 
    # 2b) Horizon trajectories (full)
    _run([sys.executable, str(SCRIPTS_DIR / "plot_rollouts.py"), "--run-dir", str(run_dir), "--mode", "horizon_trajectories"]) 
    # 2c) Horizon trajectories, stopping at SC for each rollout
    _run([sys.executable, str(SCRIPTS_DIR / "plot_rollouts.py"), "--run-dir", str(run_dir), "--mode", "horizon_trajectories", "--stop-at-sc"]) 
    # 2d) Histogram of horizon at SC
    _run([sys.executable, str(SCRIPTS_DIR / "plot_rollouts.py"), "--run-dir", str(run_dir), "--mode", "horizon_at_sc_hist"]) 

    # 3) Sensitivity analysis (+ plots)
    sens_cmd = [sys.executable, str(SCRIPTS_DIR / "sensitivity_analysis.py"), "--run-dir", str(run_dir), "--plot"]
    _run(sens_cmd)

    print(f"Pipeline complete. Artifacts are in: {run_dir}")


if __name__ == "__main__":
    main()

