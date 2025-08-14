#!/usr/bin/env python3
"""
Flask blueprint for configuring parameter distributions and launching
Monte Carlo (batch) rollouts via the existing scripts/batch_rollout.py.
"""

import os
import sys
import json
import uuid
import time
import threading
import subprocess
from pathlib import Path
import zipfile
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, render_template, request, send_file, abort

import yaml

import model_config as cfg


mc_bp = Blueprint("monte_carlo", __name__)


REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
OUTPUTS_DIR = REPO_ROOT / "outputs"
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_SAMPLING_CFG_PATH = CONFIG_DIR / "sampling_config.yaml"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _list_output_dirs() -> List[Path]:
    if not OUTPUTS_DIR.exists():
        return []
    return sorted([p for p in OUTPUTS_DIR.iterdir() if p.is_dir()])


def _build_default_distribution_config() -> Dict[str, Any]:
    """Generate a default independent distribution config from model bounds/defaults.

    Mirrors the CLI behavior: use uniform within bounds, else fixed at default.
    """
    param_cfg: Dict[str, Any] = {}
    for name, default_val in cfg.DEFAULT_PARAMETERS.items():
        if name == "automation_anchors":
            # Computed internally per sample; do not sample
            continue
        bounds = cfg.PARAMETER_BOUNDS.get(name)
        if bounds is not None:
            param_cfg[name] = {
                "dist": "uniform",
                "min": bounds[0],
                "max": bounds[1],
                "clip_to_bounds": True,
            }
        else:
            param_cfg[name] = {"dist": "fixed", "value": default_val}

    # Include categorical parameters if not already present
    if "horizon_extrapolation_type" not in param_cfg:
        param_cfg["horizon_extrapolation_type"] = {
            "dist": "choice",
            "values": cfg.HORIZON_EXTRAPOLATION_TYPES,
            "p": None,
        }
    if "taste_schedule_type" not in param_cfg:
        param_cfg["taste_schedule_type"] = {
            "dist": "choice",
            "values": cfg.TASTE_SCHEDULE_TYPES,
            "p": None,
        }

    return {
        "seed": 42,
        "parameters": param_cfg,
        "time_range": None,
        "initial_progress": 0.0,
    }


# In-memory job registry
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _detect_new_run_dir(before: List[Path], after: List[Path], stdout: str) -> Optional[Path]:
    # Prefer exact diff
    before_set = set(before)
    diff = sorted([p for p in after if p not in before_set])
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


def _launch_batch_rollout_job(effective_cfg: Dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex[:12]
    tmp_dir = OUTPUTS_DIR / "tmp" / job_id
    _ensure_dir(tmp_dir)

    # Write config file to pass to the script
    cfg_path = tmp_dir / "sampling_config.yaml"
    cfg_path.write_text(yaml.safe_dump(effective_cfg))

    # Prepare command (let the script manage the timestamped run directory under outputs/)
    python_exe = sys.executable
    cmd = [
        python_exe,
        str(SCRIPTS_DIR / "batch_rollout.py"),
        "--config",
        str(cfg_path),
    ]
    # Allow CLI overrides for num-samples, input-data, time-range, seed
    if "num_samples" in effective_cfg:
        cmd += ["--num-samples", str(int(effective_cfg["num_samples"]))]
    if "input_data" in effective_cfg and effective_cfg["input_data"]:
        cmd += ["--input-data", str(effective_cfg["input_data"])]
    if "time_range" in effective_cfg and effective_cfg["time_range"]:
        tr = effective_cfg["time_range"]
        if isinstance(tr, (list, tuple)) and len(tr) == 2:
            cmd += ["--time-range", str(float(tr[0])), str(float(tr[1]))]
    if "seed" in effective_cfg:
        cmd += ["--seed", str(int(effective_cfg["seed"]))]

    # Capture stdout/stderr to a log file and buffer for quick status
    log_path = tmp_dir / "job.log"
    before_dirs = _list_output_dirs()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    job_info: Dict[str, Any] = {
        "job_id": job_id,
        "pid": proc.pid,
        "status": "running",
        "created_at": time.time(),
        "updated_at": time.time(),
        "log_path": str(log_path),
        "output_dir": None,
        "cmd": cmd,
    }
    with _jobs_lock:
        _jobs[job_id] = job_info

    def _reader_thread():
        lines: List[str] = []
        with log_path.open("w", encoding="utf-8") as lf:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    lf.write(line)
                    lf.flush()
                    lines.append(line)
            finally:
                proc.wait()
                # Finalize status
                after_dirs = _list_output_dirs()
                stdout_text = "".join(lines)
                out_dir = _detect_new_run_dir(before_dirs, after_dirs, stdout_text)
                # If succeeded, run plotting/sensitivity to produce artifacts
                artifacts: List[str] = []
                if proc.returncode == 0 and out_dir is not None:
                    try:
                        with log_path.open("a", encoding="utf-8") as lf_app:
                            # Timeline histogram
                            try:
                                plot_cmd = [sys.executable, str(SCRIPTS_DIR / "plot_rollouts.py"), "--run-dir", str(out_dir)]
                                subprocess.run(plot_cmd, cwd=str(REPO_ROOT), stdout=lf_app, stderr=subprocess.STDOUT, check=False)
                            except Exception as _e:
                                lf_app.write(f"\n[WARN] plot_rollouts failed: {_e}\n")
                            # Sensitivity (with plots)
                            try:
                                sens_cmd = [sys.executable, str(SCRIPTS_DIR / "sensitivity_analysis.py"), "--run-dir", str(out_dir), "--plot"]
                                subprocess.run(sens_cmd, cwd=str(REPO_ROOT), stdout=lf_app, stderr=subprocess.STDOUT, check=False)
                            except Exception as _e:
                                lf_app.write(f"\n[WARN] sensitivity_analysis failed: {_e}\n")
                    except Exception:
                        pass
                    # Collect known artifact filenames if present
                    try:
                        known = [
                            "sc_time_hist.png",
                            "sensitivity_spearman_top.png",
                            "sensitivity_permutation_top.png",
                        ]
                        for name in known:
                            p = Path(out_dir) / name
                            if p.exists() and p.is_file():
                                artifacts.append(name)
                    except Exception:
                        artifacts = []
                with _jobs_lock:
                    ji = _jobs.get(job_id)
                    if ji is not None:
                        ji["updated_at"] = time.time()
                        ji["exit_code"] = proc.returncode
                        ji["output_dir"] = str(out_dir) if out_dir else None
                        if artifacts:
                            ji["artifacts"] = artifacts
                        ji["status"] = "succeeded" if proc.returncode == 0 else "failed"

    t = threading.Thread(target=_reader_thread, daemon=True)
    t.start()

    return job_id


@mc_bp.route("/monte-carlo")
def monte_carlo_page():
    return render_template("monte_carlo.html")


@mc_bp.route("/api/monte-carlo/default-config", methods=["GET"])
def get_default_sampling_config():
    source = request.args.get("source", "generated")
    # Optionally load the repo config file
    if source == "file" and DEFAULT_SAMPLING_CFG_PATH.exists():
        try:
            text = DEFAULT_SAMPLING_CFG_PATH.read_text()
            data = yaml.safe_load(text)
            return jsonify({"success": True, "config": data, "source": "file"})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # Generated default based on bounds/defaults
    config = _build_default_distribution_config()
    return jsonify({"success": True, "config": config, "source": "generated"})


@mc_bp.route("/api/monte-carlo/run", methods=["POST"])
def run_monte_carlo():
    try:
        payload = request.json or {}
        # Expect a dict with keys: seed, num_samples, time_range, initial_progress, input_data, parameters
        effective_cfg: Dict[str, Any] = {
            "seed": payload.get("seed", 42),
            "num_samples": payload.get("num_samples", payload.get("num_rollouts", 1000)),
            "time_range": payload.get("time_range"),
            "initial_progress": payload.get("initial_progress", 0.0),
            "input_data": payload.get("input_data"),
            "parameters": payload.get("parameters", {}),
        }

        # Remove problematic/null fields so the script falls back to its defaults
        # Validate input_data if provided; else remove so script uses its default
        input_path = effective_cfg.get("input_data")
        if not input_path:
            effective_cfg.pop("input_data", None)
        else:
            try:
                ip = Path(str(input_path)).expanduser()
                if not ip.is_absolute():
                    # Interpret relative paths relative to repo root
                    ip = (REPO_ROOT / ip).resolve()
                if not ip.exists():
                    return jsonify({
                        "success": False,
                        "error": f"Input data file not found: {ip}. Provide a valid path or leave blank to use input_data.csv."
                    }), 400
                effective_cfg["input_data"] = str(ip)
            except Exception:
                return jsonify({
                    "success": False,
                    "error": "Invalid input_data path provided."
                }), 400
        # Normalize time_range
        tr = effective_cfg.get("time_range")
        if not (isinstance(tr, (list, tuple)) and len(tr) == 2):
            effective_cfg["time_range"] = None

        # Validate basic structure
        if not isinstance(effective_cfg["parameters"], dict) or not effective_cfg["parameters"]:
            return jsonify({"success": False, "error": "Missing or invalid 'parameters' distribution config"}), 400

        job_id = _launch_batch_rollout_job(effective_cfg)
        return jsonify({"success": True, "job_id": job_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@mc_bp.route("/api/monte-carlo/status/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"success": False, "error": "Job not found"}), 404

        # Optionally include the last N lines of the log
        tail = int(request.args.get("tail", 50))
        log_tail: List[str] = []
        log_path = Path(job.get("log_path", ""))
        if log_path.exists() and log_path.is_file():
            try:
                lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                if tail > 0:
                    log_tail = lines[-tail:]
            except Exception:
                log_tail = []

        return jsonify({
            "success": True,
            "job": {
                k: v for k, v in job.items() if k not in {"cmd"}
            },
            "log_tail": log_tail,
        })


@mc_bp.route("/api/monte-carlo/jobs", methods=["GET"])
def list_jobs():
    with _jobs_lock:
        jobs = [
            {k: v for k, v in j.items() if k not in {"cmd"}}
            for j in _jobs.values()
        ]
    # Sort by created_at desc
    jobs.sort(key=lambda j: j.get("created_at", 0.0), reverse=True)
    return jsonify({"success": True, "jobs": jobs})


@mc_bp.route("/api/monte-carlo/artifact/<job_id>/<path:filename>", methods=["GET"])
def get_artifact(job_id: str, filename: str):
    # Serve artifacts only from the resolved output_dir for this job
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"success": False, "error": "Job not found"}), 404
    out_dir = job.get("output_dir")
    if not out_dir:
        return jsonify({"success": False, "error": "Output directory not set yet"}), 404
    base = Path(out_dir)
    # Prevent traversal by resolving and ensuring it is under base
    target = (base / filename).resolve()
    try:
        base_resolved = base.resolve()
    except Exception:
        base_resolved = base
    if not str(target).startswith(str(base_resolved)):
        abort(403)
    if not target.exists() or not target.is_file():
        return jsonify({"success": False, "error": "Artifact not found"}), 404
    return send_file(str(target))


@mc_bp.route("/api/monte-carlo/download/<job_id>", methods=["GET"])
def download_run_zip(job_id: str):
    # Create and stream a ZIP of the entire timestamped output directory for this job
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"success": False, "error": "Job not found"}), 404
    out_dir_str = job.get("output_dir")
    if not out_dir_str:
        return jsonify({"success": False, "error": "Output directory not set yet"}), 404
    out_dir = Path(out_dir_str)
    if not out_dir.exists() or not out_dir.is_dir():
        return jsonify({"success": False, "error": "Output directory not found"}), 404

    # Prepare zip path under tmp dir for this job
    zip_tmp_dir = OUTPUTS_DIR / "tmp" / job_id
    _ensure_dir(zip_tmp_dir)
    zip_name = f"{out_dir.name}.zip"
    zip_path = zip_tmp_dir / zip_name

    # Build zip fresh each request to ensure it includes all artifacts
    try:
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Add files recursively; preserve relative paths under the timestamp folder
            for root, dirs, files in os.walk(out_dir):
                root_path = Path(root)
                for file_name in files:
                    file_path = root_path / file_name
                    # Compute archive name relative to parent of out_dir so that zip contains the timestamp folder
                    arcname = str(out_dir.name / file_path.relative_to(out_dir)) if False else str(Path(out_dir.name) / file_path.relative_to(out_dir))
                    zf.write(str(file_path), arcname)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to build zip: {e}"}), 500

    # Stream the zip file
    return send_file(str(zip_path), as_attachment=True, download_name=zip_name)

