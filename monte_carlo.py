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
import shutil
import signal

from flask import Blueprint, jsonify, render_template, request, send_file, abort, Response

import yaml
import io
import numpy as np
from datetime import datetime, timedelta

# Configure matplotlib for headless servers (for live histogram rendering)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # Endpoint will handle absence gracefully

# Optional KDE support
try:
    from scipy.stats import gaussian_kde  # type: ignore
except Exception:
    gaussian_kde = None  # KDE overlay will be skipped if SciPy unavailable

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
        start_new_session=True,  # make child the leader of a new process group for safe cancellation
    )

    job_info: Dict[str, Any] = {
        "job_id": job_id,
        "pid": proc.pid,
        "pgrp": proc.pid,  # process group id (same as pid when start_new_session=True)
        "status": "running",
        "created_at": time.time(),
        "updated_at": time.time(),
        "log_path": str(log_path),
        "tmp_dir": str(tmp_dir),
        "output_dir": None,
        "output_dir_confirmed": False,
        "outputs_before": [str(p) for p in before_dirs],
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
                    # Try to detect the run directory early while the job is running
                    with _jobs_lock:
                        ji_early = _jobs.get(job_id)
                        current_out = ji_early.get("output_dir") if ji_early else None
                    if not current_out:
                        try:
                            # Choose most recently modified directory under outputs/
                            existing = _list_output_dirs()
                            if existing:
                                cand = max(existing, key=lambda p: p.stat().st_mtime)
                                with _jobs_lock:
                                    ji_set = _jobs.get(job_id)
                                    if ji_set is not None:
                                        ji_set["output_dir"] = str(cand)
                        except Exception:
                            pass
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
                            # Horizon trajectories plot
                            try:
                                plot_cmd_ht = [
                                    sys.executable,
                                    str(SCRIPTS_DIR / "plot_rollouts.py"),
                                    "--run-dir",
                                    str(out_dir),
                                    "--mode",
                                    "horizon_trajectories",
                                ]
                                subprocess.run(plot_cmd_ht, cwd=str(REPO_ROOT), stdout=lf_app, stderr=subprocess.STDOUT, check=False)
                            except Exception as _e:
                                lf_app.write(f"\n[WARN] plot_rollouts horizon_trajectories failed: {_e}\n")
                            # Sensitivity (with plots)
                            try:
                                sens_cmd = [sys.executable, str(SCRIPTS_DIR / "sensitivity_analysis.py"), "--run-dir", str(out_dir), "--plot"]
                                subprocess.run(sens_cmd, cwd=str(REPO_ROOT), stdout=lf_app, stderr=subprocess.STDOUT, check=False)
                            except Exception as _e:
                                lf_app.write(f"\n[WARN] sensitivity_analysis failed: {_e}\n")
                            # SC-by-quarter table artifacts
                            try:
                                scq_cmd = [sys.executable, str(SCRIPTS_DIR / "sc_by_quarter.py"), "--run-dir", str(out_dir)]
                                subprocess.run(scq_cmd, cwd=str(REPO_ROOT), stdout=lf_app, stderr=subprocess.STDOUT, check=False)
                            except Exception as _e:
                                lf_app.write(f"\n[WARN] sc_by_quarter failed: {_e}\n")
                    except Exception:
                        pass
                    # Collect known artifact filenames if present
                    try:
                        known = [
                            "sc_time_hist.png",
                            "horizon_trajectories.png",
                            "sensitivity_pearson_top.png",
                            "sensitivity_spearman_top.png",
                            "sensitivity_permutation_top.png",
                            "sc_by_quarter.html",
                            "sc_by_quarter.csv",
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
                        if out_dir is not None:
                            ji["output_dir"] = str(out_dir)
                            ji["output_dir_confirmed"] = True
                        if artifacts:
                            ji["artifacts"] = artifacts
                        # Don't override explicit cancellation status
                        if ji.get("status") != "cancelled":
                            ji["status"] = "succeeded" if proc.returncode == 0 else "failed"

    t = threading.Thread(target=_reader_thread, daemon=True)
    t.start()

    return job_id


@mc_bp.route("/monte-carlo")
def monte_carlo_page():
    return render_template("monte_carlo.html")


@mc_bp.route("/api/monte-carlo/default-config", methods=["GET"])
def get_default_sampling_config():
    # Always load from config/sampling_config.yaml; no generated fallback
    if not DEFAULT_SAMPLING_CFG_PATH.exists():
        return jsonify({
            "success": False,
            "error": f"sampling_config.yaml not found at {DEFAULT_SAMPLING_CFG_PATH}"
        }), 404
    try:
        text = DEFAULT_SAMPLING_CFG_PATH.read_text()
        data = yaml.safe_load(text)
        return jsonify({"success": True, "config": data, "source": "file"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


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


@mc_bp.route("/api/monte-carlo/export-config", methods=["POST"])
def export_sampling_config():
    """Convert current UI config JSON to a sampling_config.yaml text and return it.

    Expects JSON body matching the UI's collectConfigFromUI() output.
    Returns YAML text with keys: seed, initial_progress, time_range, num_rollouts, input_data, parameters.
    """
    try:
        payload = request.json or {}

        # Map UI fields to YAML schema
        yaml_cfg: Dict[str, Any] = {
            "seed": payload.get("seed", 42),
            "initial_progress": payload.get("initial_progress", 0.0),
            "time_range": payload.get("time_range"),
            "num_rollouts": payload.get("num_samples", payload.get("num_rollouts", 1000)),
            "input_data": payload.get("input_data") or None,
            "parameters": payload.get("parameters", {}),
        }

        # Basic validation
        if not isinstance(yaml_cfg["parameters"], dict) or not yaml_cfg["parameters"]:
            return jsonify({"success": False, "error": "Missing or invalid 'parameters' distribution config"}), 400

        # Normalize time_range
        tr = yaml_cfg.get("time_range")
        if not (isinstance(tr, (list, tuple)) and len(tr) == 2):
            yaml_cfg.pop("time_range", None)
        else:
            try:
                yaml_cfg["time_range"] = [float(tr[0]), float(tr[1])]
            except Exception:
                yaml_cfg.pop("time_range", None)

        # Drop input_data if empty
        if not yaml_cfg.get("input_data"):
            yaml_cfg.pop("input_data", None)

        # Dump to YAML (preserve key order, human-friendly)
        yaml_text = yaml.safe_dump(yaml_cfg, sort_keys=False)
        # Return as a file-like response so the client can download easily
        headers = {
            "Content-Disposition": "attachment; filename=\"sampling_config.yaml\""
        }
        return Response(yaml_text, mimetype="application/x-yaml", headers=headers)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@mc_bp.route("/api/monte-carlo/import-config", methods=["POST"])
def import_sampling_config():
    """Parse an uploaded sampling_config.yaml and return a UI-ready JSON config.

    Accepts either:
      - multipart/form-data with a file field named 'file'
      - application/json with a 'yaml' string field
    Returns: { success: True, config: {...} }
    """
    try:
        yaml_text: Optional[str] = None

        # Multipart file upload path
        if "file" in request.files:
            f = request.files["file"]
            yaml_text = f.read().decode("utf-8", errors="ignore")
        elif request.is_json:
            body = request.get_json(silent=True) or {}
            y = body.get("yaml")
            if isinstance(y, str) and y.strip():
                yaml_text = y

        if not yaml_text:
            return jsonify({"success": False, "error": "No YAML provided"}), 400

        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            return jsonify({"success": False, "error": "Invalid YAML: expected a mapping at top-level"}), 400

        params = data.get("parameters", {})
        if not isinstance(params, dict) or not params:
            return jsonify({"success": False, "error": "YAML missing 'parameters' mapping"}), 400

        # Build UI-ready config; UI supports either num_samples or num_rollouts
        num_rollouts = data.get("num_rollouts")
        if not isinstance(num_rollouts, (int, float)):
            num_rollouts = data.get("num_samples")

        ui_cfg: Dict[str, Any] = {
            "seed": data.get("seed", 42),
            "initial_progress": data.get("initial_progress", 0.0),
            "time_range": data.get("time_range"),
            "num_rollouts": num_rollouts,
            "num_samples": num_rollouts,
            "input_data": data.get("input_data"),
            "parameters": params,
        }

        return jsonify({"success": True, "config": ui_cfg})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@mc_bp.route("/api/monte-carlo/cancel/<job_id>", methods=["POST"]) 
def cancel_job(job_id: str):
    """Cancel a running job and delete its files (tmp and output dirs)."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"success": False, "error": "Job not found"}), 404

    pid = int(job.get("pid") or 0)
    pgrp = int(job.get("pgrp") or pid or 0)
    # Attempt to terminate the process group first
    try:
        if pgrp:
            os.killpg(pgrp, signal.SIGTERM)  # type: ignore[arg-type]
        elif pid:
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass

    # Best-effort hard kill after a short grace period
    try:
        time.sleep(0.5)
        if pgrp:
            os.killpg(pgrp, signal.SIGKILL)  # type: ignore[arg-type]
        elif pid:
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass

    # Update job status
    with _jobs_lock:
        ji = _jobs.get(job_id)
        if ji is not None:
            ji["status"] = "cancelled"
            ji["updated_at"] = time.time()
            ji["exit_code"] = -signal.SIGTERM

    # Delete files/directories associated with the job
    tmp_dir = Path(job.get("tmp_dir") or (OUTPUTS_DIR / "tmp" / job_id))
    out_dir = Path(job.get("output_dir") or "") if job.get("output_dir") else None
    try:
        if out_dir and out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
    except Exception:
        pass
    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    return jsonify({"success": True, "cancelled": True})


@mc_bp.route("/api/monte-carlo/cleanup", methods=["POST"])
def cleanup_jobs():
    """Bulk cleanup: cancel running jobs and delete their files for provided job_ids.

    Body: { "job_ids": ["abc123", ...] }
    """
    try:
        payload = request.json or {}
        job_ids = payload.get("job_ids", [])
        if not isinstance(job_ids, list):
            return jsonify({"success": False, "error": "job_ids must be a list"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

    results: List[Dict[str, Any]] = []
    for jid in job_ids:
        with _jobs_lock:
            job = _jobs.get(jid)
        if job is None:
            results.append({"job_id": jid, "found": False})
            continue
        status = job.get("status")
        pid = int(job.get("pid") or 0)
        pgrp = int(job.get("pgrp") or pid or 0)
        # If running, terminate
        if status == "running":
            try:
                if pgrp:
                    os.killpg(pgrp, signal.SIGTERM)  # type: ignore[arg-type]
                elif pid:
                    os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
            try:
                time.sleep(0.2)
                if pgrp:
                    os.killpg(pgrp, signal.SIGKILL)  # type: ignore[arg-type]
                elif pid:
                    os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        # Delete files
        tmp_dir = Path(job.get("tmp_dir") or (OUTPUTS_DIR / "tmp" / jid))
        out_dir = Path(job.get("output_dir") or "") if job.get("output_dir") else None
        try:
            if out_dir and out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
        except Exception:
            pass
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

        # Remove from registry
        with _jobs_lock:
            _jobs.pop(jid, None)

        results.append({"job_id": jid, "found": True, "deleted": True})

    return jsonify({"success": True, "results": results})


@mc_bp.route("/api/monte-carlo/live-sc-hist/<job_id>.png", methods=["GET"])
def live_sc_histogram(job_id: str):
    """Serve a live-updating SC-time histogram PNG based on current rollouts.jsonl."""
    if plt is None:
        return jsonify({"success": False, "error": "matplotlib not available"}), 500

    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"success": False, "error": "Job not found"}), 404
    out_dir = job.get("output_dir")
    if not out_dir:
        return jsonify({"success": False, "error": "Output directory not available yet"}), 404

    rollouts_path = Path(out_dir) / "rollouts.jsonl"
    if not rollouts_path.exists():
        return jsonify({"success": False, "error": "rollouts.jsonl not found yet"}), 404

    sc_times: List[float] = []
    num_no_sc: int = 0
    try:
        with rollouts_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                res = obj.get("results") if isinstance(obj, dict) else None
                if isinstance(res, dict):
                    val = res.get("sc_time")
                    try:
                        v = float(val) if val is not None else np.nan
                    except Exception:
                        v = np.nan
                    if np.isfinite(v):
                        sc_times.append(v)
                    else:
                        num_no_sc += 1
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to read rollouts: {e}"}), 500

    def _decimal_year_to_date_string(decimal_year: float) -> str:
        try:
            year = int(np.floor(decimal_year))
            frac = float(decimal_year - year)
            start = datetime(year, 1, 1)
            end = datetime(year + 1, 1, 1)
            total_seconds = (end - start).total_seconds()
            dt = start + timedelta(seconds=frac * total_seconds)
            return dt.date().isoformat()
        except Exception:
            return f"{decimal_year:.3f}"

    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    try:
        ax.set_title("Live SC-time Distribution", fontsize=10)
        total_n = len(sc_times) + num_no_sc
        if total_n > 0:
            data = np.asarray(sc_times, dtype=float) if len(sc_times) > 0 else np.asarray([], dtype=float)
            kde_counts = np.array([])
            bin_edges = np.array([0.0, 1.0])
            counts = np.array([])
            bin_width = 1.0
            nosc_x = 1.0

            if len(data) > 0:
                bins = min(50, max(10, int(np.sqrt(len(data)))))
                counts, bin_edges, _ = ax.hist(
                    data,
                    bins=bins,
                    edgecolor="black",
                    alpha=0.6,
                    label="Histogram",
                    color="#4E79A7",
                )
                bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
                # KDE overlay if available and data is not degenerate
                if gaussian_kde is not None and len(data) >= 3 and np.std(data) > 0:
                    try:
                        xs = np.linspace(float(data.min()), float(data.max()), 512)
                        kde = gaussian_kde(data)
                        kde_counts = kde(xs) * len(data) * bin_width
                        ax.plot(xs, kde_counts, color="tab:orange", linewidth=2.0, label="Gaussian KDE")
                    except Exception:
                        kde_counts = np.array([])
                nosc_x = float(bin_edges[-1] + bin_width / 2.0)

            # Draw No SC bar if needed and adjust ticks/limits
            if num_no_sc > 0:
                ax.bar(nosc_x, num_no_sc, width=bin_width, edgecolor="black", alpha=0.6, label="No SC")
                left = float(bin_edges[0]) if len(sc_times) > 0 else 0.0
                right = float(nosc_x + bin_width)
                ax.set_xlim(left, right)

            # Percentiles and annotations including No SC as +inf
            try:
                if num_no_sc > 0:
                    with_inf = np.concatenate([data, np.full(int(num_no_sc), np.inf)])
                else:
                    with_inf = data
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

                ax.axvline(x10, color="tab:gray", linestyle="--", linewidth=1.5, label="P10")
                ax.axvline(x50, color="tab:green", linestyle="-", linewidth=1.75, label="Median")
                ax.axvline(x90, color="tab:gray", linestyle="--", linewidth=1.5, label="P90")
                ax.text(x10, y_annot, f"P10: {lbl10}", rotation=90,
                        va="top", ha="right", color="tab:gray", fontsize=8,
                        bbox=dict(facecolor=(1,1,1,0.6), edgecolor='none', pad=1.5))
                ax.text(x50, y_annot, f"Median: {lbl50}", rotation=90,
                        va="top", ha="right", color="tab:green", fontsize=8,
                        bbox=dict(facecolor=(1,1,1,0.6), edgecolor='none', pad=1.5))
                ax.text(x90, y_annot, f"P90: {lbl90}", rotation=90,
                        va="top", ha="right", color="tab:gray", fontsize=8,
                        bbox=dict(facecolor=(1,1,1,0.6), edgecolor='none', pad=1.5))
            except Exception:
                pass

            ax.set_xlabel("SC Time (decimal year)")
            ax.set_ylabel("Count")
            ax.grid(True, axis='y', alpha=0.25)
            ax.legend(loc="upper left", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Collecting rollouts...", ha='center', va='center', fontsize=12)
            ax.axis('off')

        ax.figure.text(0.99, 0.01, f"n={len(sc_times)}", ha='right', va='bottom', fontsize=8, color="#666666")

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    finally:
        plt.close(fig)


@mc_bp.route("/api/monte-carlo/live-horizon-trajectories/<job_id>.png", methods=["GET"])
def live_horizon_trajectories(job_id: str):
    """Serve a live-updating horizon trajectories PNG based on current rollouts.jsonl.
    
    Query params:
      - max: int, maximum trajectories to draw (default 2000)
      - stop_at_sc: 0/1, mask each trajectory after its own sc_time (default 0)
    """
    if plt is None:
        return jsonify({"success": False, "error": "matplotlib not available"}), 500

    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"success": False, "error": "Job not found"}), 404
    out_dir = job.get("output_dir")
    if not out_dir:
        return jsonify({"success": False, "error": "Output directory not available yet"}), 404

    rollouts_path = Path(out_dir) / "rollouts.jsonl"
    if not rollouts_path.exists():
        return jsonify({"success": False, "error": "rollouts.jsonl not found yet"}), 404

    # Parse query params
    try:
        max_trajectories = int(request.args.get("max", 2000))
    except Exception:
        max_trajectories = 2000
    stop_at_sc = request.args.get("stop_at_sc", "0") in {"1", "true", "True"}

    # Read trajectories similar to scripts/plot_rollouts.py
    times_arr: Optional[np.ndarray] = None
    trajectories: List[np.ndarray] = []
    sc_times: List[Optional[float]] = []
    try:
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
                sc_val = results.get("sc_time")
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
                try:
                    sc_times.append(float(sc_val) if sc_val is not None and np.isfinite(float(sc_val)) else None)
                except Exception:
                    sc_times.append(None)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to read rollouts: {e}"}), 500

    if times_arr is None or len(trajectories) == 0:
        # Render a placeholder image
        fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
        try:
            ax.text(0.5, 0.5, "Collecting rollouts...", ha='center', va='center', fontsize=12)
            ax.axis('off')
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            return send_file(buf, mimetype="image/png")
        finally:
            plt.close(fig)

    # Clean and optionally mask trajectories
    min_horizon_minutes = 0.001
    max_work_year_minutes = float(120000 * 52 * 40 * 60)
    cleaned: List[np.ndarray] = []
    for idx, t in enumerate(trajectories):
        arr = t.astype(float)
        arr[~np.isfinite(arr)] = np.nan
        arr[arr <= 0] = np.nan
        arr = np.clip(arr, min_horizon_minutes, max_work_year_minutes)
        if stop_at_sc and idx < len(sc_times) and sc_times[idx] is not None:
            sc = sc_times[idx]
            if sc is not None:
                arr = arr.copy()
                arr[times_arr > float(sc)] = np.nan
        cleaned.append(arr)

    # Compute median trajectory
    num_plot = min(len(cleaned), max_trajectories)
    if num_plot == 0:
        num_plot = min(len(cleaned), 1)
    stacked = np.vstack([cleaned[i] for i in range(num_plot)])
    median_traj = np.nanmedian(stacked, axis=0)

    # Tick values and labels (work-time units)
    def _get_time_tick_values_and_labels() -> Tuple[List[float], List[str]]:
        tick_values = [
            0.033333,
            0.5,
            2,
            8,
            30,
            120,
            480,
            2400,
            10380,
            41520,
            124560,
            622800,
            2491200,
            12456000,
            49824000,
            199296000,
            797184000,
            3188736000,
            14947200000,
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

    def _now_decimal_year() -> float:
        now = datetime.utcnow()
        start = datetime(now.year, 1, 1)
        end = datetime(now.year + 1, 1, 1)
        frac = (now - start).total_seconds() / (end - start).total_seconds()
        return float(now.year + frac)

    # Render figure
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    try:
        ax.set_title("Live Horizon Trajectories", fontsize=10)
        # Draw individual trajectories
        alpha = 0.08
        for i in range(num_plot):
            ax.plot(times_arr, cleaned[i], color=(0.2, 0.5, 0.7, alpha), linewidth=1.0)

        # Median
        ax.plot(times_arr, median_traj, color="tab:green", linestyle="--", linewidth=2.0, label="Central Trajectory")

        # Current horizon line (15 min)
        ax.axhline(15.0, color="red", linewidth=2.0, label="Current Horizon (15 min)")

        # Current time vertical line
        ax.axvline(_now_decimal_year(), color="tab:blue", linestyle="--", linewidth=1.75, label="Current Time")

        # Axes formatting
        ax.set_yscale("log")
        tick_values, tick_labels = _get_time_tick_values_and_labels()
        ax.set_yticks(tick_values)
        ax.set_yticklabels(tick_labels)

        finite_max = float(np.nanmax(stacked)) if np.isfinite(np.nanmax(stacked)) else max_work_year_minutes
        ymin = min(tick_values)
        ymax = min(max(finite_max, max(tick_values)), max_work_year_minutes)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Year")
        ax.set_ylabel("Time Horizon")
        ax.grid(True, which="both", axis="y", alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    finally:
        plt.close(fig)


@mc_bp.route("/api/monte-carlo/quarterly-horizon-stats/<job_id>", methods=["GET"])
def quarterly_horizon_stats(job_id: str):
    """Return a JSON table of median and 90% CI of horizon across trajectories
    evaluated at the beginning of each calendar quarter spanned by the rollouts.

    Output: { success: True, rows: [ { quarter: "YYYY-Qn", date: "YYYY-MM-DD",
                                      decimal_year: float, n: int,
                                      median: float, ci5: float, ci95: float }, ... ] }
    Units for horizon statistics are minutes.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"success": False, "error": "Job not found"}), 404
    out_dir = job.get("output_dir")
    if not out_dir:
        return jsonify({"success": False, "error": "Output directory not available yet"}), 404

    rollouts_path = Path(out_dir) / "rollouts.jsonl"
    if not rollouts_path.exists():
        return jsonify({"success": False, "error": "rollouts.jsonl not found"}), 404

    # Read times and per-trajectory horizon arrays
    times_arr: Optional[np.ndarray] = None
    trajectories: List[np.ndarray] = []
    try:
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
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to read rollouts: {e}"}), 500

    if times_arr is None or len(trajectories) == 0:
        return jsonify({"success": True, "rows": []})

    # Clean trajectories similar to live plot
    min_horizon_minutes = 0.001
    max_work_year_minutes = float(120000 * 52 * 40 * 60)
    cleaned: List[np.ndarray] = []
    for t in trajectories:
        arr = t.astype(float)
        arr[~np.isfinite(arr)] = np.nan
        arr[arr <= 0] = np.nan
        arr = np.clip(arr, min_horizon_minutes, max_work_year_minutes)
        cleaned.append(arr)

    # Helper conversions for quarters
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

    # Determine quarter starts spanning the times range
    t_min = float(np.min(times_arr))
    t_max = float(np.max(times_arr))
    dt_min = _decimal_year_to_datetime(t_min)
    dt_max = _decimal_year_to_datetime(t_max)

    def _quarter_start(dt: datetime) -> datetime:
        month = ((dt.month - 1) // 3) * 3 + 1
        return datetime(dt.year, month, 1)

    start_q = _quarter_start(dt_min)
    # If dt_min is not exactly at quarter start and earlier than start_q, ensure start_q >= dt_min
    if start_q < dt_min.replace(day=1, hour=0, minute=0, second=0, microsecond=0):
        start_q = _quarter_start(dt_min)

    quarters: List[datetime] = []
    cur = start_q
    while cur <= dt_max:
        quarters.append(cur)
        # advance by 3 months
        if cur.month <= 9:
            cur = datetime(cur.year, cur.month + 3, 1)
        else:
            cur = datetime(cur.year + 1, 1, 1)

    if not quarters:
        return jsonify({"success": True, "rows": []})

    # Pre-stack for fast column extraction
    stacked = np.vstack(cleaned)  # shape: (num_traj, num_times)

    # For each quarter time, pick nearest index in times_arr
    rows: List[Dict[str, Any]] = []
    for qdt in quarters:
        q_dec = _datetime_to_decimal_year(qdt)
        # nearest index
        idx = int(np.argmin(np.abs(times_arr - q_dec)))
        col = stacked[:, idx]
        # drop NaNs
        finite = col[np.isfinite(col)]
        n = int(finite.size)
        if n == 0:
            median = None
            low = None
            high = None
        else:
            median = float(np.nanmedian(finite))
            # 90% CI: 5th and 95th percentiles
            low = float(np.nanpercentile(finite, 5))
            high = float(np.nanpercentile(finite, 95))
        # Quarter label
        q_num = (qdt.month - 1) // 3 + 1
        rows.append({
            "quarter": f"{qdt.year}-Q{q_num}",
            "date": qdt.date().isoformat(),
            "decimal_year": q_dec,
            "n": n,
            "median": median,
            "ci5": low,
            "ci95": high,
        })

    return jsonify({"success": True, "rows": rows})
