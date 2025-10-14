#!/usr/bin/env python3
"""
Batch rollout script for the AI progress model.

Features:
- Reads subjective distributions over model parameters from a YAML/JSON config
- Samples a large number of parameter sets (default: 5000)
- Runs model rollouts for each sample and stores full metrics
- Writes all artifacts to outputs/<timestamp>/, including:
  * input distribution config
  * snapshot of model_config (raw and JSON)
  * sampled parameters (NDJSON)
  * rollout results (NDJSON), each with parameters and full metrics

Usage:
  python scripts/batch_rollout.py \
    --config config/sampling_config.yaml \
    --num-samples 5000 \
    --input-data input_data.csv \
    --time-range 2015 2050 \
    --seed 123

If --config is omitted, a default independent distribution is derived from
model_config.PARAMETER_BOUNDS (uniform within bounds) and fixed defaults otherwise.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import yaml
import warnings
import contextlib
import io
import importlib
import logging
import signal
import multiprocessing as mp
import scipy.special

try:
    _tqdm = importlib.import_module("tqdm").tqdm  # type: ignore[attr-defined]
except Exception:
    _tqdm = None


# Ensure repository root is on sys.path so we can import project modules
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import model_config as cfg
from progress_model import (
    ProgressModel, Parameters, load_time_series_data
)
from time_series_generator import generate_time_series_from_dict

# Track which parameters we have already warned about for CI-first precedence
_ci_first_warned_params: Set[str] = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch rollout generator for AI progress model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config describing parameter distributions")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of parameter samples to draw and roll out")
    parser.add_argument("--input-data", type=str, default=str(REPO_ROOT / "input_data.csv"), help="Path to input time series CSV")
    parser.add_argument("--time-range", type=float, nargs=2, default=None, help="Start and end time (decimal years) for rollout")
    parser.add_argument("--initial-progress", type=float, default=0.0, help="Initial progress value")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "outputs"), help="Base output directory")
    parser.add_argument("--per-sample-timeout", type=float, default=None, help="Max seconds to spend per rollout (None disables)")
    parser.add_argument("--post-process", action="store_true", help="Automatically run post-processing scripts (plots, sensitivity analysis, etc.)")
    return parser.parse_args()


def _flag_provided(flag: str) -> bool:
    """Return True if a given CLI flag (e.g., "--seed") was provided by the user."""
    return flag in sys.argv


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    text = p.read_text()
    try:
        if p.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(text)
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"Failed to parse config {p}: {e}")


def _snapshot_model_config(output_dir: Path) -> None:
    # 1) write raw file snapshot for reproducibility
    raw_path = output_dir / "model_config_snapshot.py"
    raw_path.write_text((REPO_ROOT / "model_config.py").read_text())

    # 2) write key structures as JSON for convenient reuse
    snapshot: Dict[str, Any] = {}
    keys_to_capture = [
        # constants and dicts
        "RHO_COBB_DOUGLAS_THRESHOLD", "RHO_LEONTIEF_THRESHOLD", "SIGMOID_EXPONENT_CLAMP",
        "AUTOMATION_FRACTION_CLIP_MIN", "RHO_CLIP_MIN", "PARAM_CLIP_MIN",
        "RESEARCH_STOCK_START_MIN",
        "NORMALIZATION_MIN", "experiment_compute_exponent_CLIP_MIN", "experiment_compute_exponent_CLIP_MAX",
        "AGGREGATE_RESEARCH_TASTE_BASELINE", "AGGREGATE_RESEARCH_TASTE_FALLBACK",
        "TOP_PERCENTILE", "MEDIAN_TO_TOP_TASTE_GAP",
        "TASTE_SCHEDULE_TYPES", "DEFAULT_TASTE_SCHEDULE_TYPE",
        "HORIZON_EXTRAPOLATION_TYPES", "DEFAULT_HORIZON_EXTRAPOLATION_TYPE",
        "DEFAULT_present_day", "DEFAULT_present_horizon", "DEFAULT_present_doubling_time",
        "DEFAULT_DOUBLING_DIFFICULTY_GROWTH_FACTOR", "AI_RESEARCH_TASTE_MIN", "AI_RESEARCH_TASTE_MAX",
        "AI_RESEARCH_TASTE_MAX_SD", "BASELINE_ANNUAL_COMPUTE_MULTIPLIER_DEFAULT",
        "BASE_FOR_SOFTWARE_LOM", "MAX_RESEARCH_EFFORT", "MAX_NORMALIZED_PROGRESS_RATE",
        "TIME_EXTRAPOLATION_WINDOW", "PROGRESS_ODE_CLAMP_MAX", "RESEARCH_STOCK_ODE_CLAMP_MAX",
        "ODE_MAX_STEP", "EULER_FALLBACK_MIN_STEPS", "EULER_FALLBACK_STEPS_PER_YEAR",
        "DENSE_OUTPUT_POINTS", "ODE_STEP_SIZE_LOGGING", "ODE_SMALL_STEP_THRESHOLD",
        "ODE_STEP_VARIATION_THRESHOLD", "RELATIVE_ERROR_CLIP", "PARAMETER_BOUNDS",
        "PARAM_VALIDATION_THRESHOLDS", "FEASIBILITY_CHECK_THRESHOLDS", "OBJECTIVE_FUNCTION_CONFIG",
        "OPTIMIZATION_CONFIG", "STRATEGIC_STARTING_POINTS_CONFIG", "DEFAULT_PARAMETERS",
        "PLOT_METADATA", "TAB_CONFIGURATIONS",
    ]
    for k in keys_to_capture:
        if hasattr(cfg, k):
            snapshot[k] = getattr(cfg, k)
    (output_dir / "model_config_snapshot.json").write_text(json.dumps(snapshot, indent=2, default=str))


def _build_default_distribution_config() -> Dict[str, Any]:
    # Use uniform within PARAMETER_BOUNDS where available, else fixed at DEFAULT_PARAMETERS
    param_cfg: Dict[str, Any] = {}
    for name, default_val in cfg.DEFAULT_PARAMETERS.items():
        param_cfg[name] = {"dist": "fixed", "value": default_val}

    # Also include categorical parameters if not already present explicitly
    if "horizon_extrapolation_type" not in param_cfg:
        param_cfg["horizon_extrapolation_type"] = {"dist": "choice", "values": cfg.HORIZON_EXTRAPOLATION_TYPES, "p": None}
    if "taste_schedule_type" not in param_cfg:
        param_cfg["taste_schedule_type"] = {"dist": "choice", "values": cfg.TASTE_SCHEDULE_TYPES, "p": None}

    return {
        "seed": 42,
        "parameters": param_cfg,
        # Additional run settings can be overridden by CLI
        "time_range": None,
        "initial_progress": 0.0,
    }


@contextlib.contextmanager
def _suppress_noise():
    # Silence Python warnings, logging, and stdout/stderr temporarily
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        previous_disable_threshold = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                yield
        finally:
            logging.disable(previous_disable_threshold)


def _sample_from_dist(dist_spec: Dict[str, Any], rng: np.random.Generator, param_name: Optional[str] = None) -> Any:
    """Sample from a distribution using random sampling.
    
    This is a wrapper around _sample_from_dist_with_quantile that generates a random quantile.
    
    Args:
        dist_spec: Distribution specification
        rng: Random number generator
        param_name: Parameter name for error messages and warnings
        
    Returns:
        Sampled value
    """
    # Generate a random quantile and use the quantile-based sampling function
    quantile = rng.uniform(0.0, 1.0)
    return _sample_from_dist_with_quantile(dist_spec, quantile, param_name)


def _sample_from_dist_with_quantile(dist_spec: Dict[str, Any], quantile: float, param_name: Optional[str] = None) -> Any:
    """Sample from a distribution using a specific quantile (for correlated sampling).
    
    Args:
        dist_spec: Distribution specification
        quantile: Quantile value in [0, 1]
        param_name: Parameter name for error messages
        
    Returns:
        Sampled value
    """
    kind = dist_spec.get("dist", "fixed")

    if kind == "fixed":
        return dist_spec.get("value")
    elif kind == "choice":
        pass
    elif (dist_spec.get("min") is None or dist_spec.get("max") is None) and dist_spec.get("clip_to_bounds", True):
        raise ValueError(f"min and max are required for {kind} distribution for parameter {param_name}")

    if kind == "uniform":
        a = float(dist_spec["min"])
        b = float(dist_spec["max"])
        return a + quantile * (b - a)

    if kind == "normal":
        # Support parameterization by mean/sd (or sigma) OR by 80% CI (q10,q90)
        has_pair = "ci80_low" in dist_spec and "ci80_high" in dist_spec
        has_array = "ci80" in dist_spec
        if has_array or has_pair:
            # Warn once per parameter if both CI and mean/sd are provided
            if (
                ("mean" in dist_spec or "sd" in dist_spec or "sigma" in dist_spec)
                and param_name is not None and param_name not in _ci_first_warned_params
            ):
                import warnings as _warnings
                _warnings.warn(
                    f"Parameter '{param_name}': 80% CI (ci80_low/ci80_high) and mean/sd provided; "
                    f"using 80% CI and ignoring mean/sd.",
                )
                _ci_first_warned_params.add(param_name)
            if has_array:
                ci = dist_spec["ci80"]
                q10 = float(ci[0])
                q90 = float(ci[1])
            else:
                q10 = float(dist_spec["ci80_low"])
                q90 = float(dist_spec["ci80_high"])
            if q10 > q90:
                q10, q90 = q90, q10
            z = 1.2815515655446004
            mu = 0.5 * (q10 + q90)
            sigma = (q90 - q10) / (2.0 * z)
        else:
            mu = float(dist_spec["mean"])
            sigma = float(dist_spec["sd"]) if "sd" in dist_spec else float(dist_spec.get("sigma", 1.0))

        # Use inverse normal CDF
        x = mu + sigma * np.sqrt(2) * scipy.special.erfinv(2 * quantile - 1)
        if dist_spec.get("clip_to_bounds") and "min" in dist_spec and "max" in dist_spec:
            x = float(np.clip(x, float(dist_spec["min"]), float(dist_spec["max"])))
        return x

    if kind == "lognormal":
        # Support parameterization by mu/sigma in log-space OR by 80% CI in original space
        has_pair = "ci80_low" in dist_spec and "ci80_high" in dist_spec
        has_array = "ci80" in dist_spec
        if has_array or has_pair:
            # Warn once per parameter if both CI and mu/sigma are provided
            if (
                ("mu" in dist_spec or "sigma" in dist_spec)
                and param_name is not None and param_name not in _ci_first_warned_params
            ):
                import warnings as _warnings
                _warnings.warn(
                    f"Parameter '{param_name}': 80% CI (ci80_low/ci80_high) and mu/sigma provided; "
                    f"using 80% CI and ignoring mu/sigma.",
                )
                _ci_first_warned_params.add(param_name)
            if has_array:
                ci = dist_spec["ci80"]
                q10 = float(ci[0])
                q90 = float(ci[1])
            else:
                q10 = float(dist_spec["ci80_low"])
                q90 = float(dist_spec["ci80_high"])
            if q10 > q90:
                q10, q90 = q90, q10
            z = 1.2815515655446004
            ln_q10 = float(np.log(q10))
            ln_q90 = float(np.log(q90))
            mu = 0.5 * (ln_q10 + ln_q90)
            sigma = (ln_q90 - ln_q10) / (2.0 * z)
        else:
            mu = float(dist_spec["mu"])
            sigma = float(dist_spec["sigma"])

        # Use inverse lognormal CDF
        x = np.exp(mu + sigma * np.sqrt(2) * scipy.special.erfinv(2 * quantile - 1))
        if dist_spec.get("clip_to_bounds") and "min" in dist_spec and "max" in dist_spec:
            x = float(np.clip(x, float(dist_spec["min"]), float(dist_spec["max"])))
        return x

    if kind == "shifted_lognormal":
        # Shifted lognormal: x = shift + LogNormal(mu, sigma)
        has_pair = "ci80_low" in dist_spec and "ci80_high" in dist_spec
        has_array = "ci80" in dist_spec
        if has_array or has_pair:
            # Warn once per parameter if both CI and mu/sigma are provided
            if (
                ("mu" in dist_spec or "sigma" in dist_spec)
                and param_name is not None and param_name not in _ci_first_warned_params
            ):
                import warnings as _warnings
                _warnings.warn(
                    f"Parameter '{param_name}': 80% CI (ci80_low/ci80_high) and mu/sigma provided; "
                    f"using 80% CI and ignoring mu/sigma.",
                )
                _ci_first_warned_params.add(param_name)
            if has_array:
                ci = dist_spec["ci80"]
                q10 = float(ci[0])
                q90 = float(ci[1])
            else:
                q10 = float(dist_spec["ci80_low"])
                q90 = float(dist_spec["ci80_high"])
            if q10 > q90:
                q10, q90 = q90, q10
            z = 1.2815515655446004
            ln_q10 = float(np.log(q10))
            ln_q90 = float(np.log(q90))
            mu = 0.5 * (ln_q10 + ln_q90)
            sigma = (ln_q90 - ln_q10) / (2.0 * z)
        else:
            mu = float(dist_spec["mu"])
            sigma = float(dist_spec["sigma"])

        # Use inverse shifted lognormal CDF
        x_core = np.exp(mu + sigma * np.sqrt(2) * scipy.special.erfinv(2 * quantile - 1))
        shift = dist_spec.get("shift")
        if shift is None:
            raise ValueError(f"shift for shifted_lognormal is not set for parameter {param_name}")
        shift = float(shift)
        x = float(shift + x_core)
        if dist_spec.get("clip_to_bounds") and "min" in dist_spec and "max" in dist_spec:
            x = float(np.clip(x, float(dist_spec["min"]), float(dist_spec["max"])))
        return x

    if kind == "beta":
        a = float(dist_spec["alpha"])
        b = float(dist_spec["beta"])
        lo = float(dist_spec.get("min", 0.0))
        hi = float(dist_spec.get("max", 1.0))
        # Use inverse beta CDF
        x01 = scipy.special.betaincinv(a, b, quantile)
        return lo + (hi - lo) * x01

    if kind == "choice":
        values = dist_spec["values"]
        p = dist_spec.get("p")
        if p is None:
            # Uniform choice
            idx = int(quantile * len(values))
            idx = min(idx, len(values) - 1)
            return values[idx]
        else:
            # Weighted choice - find cumulative sum
            cumsum = np.cumsum(p)
            idx = np.searchsorted(cumsum, quantile)
            idx = min(idx, len(values) - 1)
            return values[idx]

    raise ValueError(f"Unknown distribution kind: {kind}")


def _clip_to_param_bounds(param_name: str, value: Any) -> Any:
    bounds = cfg.PARAMETER_BOUNDS.get(param_name)
    if value is None:
        assert False, f"value is None for parameter {param_name}"
    if bounds is None:
        return value
    lo, hi = bounds
    return float(np.clip(float(value), float(lo), float(hi)))


# =======================
# Correlation calibration
# =======================

def _gaussian_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2.0)))


def _quantile_transform_vectorized(dist_spec: Dict[str, Any], u: np.ndarray, param_name: Optional[str] = None) -> np.ndarray:
    # Thin wrapper to reuse the scalar quantile sampler in vectorized fashion
    # Using np.vectorize keeps code simple; performance is adequate for calibration sizes (~1e4-5e4)
    vec = np.vectorize(lambda q: _sample_from_dist_with_quantile(dist_spec, float(q), param_name), otypes=[float])
    return vec(u)


def _estimate_pearson_for_rho(
    spec_i: Dict[str, Any],
    spec_j: Dict[str, Any],
    rho_z: float,
    base_normals: Tuple[np.ndarray, np.ndarray],
    param_i_name: str,
    param_j_name: str,
) -> float:
    # Build correlated normals from shared base normals (A,B) for reproducibility across rho values
    A, B = base_normals
    # Guard against numerical issues near |rho|=1
    rho = float(np.clip(rho_z, -0.9999, 0.9999))
    Z1 = A
    Z2 = rho * A + np.sqrt(max(0.0, 1.0 - rho * rho)) * B
    U1 = _gaussian_cdf(Z1)
    U2 = _gaussian_cdf(Z2)
    X1 = _quantile_transform_vectorized(spec_i, U1, param_i_name)
    X2 = _quantile_transform_vectorized(spec_j, U2, param_j_name)
    c = np.corrcoef(X1, X2)[0, 1]
    if not np.isfinite(c):
        return 0.0
    return float(c)


def _solve_latent_rho_for_pair(
    spec_i: Dict[str, Any],
    spec_j: Dict[str, Any],
    target_corr: float,
    base_normals: Tuple[np.ndarray, np.ndarray],
    param_i_name: str,
    param_j_name: str,
    tol: float = 1e-2,
    max_iter: int = 40,
) -> float:
    # Bisection on rho_z in [-0.999, 0.999] to achieve target Pearson correlation after transforms
    lo, hi = -0.999, 0.999
    f_lo = _estimate_pearson_for_rho(spec_i, spec_j, lo, base_normals, param_i_name, param_j_name)
    f_hi = _estimate_pearson_for_rho(spec_i, spec_j, hi, base_normals, param_i_name, param_j_name)

    # If target is outside achievable range, clamp to nearest endpoint
    if target_corr <= min(f_lo, f_hi):
        return lo if f_lo <= f_hi else hi
    if target_corr >= max(f_lo, f_hi):
        return lo if f_lo >= f_hi else hi

    # Ensure f is increasing with rho for bisection; if not, swap ends
    increasing = f_lo < f_hi
    left, right = (lo, hi) if increasing else (hi, lo)
    f_left = f_lo if increasing else f_hi
    f_right = f_hi if increasing else f_lo

    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = _estimate_pearson_for_rho(spec_i, spec_j, mid, base_normals, param_i_name, param_j_name)
        if abs(f_mid - target_corr) <= tol:
            return float(mid)
        if f_mid < target_corr:
            left, f_left = mid, f_mid
        else:
            right, f_right = mid, f_mid

    return float(0.5 * (left + right))


def _nearest_correlation_matrix(A: np.ndarray, tol: float = 1e-8, max_iters: int = 100) -> np.ndarray:
    # Higham's nearest correlation matrix algorithm (simplified)
    np.linalg.cholesky(A)
    return A
    Y = A.copy()
    np.fill_diagonal(Y, 1.0)
    delta_S = np.zeros_like(Y)
    for _ in range(max_iters):
        R = Y - delta_S
        # PSD projection
        eigvals, eigvecs = np.linalg.eigh((R + R.T) * 0.5)
        eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
        X = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        delta_S = X - R
        Y = X
        # Unit diagonal and symmetry
        np.fill_diagonal(Y, 1.0)
        Y = (Y + Y.T) * 0.5
        # Convergence check (diagonal already unit; check min eigenvalue)
        if np.min(eigvals_clipped) >= -tol:
            break
    # Ensure strict PD by jitter if needed
    try:
        np.linalg.cholesky(Y)
    except np.linalg.LinAlgError:
        # Add a small jitter
        jitter = 1e-8
        for _ in range(5):
            try:
                Y_j = Y + np.eye(Y.shape[0]) * jitter
                np.fill_diagonal(Y_j, 1.0)
                np.linalg.cholesky(Y_j)
                Y = Y_j
                break
            except np.linalg.LinAlgError:
                jitter *= 10
    return Y


def _calibrate_gaussian_copula_latent_corr(
    param_dists: Dict[str, Any],
    correlated_params: List[str],
    target_corr: np.ndarray,
    rng: np.random.Generator,
    interpretation: str = "spearman",
    calibration_samples: int = 20000,
    tol: float = 1e-2,
) -> np.ndarray:
    n = len(correlated_params)
    if n <= 1:
        return np.eye(n)

    # If interpreting as Spearman rank correlation, convert to latent Gaussian Pearson via closed form
    if interpretation.lower() in {"spearman", "rank", "spearman_rho"}:
        # For a Gaussian copula: rho_S = (6/pi) * arcsin(rho/2)  =>  rho = 2 * sin(pi * rho_S / 6)
        rho_latent = 2.0 * np.sin(np.pi * target_corr / 6.0)
        rho_latent = np.clip(rho_latent, -0.9999, 0.9999)
        np.fill_diagonal(rho_latent, 1.0)
        return _nearest_correlation_matrix(rho_latent)

    # Otherwise, interpret target_corr as Pearson in original space; use NORTA pairwise calibration
    # Pre-generate base normals shared across all pairs and iterations for reproducibility and monotonic mapping
    A = rng.standard_normal(calibration_samples)
    B = rng.standard_normal(calibration_samples)
    base_normals = (A, B)

    rho_latent = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            name_i = correlated_params[i]
            name_j = correlated_params[j]
            spec_i = param_dists[name_i]
            spec_j = param_dists[name_j]
            target = float(target_corr[i, j])
            # Solve latent rho
            rho_ij = _solve_latent_rho_for_pair(spec_i, spec_j, target, base_normals, name_i, name_j, tol=tol)
            rho_latent[i, j] = rho_ij
            rho_latent[j, i] = rho_ij

    # Project to nearest valid correlation matrix
    rho_latent = _nearest_correlation_matrix(rho_latent)
    return rho_latent


def _sample_parameter_dict(param_dists: Dict[str, Any], rng: np.random.Generator, correlation_matrix: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Sample parameters, optionally applying correlations between specified parameters.
    
    Args:
        param_dists: Parameter distribution specifications
        rng: Random number generator
        correlation_matrix: Optional correlation specification with format:
            {
                "parameters": ["param1", "param2", ...],  # List of parameter names
                "correlation_matrix": [[1.0, 0.5, ...], [0.5, 1.0, ...], ...]  # Correlation matrix
            }
    """
    sampled: Dict[str, Any] = {}
    
    # Handle correlated parameters if specified
    if correlation_matrix is not None:
        correlated_params = correlation_matrix.get("parameters", [])
        corr_matrix = correlation_matrix.get("correlation_matrix")
        if correlated_params and corr_matrix is not None:
            # Validate correlation matrix
            n_params = len(correlated_params)
            if len(corr_matrix) != n_params:
                raise ValueError(f"Correlation matrix must be {n_params}x{n_params} for {n_params} parameters")
            for i, row in enumerate(corr_matrix):
                if len(row) != n_params:
                    raise ValueError(f"Correlation matrix row {i} must have {n_params} elements")
            
            # Convert to numpy array and validate
            # Prefer a pre-calibrated latent correlation matrix if provided
            latent_corr = correlation_matrix.get("latent_correlation_matrix")
            corr_array = np.array(latent_corr if latent_corr is not None else corr_matrix, dtype=float)
            assert latent_corr is not None
            if not np.allclose(corr_array, corr_array.T):
                raise ValueError("Correlation matrix must be symmetric")
            if not np.allclose(np.diag(corr_array), 1.0):
                raise ValueError("Correlation matrix diagonal must be 1.0")
            
            # Check if correlation matrix is positive semi-definite
            try:
                np.linalg.cholesky(corr_array)
            except np.linalg.LinAlgError:
                raise ValueError("Correlation matrix must be positive semi-definite")
            
            # Sample correlated parameters via Gaussian copula
            # Generate multivariate normal samples
            mv_samples = rng.multivariate_normal(np.zeros(n_params), corr_array)
            
            # Convert to uniform marginals using normal CDF
            uniform_samples = np.array([0.5 * (1 + scipy.special.erf(x / np.sqrt(2))) for x in mv_samples])
            
            # Sample each correlated parameter using its distribution with the uniform quantile
            for i, param_name in enumerate(correlated_params):
                if param_name not in param_dists:
                    raise ValueError(f"Correlated parameter '{param_name}' not found in parameter distributions")
                
                spec = param_dists[param_name]
                val = _sample_from_dist_with_quantile(spec, uniform_samples[i], param_name)
                
                # Clip to bounds unless explicitly disabled
                clip_requested = spec.get("clip_to_bounds", True)
                if clip_requested:
                    val = _clip_to_param_bounds(param_name, val)
                
                sampled[param_name] = val
    
    # Sample remaining independent parameters
    for name, spec in param_dists.items():
        if name == "automation_anchors":
            continue
        if name in sampled:  # Skip if already sampled as part of correlated group
            continue
            
        val = _sample_from_dist(spec, rng, name)

        # Clip to bounds unless explicitly disabled, when bounds exist
        clip_requested = spec.get("clip_to_bounds", True)
        if clip_requested:
            val = _clip_to_param_bounds(name, val)

        sampled[name] = val

    return sampled


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _write_ndjson(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str))
            f.write("\n")


def _to_jsonable(obj: Any) -> Any:
    # Convert nested structures and numpy types to JSON-friendly forms
    try:
        import numpy as _np
    except Exception:
        _np = None

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        # NaNs/inf handled below
        if isinstance(obj, float):
            if not np.isfinite(obj):
                return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if _np is not None:
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, _np.generic):
            val = obj.item()
            if isinstance(val, float) and not np.isfinite(val):
                return None
            return val
    # Fallback to string
    return str(obj)


def _rollout_worker(conn, sampled_params: Dict[str, Any], sampled_ts_params: Dict[str, Any], input_data_path: str, time_range: List[float], initial_progress: float) -> None:
    """Subprocess worker: runs one rollout and sends back results or error."""
    try:
        # Lazy imports in subprocess to avoid parent state issues
        from progress_model import ProgressModel as _PM, Parameters as _Params, load_time_series_data as _load
        from time_series_generator import generate_time_series_from_dict as _gen_ts

        # Load base time series data
        base_data = _load(input_data_path)

        # Check if time series parameters are present; if so, generate new time series
        if sampled_ts_params:
            # Generate time series with uncertainty
            data = _gen_ts(sampled_ts_params, input_data_path, base_data)
        else:
            # Use base data as-is
            data = base_data

        params_obj = _Params(**sampled_params)
        model = _PM(params_obj, data)
        with _suppress_noise():
            model.compute_progress_trajectory(time_range, initial_progress)
        # Include computed r_software in results
        conn.send({"ok": True, "results": _to_jsonable(model.results), "r_software": float(model.params.r_software)})
    except Exception as e:  # pragma: no cover
        import traceback
        conn.send({"ok": False, "error": str(e), "traceback": traceback.format_exc()})
    finally:
        try:
            conn.close()
        except Exception as e2:
            print(f"Error closing connection: {e2}")
            pass


def _run_rollout_subprocess(sampled_params: Dict[str, Any], sampled_ts_params: Dict[str, Any], input_data_path: str, time_range: List[float], initial_progress: float, timeout_s: Optional[float]) -> Dict[str, Any]:
    """Run one rollout in a child process; enforce timeout by termination.

    Returns dict with 'results' and 'r_software' on success. Raises TimeoutError on timeout. Raises Exception on failure.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_rollout_worker, args=(child_conn, sampled_params, sampled_ts_params, input_data_path, time_range, initial_progress))
    proc.daemon = False
    proc.start()
    child_conn.close()  # close in parent
    try:
        if timeout_s is None or timeout_s <= 0:
            # Wait indefinitely
            result = parent_conn.recv()
        else:
            if parent_conn.poll(timeout_s):
                result = parent_conn.recv()
            else:
                # Timeout: kill child
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                raise TimeoutError(f"Rollout timed out after {timeout_s} seconds")
        # Ensure the child is fully reaped (avoid zombies)
        try:
            proc.join(timeout=1)
        except Exception as e:
            print(f"Error joining process: {e}")
            pass
        if not isinstance(result, dict) or not result.get("ok"):
            err = None if not isinstance(result, dict) else result.get("error")
            traceback = None if not isinstance(result, dict) else result.get("traceback")
            error_msg = err or "Unknown rollout failure"
            if traceback:
                error_msg += f"\nSubprocess traceback:\n{traceback}"
            raise RuntimeError(error_msg)
        # Return both results and r_software
        return {"results": result["results"], "r_software": result.get("r_software")}
    finally:
        try:
            parent_conn.close()
        except Exception as e:
            print(f"Error closing parent connection: {e}")
            pass
        try:
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception as e:
                    print(f"Error terminating process: {e}")
                    pass
            proc.join(timeout=1)
        except Exception as e:
            print(f"Error joining process: {e}")
            pass

def _run_with_timeout(seconds: Optional[float], func, *args, **kwargs):
    """Run func(*args, **kwargs) with a wall-clock timeout in seconds using SIGALRM.

    If seconds is None or <= 0, runs without a timeout. Raises TimeoutError on expiry.
    """
    if seconds is None or seconds <= 0:
        return func(*args, **kwargs)

    def _alarm_handler(signum, frame):
        raise TimeoutError(f"Rollout timed out after {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        # setitimer supports sub-second precision
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        return func(*args, **kwargs)
    finally:
        # Always cancel timer and restore previous handler
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def main() -> None:
    args = parse_args()

    # Load distribution config or build default
    user_cfg = _load_config(args.config)
    if not user_cfg:
        user_cfg = _build_default_distribution_config()

    # Resolve effective run settings (CLI overrides config)
    # Seed
    if _flag_provided("--seed"):
        seed = int(args.seed)
    else:
        seed = int(user_cfg.get("seed", 42))

    # Number of samples
    if _flag_provided("--num-samples"):
        num_samples = int(args.num_samples)
    else:
        num_samples = int(user_cfg.get("num_samples", user_cfg.get("num_rollouts", args.num_samples)))

    # Input data path
    if _flag_provided("--input-data"):
        input_data_path = str(args.input_data)
    else:
        input_data_path = str(user_cfg.get("input_data", args.input_data))

    # Initial progress
    if _flag_provided("--initial-progress"):
        initial_progress = float(args.initial_progress)
    else:
        initial_progress = float(user_cfg.get("initial_progress", 0.0))

    # Per-sample timeout (seconds)
    if _flag_provided("--per-sample-timeout"):
        per_sample_timeout = float(args.per_sample_timeout) if args.per_sample_timeout is not None else None
    else:
        _cfg_timeout = user_cfg.get("per_sample_timeout")
        per_sample_timeout = float(_cfg_timeout) if _cfg_timeout is not None else None

    # Prepare output directories
    rng = np.random.default_rng(seed)
    base_out = Path(args.output_dir)
    _ensure_dir(base_out)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_out / timestamp
    _ensure_dir(run_dir)

    # Load time series data once
    data = load_time_series_data(input_data_path)

    # Determine time range
    if _flag_provided("--time-range") and args.time_range is not None:
        time_range = [float(args.time_range[0]), float(args.time_range[1])]
    else:
        cfg_time_range = user_cfg.get("time_range")
        if cfg_time_range is not None:
            time_range = [float(cfg_time_range[0]), float(cfg_time_range[1])]
        else:
            # Default to full available range in input_data.csv
            time_range = [float(np.min(data.time)), float(np.max(data.time))]

    # Persist input config for reproducibility
    # Update user_cfg snapshot to reflect effective settings
    user_cfg["seed"] = seed
    user_cfg["initial_progress"] = initial_progress
    user_cfg["time_range"] = time_range
    user_cfg["num_samples"] = num_samples
    user_cfg["input_data"] = str(Path(input_data_path).resolve())
    user_cfg["per_sample_timeout"] = per_sample_timeout
    input_cfg_path = run_dir / "input_distributions.yaml"
    input_cfg_path.write_text(yaml.safe_dump(user_cfg))

    # Snapshot model_config
    _snapshot_model_config(run_dir)

    # Prepare output files for samples and rollouts
    samples_out_path = run_dir / "samples.jsonl"
    rollouts_out_path = run_dir / "rollouts.jsonl"

    # Avoid holding everything in memory: stream as NDJSON
    param_dists: Dict[str, Any] = user_cfg.get("parameters", {})
    ts_param_dists: Dict[str, Any] = user_cfg.get("time_series_parameters", {})
    correlation_matrix: Optional[Dict[str, Any]] = user_cfg.get("correlation_matrix")

    # print(param_dists["saturation_horizon_minutes"])

    # Write minimal metadata
    _write_json(run_dir / "metadata.json", {
        "created_at": datetime.now().isoformat(),
        "num_samples": int(num_samples),
        "seed": int(seed),
        "input_data_path": str(Path(input_data_path).resolve()),
        "time_range": time_range,
        "per_sample_timeout": per_sample_timeout,
    })

    # Precompute latent correlation (Gaussian space) once, if correlations are configured
    if correlation_matrix is not None:
        try:
            correlated_params = correlation_matrix.get("parameters", [])
            target_corr = correlation_matrix.get("correlation_matrix")
            if correlated_params and target_corr is not None:
                # Allow specifying interpretation of provided correlations: 'pearson' (default) or 'spearman'
                interpretation = str(correlation_matrix.get("correlation_type", correlation_matrix.get("interpretation", "spearman")))
                calib_samples = int(correlation_matrix.get("calibration_samples", 20000))
                # Use a derived RNG for calibration to keep overall reproducibility
                calib_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
                latent = _calibrate_gaussian_copula_latent_corr(
                    param_dists,
                    correlated_params,
                    np.array(target_corr, dtype=float),
                    calib_rng,
                    interpretation=interpretation,
                    calibration_samples=calib_samples,
                )
                correlation_matrix["latent_correlation_matrix"] = latent.tolist()
        except Exception as e:
            # If calibration fails, fall back to user-provided matrix; sampling will validate PSD
            print(f"WARNING: correlation calibration failed; using provided correlation_matrix. Error: {e}")

    # Merge parameter distributions to enable cross-group correlations
    # All parameters are sampled together, then split back into their respective groups
    all_param_dists = {}
    all_param_dists.update(param_dists)
    all_param_dists.update(ts_param_dists)
    
    # Track which parameters belong to which group for later splitting
    model_param_names = set(param_dists.keys())
    ts_param_names = set(ts_param_dists.keys())

    # Sampling and rollouts
    with samples_out_path.open("w", encoding="utf-8") as f_samples, rollouts_out_path.open("w", encoding="utf-8") as f_rollouts:
        iterable = range(int(num_samples))
        progress_iter = _tqdm(iterable, desc="Rollouts", unit="run") if _tqdm is not None else iterable
        for i in progress_iter:
            try:
                # Sample all parameters together (enables correlations across parameter groups)
                all_sampled = _sample_parameter_dict(all_param_dists, rng, correlation_matrix)
                
                # Split sampled parameters back into model and time series groups
                sampled_params = {k: v for k, v in all_sampled.items() if k in model_param_names}
                sampled_ts_params = {k: v for k, v in all_sampled.items() if k in ts_param_names}
                
                # Make sure categorical defaults exist if user omitted
                sampled_params.setdefault("horizon_extrapolation_type", cfg.DEFAULT_HORIZON_EXTRAPOLATION_TYPE)
                sampled_params.setdefault("taste_schedule_type", cfg.DEFAULT_TASTE_SCHEDULE_TYPE)

                # Generate time series with uncertainty if time series parameters are present
                if sampled_ts_params:
                    rollout_data = generate_time_series_from_dict(sampled_ts_params, input_data_path, data)
                else:
                    # Use base data as-is
                    rollout_data = data

                # Prepare constructor kwargs
                params_obj = Parameters(**sampled_params)

                # Suppress stdout/stderr, warnings, and logging during simulation
                # with _suppress_noise():
                if per_sample_timeout is not None and per_sample_timeout > 0:
                    # Use subprocess isolation when timeout is requested to avoid internal catches
                    subprocess_result = _run_rollout_subprocess(sampled_params, sampled_ts_params, str(Path(input_data_path).resolve()), time_range, initial_progress, per_sample_timeout)
                    # Extract r_software if available from subprocess
                    if isinstance(subprocess_result, dict) and "r_software" in subprocess_result:
                        sampled_params["r_software"] = subprocess_result["r_software"]
                        rollout_results = subprocess_result.get("results")
                    else:
                        rollout_results = subprocess_result
                    model = None  # no in-process model
                else:
                    model = ProgressModel(params_obj, rollout_data)
                    times, progress_values, research_stock_values = model.compute_progress_trajectory(time_range, initial_progress)
                    # Add computed r_software to sampled_params for future analysis
                    sampled_params["r_software"] = model.params.r_software
                    rollout_results = _to_jsonable(model.results)

                # Persist sample record (include both parameter types)
                sample_record = {
                    "sample_id": i,
                    "parameters": _to_jsonable(sampled_params),
                    "time_series_parameters": _to_jsonable(sampled_ts_params),
                }
                f_samples.write(json.dumps(sample_record) + "\n")
                f_samples.flush()

                # Persist rollout record
                rollout_record = {
                    "sample_id": i,
                    "parameters": _to_jsonable(sampled_params),
                    "time_series_parameters": _to_jsonable(sampled_ts_params),
                    "results": rollout_results,
                }
                f_rollouts.write(json.dumps(rollout_record) + "\n")
                f_rollouts.flush()

            except TimeoutError as e:
                # Persist timeout info to keep alignment between files
                f_samples.write(json.dumps({"sample_id": i, "parameters": _to_jsonable(sampled_params), "time_series_parameters": _to_jsonable(sampled_ts_params), "error": str(e)}) + "\n")
                f_samples.flush()
                f_rollouts.write(json.dumps({"sample_id": i, "parameters": _to_jsonable(sampled_params), "time_series_parameters": _to_jsonable(sampled_ts_params), "results": None, "error": str(e)}) + "\n")
                f_rollouts.flush()
            except Exception as e:
                import traceback
                # Persist failure info to keep alignment between files
                print(f"Rollout failed: {e}")
                print(traceback.format_exc())
                f_samples.write(json.dumps({"sample_id": i, "parameters": None, "time_series_parameters": None, "error": str(e), "traceback": traceback.format_exc()}) + "\n")
                f_samples.flush()
                f_rollouts.write(json.dumps({"sample_id": i, "parameters": None, "time_series_parameters": None, "results": None, "error": str(e), "traceback": traceback.format_exc()}) + "\n")
                f_rollouts.flush()
            # Manual progress update when tqdm is unavailable
            if _tqdm is None:
                completed = i + 1
                total = int(num_samples)
                pct = int(100 * completed / total)
                print(f"\rRollouts: {completed}/{total} ({pct}%)", end="", flush=True)

        # Ensure newline after manual progress output
        if _tqdm is None:
            print()

    print(f"Run complete. Artifacts in: {run_dir}")

    # Always generate batch plots
    python_exe = sys.executable
    try:
        print("\nGenerating batch plots (milestone histograms and transitions)...")
        subprocess.run(
            [python_exe, str(SCRIPTS_DIR / "plot_rollouts.py"), "--run-dir", str(run_dir), "--batch-all"],
            cwd=str(REPO_ROOT),
            check=True
        )
        print("Batch plots complete!")
    except Exception as e:
        print(f"WARNING: batch plot_rollouts failed: {e}")

    # Run post-processing scripts if requested
    if args.post_process:
        print("\nRunning post-processing scripts...")
        python_exe = sys.executable

        # 1. ACD-AI time histogram
        try:
            print("  - Generating ACD-AI time histogram...")
            subprocess.run(
                [python_exe, str(SCRIPTS_DIR / "plot_rollouts.py"), "--run-dir", str(run_dir)],
                cwd=str(REPO_ROOT),
                check=True
            )
        except Exception as e:
            print(f"    WARNING: plot_rollouts failed: {e}")

        # 2. Horizon trajectories plot
        try:
            print("  - Generating horizon trajectories plot...")
            subprocess.run(
                [python_exe, str(SCRIPTS_DIR / "plot_rollouts.py"), "--run-dir", str(run_dir), "--mode", "horizon_trajectories"],
                cwd=str(REPO_ROOT),
                check=True
            )
        except Exception as e:
            print(f"    WARNING: plot_rollouts (horizon_trajectories) failed: {e}")

        # 3. Sensitivity analysis
        try:
            print("  - Running sensitivity analysis...")
            subprocess.run(
                [python_exe, str(SCRIPTS_DIR / "sensitivity_analysis.py"), "--run-dir", str(run_dir), "--plot"],
                cwd=str(REPO_ROOT),
                check=True
            )
        except Exception as e:
            print(f"    WARNING: sensitivity_analysis failed: {e}")

        # 4. SC by quarter table
        try:
            print("  - Generating SC-by-quarter table...")
            subprocess.run(
                [python_exe, str(SCRIPTS_DIR / "sc_by_quarter.py"), "--run-dir", str(run_dir)],
                cwd=str(REPO_ROOT),
                check=True
            )
        except Exception as e:
            print(f"    WARNING: sc_by_quarter failed: {e}")

        print("Post-processing complete!")


if __name__ == "__main__":
    main()

