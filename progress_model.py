#!/usr/bin/env python3
"""
Progress Modeling Script

Models AI progress over time using nested CES production functions with
feedback loops between automation fraction and cumulative progress.
"""

# Set matplotlib backend before any imports to prevent GUI errors on headless servers
import matplotlib
matplotlib.use('Agg')

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union, NamedTuple
from scipy import optimize, integrate, interpolate
import logging
import time
import model_config as cfg
import yaml
import copy
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesData:
    """Input time series data"""
    time: np.ndarray  # Decimal years
    L_HUMAN: np.ndarray  # Human labor supply
    inference_compute: np.ndarray  # AI labor supply (human-equivalents)
    experiment_compute: np.ndarray  # Experiment compute budget
    training_compute_growth_rate: np.ndarray  # Training compute budget


class TasteDistribution:
    """
    Manages human research taste distribution and provides methods for working
    with taste values in terms of quantiles and standard deviations.
    
    The distribution is modeled as T ~ LogNormal(μ, σ²), where the parameters
    are derived from empirical anchors:
    - top_percentile: fraction of researchers classified as "top"
    - median_to_top_gap: ratio of threshold taste to median taste  
    - baseline_mean: company-wide mean taste
    
    Example usage:
        # Create distribution with default parameters
        taste_dist = TasteDistribution()
        
        # Get taste at 90th percentile
        top_taste = taste_dist.get_taste_at_quantile(0.9)
        
        # Get taste at 2 standard deviations above mean in normal space
        ai_taste = taste_dist.get_taste_at_sd(2.0)
        
        # Compute aggregate taste with AI floor growing at 0.5 SD per progress unit
        progress = 10.0
        ai_research_taste = taste_dist.get_taste_at_sd(0.5 * progress)
        aggregate_taste = taste_dist.get_mean_with_floor(ai_research_taste)
    """
    
    def __init__(self, 
                 top_percentile: float = cfg.TOP_PERCENTILE,
                 median_to_top_gap: float = cfg.MEDIAN_TO_TOP_TASTE_MULTIPLIER,
                 baseline_mean: float = cfg.AGGREGATE_RESEARCH_TASTE_BASELINE):
        """
        Initialize the taste distribution with empirical anchors.
        
        Args:
            top_percentile: Fraction of researchers classified as "top"
            median_to_top_gap: Ratio of threshold taste to median taste
            baseline_mean: Company-wide mean taste
        """
        from scipy.stats import norm
        import math
        
        # Store parameters
        self.top_percentile = top_percentile
        self.median_to_top_gap = median_to_top_gap
        self.baseline_mean = baseline_mean
        
        # Validate parameters
        if not (0 < top_percentile < 1):
            raise ValueError(f"top_percentile must be between 0 and 1, got {top_percentile}")
        if median_to_top_gap <= 1:
            raise ValueError(f"median_to_top_gap must be > 1, got {median_to_top_gap}")
        if baseline_mean <= 0:
            raise ValueError(f"baseline_mean must be > 0, got {baseline_mean}")
        
        # Compute log-normal distribution parameters
        z_p = norm.ppf(1 - top_percentile)
        self.sigma = math.log(median_to_top_gap) / z_p
        self.mu = math.log(baseline_mean) - 0.5 * self.sigma ** 2
        
        logger.debug(f"TasteDistribution initialized: μ={self.mu:.4f}, σ={self.sigma:.4f}")
    
    def get_taste_at_quantile(self, quantile: float) -> float:
        """
        Get the taste value at a given quantile of the distribution.
        
        Args:
            quantile: Quantile (0 to 1)
            
        Returns:
            Taste value at the specified quantile
        """
        from scipy.stats import norm
        import math
        
        if not (0 <= quantile <= 1):
            raise ValueError(f"quantile must be between 0 and 1, got {quantile}")
        
        if quantile == 0:
            return 0.0
        if quantile == 1:
            return float('inf')
        
        # For LogNormal(μ, σ²), quantile function is exp(μ + σ * Φ^(-1)(quantile))
        return math.exp(self.mu + self.sigma * norm.ppf(quantile))
    
    def get_quantile_of_taste(self, taste: float) -> float:
        """
        Get the quantile of a given taste value in the distribution.
        
        Args:
            taste: Taste value
            
        Returns:
            Quantile (0 to 1) of the taste value
        """
        from scipy.stats import norm
        import math
        
        if taste <= 0:
            return 0.0
        
        # For LogNormal(μ, σ²), if X = taste, then Φ((ln(X) - μ)/σ) gives the quantile
        return norm.cdf((math.log(taste) - self.mu) / self.sigma)
    
    def get_taste_at_sd(self, num_sds: float) -> float:
        """
        Get taste value at a given number of standard deviations in the underlying normal distribution.
        
        Since T ~ LogNormal(μ, σ²), we have ln(T) ~ Normal(μ, σ²).
        This method returns exp(μ + num_sds * σ).
        
        Args:
            num_sds: Number of standard deviations (can be negative)
            
        Returns:
            Taste value at the specified standard deviation
        """
        import math
        return math.exp(self.mu + num_sds * self.sigma)
    
    def get_sd_of_taste(self, taste: float) -> float:
        """
        Get how many standard deviations a taste value is in the underlying normal distribution.
        
        Args:
            taste: Taste value
            
        Returns:
            Number of standard deviations (can be negative)
        """
        import math
        
        if taste <= 0:
            return float('-inf')
        
        return (math.log(taste) - self.mu) / self.sigma
    
    def get_mean_with_floor(self, floor_taste: float) -> float:
        """
        Compute the mean of the distribution with a floor applied using clip-and-keep logic.
        
        This implements the closed-form expectation for the clipped distribution:
        E[max(T, F)] = F·Φ(a) + exp(μ + σ²/2) · Φ(σ − a),  a = (ln F − μ)/σ
        where T ~ LogNormal(μ, σ²) and F is the floor value.
        
        Args:
            floor_taste: Floor value (any draw below this is lifted to this value)
            
        Returns:
            Mean taste after applying the floor
        """
        from scipy.stats import norm
        import math
        
        # Input validation
        if not np.isfinite(floor_taste):
            logger.warning(f"Non-finite floor_taste: {floor_taste}")
            return cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK
        
        if floor_taste < 0:
            logger.warning(f"Negative floor_taste: {floor_taste}, using 0")
            floor_taste = 0.0
        
        try:
            # Return the unconditional mean if floor ≤ 0 (no clipping needed)
            if floor_taste <= 0:
                return math.exp(self.mu + 0.5 * self.sigma ** 2)
            
            # --- Clip-and-keep expectation -----------------------------------
            try:
                F = floor_taste  # Floor value
                a = (math.log(F) - self.mu) / self.sigma
                # Upper tail unchanged
                upper = math.exp(self.mu + 0.5 * self.sigma ** 2) * norm.cdf(self.sigma - a)
                # Lower tail lifted to the floor
                lower = F * norm.cdf(a)
                clipped_mean = upper + lower
            except (ValueError, OverflowError) as e:
                logger.warning(f"Numerical error in clipped-mean calculation: {e}")
                return max(floor_taste, cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK)
            
            # Validate result
            if not np.isfinite(clipped_mean):
                logger.warning(f"Invalid clipped mean: {clipped_mean}, returning fallback")
                return max(floor_taste, cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK)
            
            return clipped_mean
        
        except Exception as e:
            logger.warning(f"Error computing mean with floor: {e}")
            # Fallback: return max of floor and baseline
            return max(floor_taste, cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK)
    
    def get_median(self) -> float:
        """Get the median of the distribution."""
        return self.get_taste_at_quantile(0.5)
    
    def get_mean(self) -> float:
        """Get the unconditional mean of the distribution."""
        return self.baseline_mean
    
    def __repr__(self) -> str:
        return (f"TasteDistribution(top_percentile={self.top_percentile:.3f}, "
                f"median_to_top_gap={self.median_to_top_gap:.2f}, "
                f"baseline_mean={self.baseline_mean:.2f})")

class AutomationModel:
    """Automation model"""
    def __init__(self, params):
        self.initial_FTE_per_GPU = 1
        self.FTE_per_GPU_slope = 1.0
        self.progress_base_unit = cfg.BASE_FOR_SOFTWARE_LOM
        self.schedule_type = getattr(params, 'automation_interp_type', cfg.DEFAULT_PARAMETERS['automation_interp_type'])
        anchors = list(params.automation_anchors.items())
        anchors.sort(key=lambda x: x[0])
        self.anchor_points = anchors
        (prog_1, aut_1), (prog_2, aut_2) = self.anchor_points
        self.linear_aut_slope = (aut_2 - aut_1) / (prog_2 - prog_1)
        self.linear_aut_intercept = aut_1 - self.linear_aut_slope * prog_1
        self.exponential_aut_slope = (np.log(aut_2) - np.log(aut_1)) / (prog_2 - prog_1)

        # Optimal CES frontier precompute cache (lazy init)
        self._frontier_pc: Optional[Dict[str, Any]] = None
        self._frontier_params_signature: Optional[Tuple] = None

    def get_automation_fraction(self, progress:float) -> float:
        """Compute the automation fraction"""
        if self.schedule_type == "linear":
            return np.clip(self.linear_aut_intercept + self.linear_aut_slope * progress, 0.0, 1.0)
        elif self.schedule_type == "exponential":
            assert False, "Exponential schedule type not implemented"
        else:
            assert False, "Invalid schedule type"

    def get_progress_from_index(self, index:float) -> float:
        """Get the progress from the index"""
        def get_prog_linear(index:float) -> float:
            """Get the progress from the index"""
            (prog_1, aut_1), (prog_2, aut_2) = self.anchor_points
            # Guard against zero or near-zero slope to avoid division-by-zero
            if abs(self.linear_aut_slope) < 1e-12:
                # Fallback: interpolate progress directly between anchors
                return (1.0 - index) * prog_1 + index * prog_2
            prog_slope = 1 / self.linear_aut_slope
            prog_for_zero = prog_1 - prog_slope * aut_1
            prog_for_one = prog_2 + prog_slope * (1 - aut_2)
            return index * prog_for_one + (1 - index) * prog_for_zero
        def get_prog_exponential(index:float) -> float:
            """Get the progress from the index"""
            assert False, "Exponential schedule type not implemented"
        if self.schedule_type == "linear":
            return get_prog_linear(index)
        elif self.schedule_type == "exponential":
            return get_prog_exponential(index)
        else:
            assert False, "Invalid schedule type"

    def get_FTE_per_GPU(self, index:float, progress:float) -> float:
        """Compute the FTE per GPU for a given task index at a given E.C. level"""
        index_progress = self.get_progress_from_index(index)
        if progress < index_progress:
            return 0.0
        progress_diff = progress - index_progress
        growth_factor = cfg.BASE_FOR_SOFTWARE_LOM ** (self.FTE_per_GPU_slope * progress_diff)
        return self.initial_FTE_per_GPU * growth_factor

    def get_crit_index(self, progress:float, aut_compute: float, L_HUMAN: float, rho: float) -> float:
        """
        Compute the critical index for a given progress
        TODO: Implement actual optimization to find the critical index
        """
        return self.get_automation_fraction(progress)

    def get_compute_allocation(self, index:float, progress:float, aut_compute: float, L_HUMAN: float, crit_index:float, rho: float) -> float:
        """Compute the optimal compute allocation for a given task index."""
        return 0
        
    
    def get_coding_labor(self, crit_index:float, progress:float, aut_compute: float, L_HUMAN: float, rho: float) -> float:
        """Compute the coding labor for a given critical index and progress"""
        
        
        return self.get_FTE_per_GPU(crit_index, progress)

    # ===================== Optimal CES fast path (embedded) =====================
    class _FrontierPrecomp(NamedTuple):
        grid_i: np.ndarray
        log_Eaut: np.ndarray
        log_B: np.ndarray
        log_F: np.ndarray
        log_Q: np.ndarray
        log_R: np.ndarray
        rho: float
        theta: float
        eta_init: float
        eps_i1: float

    @staticmethod
    def _interp(x_grid: np.ndarray, y_grid: np.ndarray, x: float) -> float:
        j = int(np.searchsorted(x_grid, x, side='right') - 1)
        j = int(np.clip(j, 0, len(x_grid) - 2))
        x0 = x_grid[j]
        x1 = x_grid[j + 1]
        t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        return (1.0 - t) * y_grid[j] + t * y_grid[j + 1]

    @staticmethod
    def _invert_monotone(x_grid: np.ndarray, y_grid: np.ndarray, y: float) -> float:
        lo, hi = 0, len(x_grid) - 1
        if y <= y_grid[lo]:
            return float(x_grid[lo])
        if y >= y_grid[hi]:
            return float(x_grid[hi])
        while hi - lo > 1:
            mid = (hi + lo) // 2
            if y_grid[mid] < y:
                lo = mid
            else:
                hi = mid
        y0, y1 = y_grid[lo], y_grid[hi]
        # Handle non-finite brackets robustly to avoid NaNs
        if not (np.isfinite(y0) and np.isfinite(y1) and np.isfinite(y)):
            if not np.isfinite(y0) and np.isfinite(y1):
                # Lower endpoint is -inf; any finite y falls near the upper x
                assert False, "Lower endpoint is -inf in _invert_monotone"
                return float(x_grid[hi])
            if np.isfinite(y0) and not np.isfinite(y1):
                # Upper endpoint is +inf; any finite y falls near the lower x
                assert False, "Upper endpoint is +inf in _invert_monotone"
                return float(x_grid[lo])
            # Both endpoints (or y) are non-finite; fall back to midpoint
            assert False, "Non-finite endpoints in _invert_monotone"
            return float(0.5 * (x_grid[lo] + x_grid[hi]))
        denom = y1 - y0
        if denom == 0.0:
            assert False, "Denominator is 0 in _invert_monotone"
            t = 0.0
        else:
            t = (y - y0) / denom
        t = float(np.clip(t, 0.0, 1.0))
        return float((1.0 - t) * x_grid[lo] + t * x_grid[hi])

    def _precompute_frontier(self, params) -> Optional[_FrontierPrecomp]:
        try:
            M = int(max(256, min(int(params.optimal_ces_grid_size), 16384)))
            grid_i = np.linspace(0.0, 1.0 - 1e-6, M)

            # Build E_aut(i) using existing inverse mapping: get_progress_from_index(i)
            # Then convert progress (OOMs of effective compute) to effective compute using baseline multiplier.
            # E_aut = (baseline_annual_compute_multiplier) ** progress_threshold
            log_base = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM))
            # progress threshold at each index (OOMs of effective compute)
            idx_progress = np.array([self.get_progress_from_index(float(ix)) for ix in grid_i], dtype=float)
            idx_progress = np.maximum.accumulate(idx_progress)
            # Effective capability threshold in log-space to avoid overflow
            log_Eaut = idx_progress * log_base

            rho = float(params.rho_coding_labor)
            theta = float(params.optimal_ces_theta)
            # Use FTE-per-GPU as baseline eta; allow multiplicative tweak via optimal_ces_eta_init
            eta_init = float(self.initial_FTE_per_GPU) * float(params.optimal_ces_eta_init)
            if not (rho < 1 and abs(rho) > 1e-18 and theta > 0 and eta_init > 0):
                return None

            di = grid_i[1] - grid_i[0]
            alpha = rho / (1.0 - rho)
            beta = alpha * theta
            gamma = theta / (1.0 - rho)

            # Compute weights in log space: w = exp(-beta * log_Eaut)
            w = np.exp(-beta * log_Eaut)
            B = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1])) * di])
            # log_B with safe handling of zeros using ufunc where/out (avoid evaluating log(0))
            log_B = np.empty_like(B)
            log_B.fill(-np.inf)
            np.log(B, out=log_B, where=(B > 0.0))

            one_minus_i = 1.0 - grid_i
            # log_Q = (1-rho) * log(1-i)
            log_one_minus_i = np.log(one_minus_i + 1e-18)
            log_Q = (1.0 - rho) * log_one_minus_i
            # log_R = log_Q - theta * log_Eaut
            log_R = log_Q - theta * log_Eaut
            # log_F = log_B + gamma * log_Eaut - log(1 - i)
            log_F = log_B + (gamma * log_Eaut) - log_one_minus_i
            # Enforce monotone in log-space
            log_F = np.maximum.accumulate(log_F)

            return AutomationModel._FrontierPrecomp(grid_i, log_Eaut, log_B, log_F, log_Q, log_R, rho, theta, eta_init, float(1.0 - grid_i[-1]))
        except Exception as e:
            logger.warning(f"Frontier precompute failed: {e}")
            return None

    def _ensure_frontier(self, params) -> Optional[_FrontierPrecomp]:
        signature = (
            float(params.rho_coding_labor),
            float(params.optimal_ces_theta),
            float(params.optimal_ces_eta_init),
            int(params.optimal_ces_grid_size),
            tuple(self.anchor_points),
        )
        if self._frontier_pc is None or self._frontier_params_signature != signature:
            pc = self._precompute_frontier(params)
            if pc is None:
                self._frontier_pc = None
                self._frontier_params_signature = None
            else:
                self._frontier_pc = pc
                self._frontier_params_signature = signature
        return self._frontier_pc

    def coding_labor_optimal_ces(self, H: float, C: float, logE: float, params, return_details: bool = False):
        pc = self._ensure_frontier(params)
        if pc is None:
            return None if return_details else None
        try:
            # Capability-limited index i_E: find rightmost log_Eaut <= logE (if logE < min(log_Eaut), force i_E=0)
            j = int(np.searchsorted(pc.log_Eaut, logE, side='right') - 1)
            if j < 0:
                # No tasks above threshold; boundary at 0
                i_E = float(pc.grid_i[0])
                i_c = i_E
                log_human = AutomationModel._interp(pc.grid_i, pc.log_Q, i_c)
                log_B_i = AutomationModel._interp(pc.grid_i, pc.log_B, i_c)
                if H <= 0.0:
                    L = 0.0
                    if return_details:
                        return L, {"i_cut": i_c, "i_E": i_E, "case": "boundary:E_below_min", "human_term": float(np.exp(log_human)), "comp_term": 0.0}
                    return L
                kappa = C / max(H, 1e-18)
                logS = np.log(max(kappa * pc.eta_init, 1e-300)) + (pc.theta * logE)
                log_comp = (pc.rho * logS) + ((1.0 - pc.rho) * log_B_i)
                log_L_norm_rho = np.logaddexp(log_human, log_comp)
                log_L = np.log(H) + (1.0 / pc.rho) * log_L_norm_rho
                L = float(np.exp(log_L))
                if return_details:
                    human = float(np.exp(log_human))
                    comp = float(np.exp(log_comp))
                    return L, {"i_cut": i_c, "i_E": i_E, "case": "boundary:E_below_min", "human_term": human, "comp_term": comp}
                return L
            j = int(np.clip(j, 0, len(pc.grid_i) - 2))
            i0, i1 = pc.grid_i[j], pc.grid_i[j + 1]
            le0, le1 = pc.log_Eaut[j], pc.log_Eaut[j + 1]
            tE = 0.0 if le1 == le0 else float(np.clip((logE - le0) / (le1 - le0), 0.0, 1.0))
            i_E = float((1.0 - tE) * i0 + tE * i1)

            if H <= 0.0:
                return 0.0 if not return_details else (0.0, {"i_cut": i_E, "i_E": i_E, "case": "H_zero", "human_term": 0.0, "comp_term": 0.0})
            kappa = C / max(H, 1e-18)
            # logS = log(kappa * eta_init) + theta * logE
            logS = np.log(max(kappa * pc.eta_init, 1e-300)) + (pc.theta * logE)

            log_F_iE = AutomationModel._interp(pc.grid_i, pc.log_F, i_E)
            if logS <= log_F_iE:
                i_c = AutomationModel._invert_monotone(pc.grid_i, pc.log_F, logS)
                log_human = AutomationModel._interp(pc.grid_i, pc.log_Q, i_c)
                log_R_i = AutomationModel._interp(pc.grid_i, pc.log_R, i_c)
                log_comp = logS + log_R_i
                log_L_norm_rho = np.logaddexp(log_human, log_comp)
                case = "interior"
            else:
                i_c = i_E
                log_human = AutomationModel._interp(pc.grid_i, pc.log_Q, i_c)
                log_B_i = AutomationModel._interp(pc.grid_i, pc.log_B, i_c)
                log_comp = (pc.rho * logS) + ((1.0 - pc.rho) * log_B_i)
                log_L_norm_rho = np.logaddexp(log_human, log_comp)
                case = "boundary"

            log_L = np.log(H) + (1.0 / pc.rho) * log_L_norm_rho
            L = float(np.exp(log_L))
            if return_details:
                human = float(np.exp(log_human))
                comp = float(np.exp(log_comp))
                return L, {"i_cut": i_c, "i_E": i_E, "case": case, "human_term": human, "comp_term": comp}
            return L
        except Exception as e:
            logger.warning(f"coding_labor_optimal_ces failed at runtime: {e}")
            return None if not return_details else (None, {"error": str(e)})


@dataclass
class Parameters:
    """Model parameters with validation"""

    human_only: bool = field(default_factory=lambda: False)
    
    # Production function parameters
    rho_coding_labor: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['rho_coding_labor'])
    direct_input_exp_cap_ces_params: bool = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['direct_input_exp_cap_ces_params'])
    rho_experiment_capacity: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['rho_experiment_capacity'])
    alpha_experiment_capacity: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['alpha_experiment_capacity'])
    r_software: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['r_software'])
    software_progress_rate_at_reference_year: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['software_progress_rate_at_reference_year'])
    experiment_compute_exponent: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['experiment_compute_exponent'])
    
    # Automation parameters
    automation_fraction_at_superhuman_coder: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_fraction_at_superhuman_coder'])
    automation_anchors: Optional[Dict[float, float]] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_anchors'])
    automation_model: Optional[AutomationModel] = field(default_factory=lambda: None)
    automation_interp_type: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_interp_type'])
    swe_multiplier_at_present_day: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['swe_multiplier_at_present_day'])
    # AI Research Taste sigmoid parameters
    ai_research_taste_at_superhuman_coder: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['ai_research_taste_at_superhuman_coder'])
    # Optional: allow specifying the superhuman-coder taste as SD within the human range
    ai_research_taste_at_superhuman_coder_sd: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('ai_research_taste_at_superhuman_coder_sd'))
    ai_research_taste_slope: float = field(default_factory=lambda: cfg.TASTE_SLOPE_DEFAULTS.get(cfg.DEFAULT_TASTE_SCHEDULE_TYPE, cfg.DEFAULT_PARAMETERS['ai_research_taste_slope']))
    taste_schedule_type: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['taste_schedule_type'])
    progress_at_sc: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('progress_at_sc'))
    sc_time_horizon_minutes: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['sc_time_horizon_minutes'])
    # Pre-gap SC horizon minutes (formerly saturation_horizon_minutes)
    pre_gap_sc_time_horizon: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['pre_gap_sc_time_horizon'])
    horizon_extrapolation_type: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['horizon_extrapolation_type'])
    
    # Manual horizon fitting parameters
    present_day: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['present_day'])
    present_horizon: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['present_horizon'])
    present_doubling_time: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['present_doubling_time'])
    doubling_difficulty_growth_rate: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['doubling_difficulty_growth_rate'])
    
    # Normalization
    coding_labor_normalization: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['coding_labor_normalization'])
    
    # Baseline Annual Compute Multiplier
    baseline_annual_compute_multiplier: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['baseline_annual_compute_multiplier'])
    
    # exp capacity pseudoparameters
    inf_labor_asymptote: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['inf_labor_asymptote'])
    inf_compute_asymptote: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['inf_compute_asymptote'])
    labor_anchor_exp_cap: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['labor_anchor_exp_cap'])
    compute_anchor_exp_cap: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['compute_anchor_exp_cap'])
    inv_compute_anchor_exp_cap: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['inv_compute_anchor_exp_cap'])
    # penalty on parallel coding labor in exp capacity CES
    parallel_penalty: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['parallel_penalty'])

    # Research taste distribution parameter
    median_to_top_taste_multiplier: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['median_to_top_taste_multiplier'])

    # Benchmarks and gaps mode
    include_gap: Union[str, bool] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['include_gap'])
    gap_years: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['gap_years'])

    # Coding labor mode and optimal CES params (see AUTOMATION_SUGGESTION.md)
    coding_labor_mode: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('coding_labor_mode', 'simple_ces'))
    optimal_ces_theta: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('optimal_ces_theta', 1.0))
    optimal_ces_eta_init: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('optimal_ces_eta_init', 1.0))
    optimal_ces_grid_size: int = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('optimal_ces_grid_size', 4096))
    
    def __post_init__(self):
        """Validate and sanitize parameters after initialization"""
        # Sanitize elasticity parameters
        if not np.isfinite(self.rho_coding_labor):
            logger.warning(f"Non-finite rho_coding_labor: {self.rho_coding_labor}, setting to 0")
            self.rho_coding_labor = 0.0

                
        if not np.isfinite(self.r_software):
            logger.warning(f"Non-finite r_software: {self.r_software}, setting to 1.0")
            self.r_software = 1.0
                
        # Sanitize automation parameters
        if not np.isfinite(self.automation_fraction_at_superhuman_coder):
            logger.warning(f"Non-finite automation_fraction_at_superhuman_coder: {self.automation_fraction_at_superhuman_coder}, setting to {cfg.DEFAULT_PARAMETERS['automation_fraction_at_superhuman_coder']}")
            self.automation_fraction_at_superhuman_coder = cfg.DEFAULT_PARAMETERS['automation_fraction_at_superhuman_coder']
                
        # Sanitize AI research taste parameters
        # If SD-specification is provided, convert to raw taste via TasteDistribution
        if self.ai_research_taste_at_superhuman_coder_sd is not None and np.isfinite(self.ai_research_taste_at_superhuman_coder_sd):
            try:
                taste_distribution_tmp = TasteDistribution(median_to_top_gap=self.median_to_top_taste_multiplier)
                converted_taste = taste_distribution_tmp.get_taste_at_sd(float(self.ai_research_taste_at_superhuman_coder_sd))
                if np.isfinite(converted_taste):
                    self.ai_research_taste_at_superhuman_coder = float(converted_taste)
            except Exception as e:
                logger.warning(f"Failed converting ai_research_taste_at_superhuman_coder_sd to taste: {e}")
        if not np.isfinite(self.ai_research_taste_at_superhuman_coder):
            logger.warning(f"Non-finite ai_research_taste_at_superhuman_coder: {self.ai_research_taste_at_superhuman_coder}, setting to {cfg.DEFAULT_PARAMETERS['ai_research_taste_at_superhuman_coder']}")
            self.ai_research_taste_at_superhuman_coder = cfg.DEFAULT_PARAMETERS['ai_research_taste_at_superhuman_coder']
                
        if not np.isfinite(self.ai_research_taste_slope):
            logger.warning(f"Non-finite ai_research_taste_slope: {self.ai_research_taste_slope}, setting to 1.0")
            self.ai_research_taste_slope = 1.0        
        # Sanitize time horizon parameter
        if not np.isfinite(self.sc_time_horizon_minutes) or self.sc_time_horizon_minutes <= 0:
            logger.warning(f"Invalid sc_time_horizon_minutes: {self.sc_time_horizon_minutes}, setting to {cfg.DEFAULT_PARAMETERS['sc_time_horizon_minutes']}")
            self.sc_time_horizon_minutes = cfg.DEFAULT_PARAMETERS['sc_time_horizon_minutes']

        # Sanitize parallel_penalty
        if not np.isfinite(self.parallel_penalty):
            logger.warning(f"Non-finite parallel_penalty: {self.parallel_penalty}, setting to {cfg.DEFAULT_PARAMETERS['parallel_penalty']}")
            self.parallel_penalty = cfg.DEFAULT_PARAMETERS['parallel_penalty']
        else:
            self.parallel_penalty = float(np.clip(self.parallel_penalty, cfg.PARALLEL_PENALTY_MIN, cfg.PARALLEL_PENALTY_MAX))
        # Validate pre-gap SC horizon
        if not np.isfinite(self.pre_gap_sc_time_horizon) or self.pre_gap_sc_time_horizon <= 0:
            logger.warning(f"Invalid pre_gap_sc_time_horizon: {self.pre_gap_sc_time_horizon}, setting to {cfg.DEFAULT_PARAMETERS['pre_gap_sc_time_horizon']}")
            self.pre_gap_sc_time_horizon = float(cfg.DEFAULT_PARAMETERS['pre_gap_sc_time_horizon'])
        
        # Sanitize categorical parameters
        if self.horizon_extrapolation_type not in cfg.HORIZON_EXTRAPOLATION_TYPES:
            logger.warning(f"Invalid horizon_extrapolation_type: {self.horizon_extrapolation_type}, setting to default")
            self.horizon_extrapolation_type = cfg.DEFAULT_HORIZON_EXTRAPOLATION_TYPE
        if self.automation_interp_type not in ["linear", "exponential"]:
            logger.warning(f"Invalid automation_interp_type: {self.automation_interp_type}, setting to default 'exponential'")
            self.automation_interp_type = cfg.DEFAULT_PARAMETERS['automation_interp_type']
        
        # Sanitize manual horizon fitting parameters
        if not np.isfinite(self.present_day):
            logger.warning(f"Non-finite present_day: {self.present_day}, setting to default")
            self.present_day = cfg.DEFAULT_present_day
        
        # Validate optional parameters - if provided, ensure they're finite and positive
        if self.present_horizon is not None:
            if not np.isfinite(self.present_horizon) or self.present_horizon <= 0:
                logger.warning(f"Invalid present_horizon: {self.present_horizon}, setting to None for optimization")
                self.present_horizon = None
        
        if self.present_doubling_time is not None:
            if not np.isfinite(self.present_doubling_time) or self.present_doubling_time <= 0:
                logger.warning(f"Invalid present_doubling_time: {self.present_doubling_time}, setting to None for optimization")
                self.present_doubling_time = None
        
        if self.doubling_difficulty_growth_rate is not None:
            if not np.isfinite(self.doubling_difficulty_growth_rate):
                logger.warning(f"Invalid doubling_difficulty_growth_rate: {self.doubling_difficulty_growth_rate}, setting to None for optimization")
                self.doubling_difficulty_growth_rate = None

        if self.inv_compute_anchor_exp_cap is not None:
            logger.warning(f"inv_compute_anchor_exp_cap is not None, overriding compute_anchor_exp_cap to 1 / inv_compute_anchor_exp_cap")
            self.compute_anchor_exp_cap = 1 / self.inv_compute_anchor_exp_cap

        # Sanitize include_gap parameter (new API)
        include_gap_bool = False
        try:
            if isinstance(self.include_gap, str):
                val = self.include_gap.strip().lower()
                if val in ("gap", "yes", "true", "1"):  # accept common truthy variants
                    include_gap_bool = True
                elif val in ("no gap", "no", "false", "0"):
                    include_gap_bool = False
                else:
                    include_gap_bool = bool(cfg.DEFAULT_PARAMETERS['include_gap'] == 'gap')
            else:
                include_gap_bool = bool(self.include_gap)
        except Exception:
            include_gap_bool = bool(cfg.DEFAULT_PARAMETERS['include_gap'] == 'gap')
        # Normalize include_gap to canonical string for consistency
        self.include_gap = "gap" if include_gap_bool else "no gap"
        if not np.isfinite(self.gap_years) or self.gap_years < 0:
            logger.warning(f"Invalid gap_years: {self.gap_years}, setting to default")
            self.gap_years = cfg.DEFAULT_PARAMETERS['gap_years']

        # Sanitize normalization parameters
        if not np.isfinite(self.coding_labor_normalization) or self.coding_labor_normalization <= 0:
            logger.warning(f"Invalid coding_labor_normalization: {self.coding_labor_normalization}, setting to 1.0")
            self.coding_labor_normalization = 1.0
        
        # Validate baseline annual compute multiplier
        if not np.isfinite(self.baseline_annual_compute_multiplier) or self.baseline_annual_compute_multiplier <= 0:
            logger.warning(f"Invalid baseline_annual_compute_multiplier: {self.baseline_annual_compute_multiplier}, setting to default")
            self.baseline_annual_compute_multiplier = cfg.BASELINE_ANNUAL_COMPUTE_MULTIPLIER_DEFAULT

        # Sanitize coding_labor_mode
        if self.coding_labor_mode not in ("simple_ces", "optimal_ces"):
            logger.warning(f"Invalid coding_labor_mode: {self.coding_labor_mode}, defaulting to 'simple_ces'")
            self.coding_labor_mode = 'simple_ces'
        # Sanitize optimal CES parameters
        try:
            if not np.isfinite(self.optimal_ces_theta) or self.optimal_ces_theta <= 0:
                self.optimal_ces_theta = float(cfg.DEFAULT_PARAMETERS.get('optimal_ces_theta', 1.0))
            if not np.isfinite(self.optimal_ces_eta_init) or self.optimal_ces_eta_init <= 0:
                self.optimal_ces_eta_init = float(cfg.DEFAULT_PARAMETERS.get('optimal_ces_eta_init', 1.0))
            if not np.isfinite(self.optimal_ces_grid_size) or int(self.optimal_ces_grid_size) < 256:
                self.optimal_ces_grid_size = int(cfg.DEFAULT_PARAMETERS.get('optimal_ces_grid_size', 4096))
            else:
                self.optimal_ces_grid_size = int(self.optimal_ces_grid_size)
        except Exception:
            self.coding_labor_mode = 'simple_ces'

@dataclass
class AnchorConstraint:
    """Specifies a constraint for parameter estimation"""
    # Dict mapping variable names to values (can be partial)
    conditions: Dict[str, float]  # e.g., {"automation_fraction": 0.9, "inference_compute": 1e6}
    # Expected outcome
    target_variable: str  # e.g., "progress_rate"
    target_value: float   # e.g., 5.0
    weight: float = 1.0   # Weight in optimization


def _ces_function(X1: float, X2: float, w1: float, rho: float) -> float:
    """
    Computes the CES function with the standard substitution parameter rho.
    Y = (w1*X1^rho + (1-w1)*X2^rho)^(1/rho)

    Args:
        X1: First input
        X2: Second input
        w1: Weight of the first input [0,1]
        rho: Standard substitution parameter in (-inf, 1]

    Returns:
        Combined output.
    """
    w1 = np.clip(w1, 0.0, 1.0)
    w2 = 1 - w1

    # Handle edge cases for rho
    if abs(rho) < cfg.RHO_COBB_DOUGLAS_THRESHOLD:  # Cobb-Douglas case (limit of CES as rho -> 0)
        if X1 > 0 and X2 > 0:
            try:
                # Use log to prevent overflow
                log_result = w1 * np.log(X1) + w2 * np.log(X2)
                return np.exp(log_result)
            except (ValueError, OverflowError):
                # Fallback for very large numbers
                logger.warning("Overflow in Cobb-Douglas case, using linear fallback")
                return w1 * X1 + w2 * X2 # Fallback to linear
        else:
            return 0.0

    if rho == 1.0:  # Perfect substitutes
        return w1 * X1 + w2 * X2

    if rho < cfg.RHO_LEONTIEF_THRESHOLD:  # Nearly perfect complements (limit as rho -> -inf)
        return min(X1, X2)

    # Handle edge cases for weights
    if w1 == 0: return X2
    if w2 == 0: return X1
    if X1 == 0 and X2 == 0: return 0

    # Standard CES formula with numerical safeguards
    try:
        # Handle cases where one input is zero
        if X1 == 0:
            return np.power(w2, 1 / rho) * X2
        if X2 == 0:
            return np.power(w1, 1 / rho) * X1

        term1 = w1 * np.power(X1, rho)
        term2 = w2 * np.power(X2, rho)
        total = term1 + term2

        if total <= 0:
            logger.warning(f"CES inner term is non-positive ({total}) with rho={rho}. Fallback to min.")
            return min(X1, X2)

        result = np.power(total, 1 / rho)

        if not np.isfinite(result):
            logger.warning(f"Non-finite result in CES (rho={rho}), using linear fallback.")
            return w1 * X1 + w2 * X2 # Fallback to linear
        return result

    except (OverflowError, ZeroDivisionError, ValueError) as e:
        logger.warning(f"Numerical error in CES (rho={rho}): {e}, using linear fallback.")
        return w1 * X1 + w2 * X2


def compute_coding_labor(automation_fraction: float, inference_compute: float, L_HUMAN: float, rho: float, parallel_penalty: float, cognitive_normalization: float = 1.0, human_only: bool = False) -> float:
    """
    CES combination of AI and human labor using an alternative formulation inspired by
    the structure in FORMULAS.md: Y = ( (A^(1-rho) * inference_compute^rho) + ((1-A)^(1-rho) * L_HUMAN^rho) )^(1/rho)
    This can be seen as a standard CES function on effective labor inputs inference_compute/A and L_HUMAN/(1-A).
    
    Args:
        automation_fraction: Fraction of work automated (A) [0,1]
        inference_compute: AI labor supply
        L_HUMAN: Human labor supply
        rho: Standard substitution parameter in (-inf, 1].
             rho -> 1: perfect substitutes (Y = inference_compute + L_HUMAN)
             rho -> 0: Cobb-Douglas (Y = (inference_compute/A)^A * (L_HUMAN/(1-A))^(1-A))
             rho -> -inf: perfect complements (Y = min(inference_compute/A, L_HUMAN/(1-A)))
        parallel_penalty: Power transformation parameter applied to CES output before normalization
        cognitive_normalization: Normalization constant for cognitive output
    
    Returns:
        Cognitive output
    """
    if human_only:
        return (L_HUMAN ** parallel_penalty) * cognitive_normalization 
    
    # Input validation
    if not all(np.isfinite([automation_fraction, inference_compute, L_HUMAN, rho, parallel_penalty, cognitive_normalization])):
        logger.warning("Non-finite inputs to compute_coding_labor")
        return 0.0
    
    if inference_compute < 0 or L_HUMAN < 0:
        logger.warning("Negative labor inputs")
        return 0.0
    
    # Clamp automation fraction to valid range, avoiding 0 and 1 for division.
    a = np.clip(automation_fraction, cfg.AUTOMATION_FRACTION_CLIP_MIN, 1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN)

    # Handle edge cases for rho
    if abs(rho) < cfg.RHO_COBB_DOUGLAS_THRESHOLD:  # Cobb-Douglas case
        if inference_compute > 0 and L_HUMAN > 0:
            try:
                # Effective inputs are inference_compute/a and L_HUMAN/(1-a)
                # Cobb-Douglas form is (inference_compute/a)^a * (L_HUMAN/(1-a))^(1-a)
                term1 = a * (np.log(inference_compute) - np.log(a))
                term2 = (1 - a) * (np.log(L_HUMAN) - np.log(1 - a))
                log_result = term1 + term2
                result = np.exp(log_result)
            except (ValueError, OverflowError):
                logger.warning("Overflow in Cobb-Douglas case, using linear fallback")
                result = a * inference_compute + (1-a) * L_HUMAN
        else:
            result = 0.0

    elif rho == 1.0:  # Perfect substitutes
        # Formula simplifies to inference_compute + L_HUMAN
        result = inference_compute + L_HUMAN

    elif rho < cfg.RHO_LEONTIEF_THRESHOLD:  # Nearly perfect complements (Leontief)
        # Limit is min(inference_compute/a, L_HUMAN/(1-a))
        result = min(inference_compute / a, L_HUMAN / (1 - a))

    else: # Standard CES formula for the alternative model
        try:
            term1 = np.power(a, 1 - rho) * np.power(inference_compute, rho) if inference_compute > 0 else 0
            term2 = np.power(1 - a, 1 - rho) * np.power(L_HUMAN, rho) if L_HUMAN > 0 else 0
            
            total = term1 + term2
            if total <= 0:
                logger.warning(f"Alternative CES inner term is non-positive ({total}) with rho={rho}. Fallback to min.")
                result = min(inference_compute / a, L_HUMAN / (1 - a))
            else:
                result = np.power(total, 1 / rho)

            if not np.isfinite(result):
                logger.warning(f"Non-finite result in Alternative CES (rho={rho}), using linear fallback.")
                result = a * inference_compute + (1-a) * L_HUMAN
        
        except (OverflowError, ZeroDivisionError, ValueError) as e:
            logger.warning(f"Numerical error in Alternative CES (rho={rho}): {e}, using linear fallback.")
            result = a * inference_compute + (1-a) * L_HUMAN
            
    # Apply parallel_penalty transformation before normalization
    try:
        result_with_lambda = np.power(result, parallel_penalty)
        if not np.isfinite(result_with_lambda):
            logger.warning(f"Non-finite result after parallel_penalty transformation (parallel_penalty={parallel_penalty}), using original result")
            result_with_lambda = result
    except (OverflowError, ValueError) as e:
        logger.warning(f"Error applying parallel_penalty transformation (parallel_penalty={parallel_penalty}): {e}, using original result")
        result_with_lambda = result
    
    return result_with_lambda * cognitive_normalization

def compute_rho_from_asymptotes(inf_labor_asymptote: float, inf_compute_asymptote: float) -> float:
    """
    Compute the substitution parameter rho from the asymptotes of the experiment capacity CES.
    Solve for rho in the equation:
    inf_labor_asymptote**rho + inf_compute_asymptote**rho = 1
    """
    # Validate inputs
    if not np.isfinite(inf_labor_asymptote) or not np.isfinite(inf_compute_asymptote):
        logger.warning("Non-finite asymptotes provided to compute_rho_from_asymptotes; falling back to 0.0")
        return 0.0

    # Ensure strictly positive bases to avoid undefined powers
    a = float(inf_labor_asymptote)
    b = float(inf_compute_asymptote)

    if a <= 0 or b <= 0:
        logger.warning(f"Non-positive asymptote(s) a={a}, b={b}; falling back to 0.0")
        return 0.0

    # Small epsilon to avoid numerical issues when values are extremely close to 0
    a = max(a, cfg.NORMALIZATION_MIN)
    b = max(b, cfg.NORMALIZATION_MIN)

    def equation(rho: float) -> float:
        return np.power(a, rho) + np.power(b, rho) - 1.0

    # Bracket within the valid CES range for rho
    lower = cfg.RHO_CLIP_MIN
    upper = 1.0

    f_lower = equation(lower)
    f_upper = equation(upper)

    # If already satisfies at a bound, return that bound (rare but possible)
    if abs(f_lower) < 1e-12:
        return float(lower)
    if abs(f_upper) < 1e-12:
        return float(upper)

    # Check for a valid bracket
    if f_lower * f_upper > 0:
        # This indicates inputs inconsistent with rho in (-inf, 1]; fallback to Cobb-Douglas
        logger.warning(
            f"Asymptotes a={a}, b={b} do not bracket a root in [${lower}, ${upper}]; falling back to rho=0.0"
        )
        return 0.0

    try:
        rho = optimize.brentq(equation, lower, upper, maxiter=100, xtol=1e-12)
    except Exception as e:
        logger.warning(f"Root finding failed in compute_rho_from_asymptotes: {e}; falling back to 0.0")
        return 0.0

    # Clip to valid range and return
    return float(np.clip(rho, cfg.RHO_CLIP_MIN, 1.0))

def compute_experiment_compute_exponent_from_anchor(inf_compute_asymptote: float, inf_labor_asymptote: float, compute_anchor: tuple[float, float], rho: float) -> float:
    """
    Compute the experiment compute exponent from the asymptotes and anchor.

    """
    k = (inf_compute_asymptote/inf_labor_asymptote)**rho
    N = compute_anchor[0]
    M = compute_anchor[1]

    # return (1/(rho*np.log(N))) * np.log(M**rho + k*(M**rho-1))
    res = (1/(rho*np.log(N))) * np.log((1+k)*(M**rho) - k)
    return res


# def compute_coding_labor_exponent_from_anchor(inf_compute_asymptote: float, inf_labor_asymptote: float, labor_anchor: tuple[float, float], rho: float) -> float:
#     """
#     Compute the experiment compute exponent from the asymptotes and anchor.
#     Currently unused.
#     """
#     k = (inf_compute_asymptote/inf_labor_asymptote)**rho
#     L = labor_anchor[0]
#     S = labor_anchor[1]

#     return (1/(rho*np.log(L))) * np.log(S**rho + (S**rho-1)/k)
    
def compute_alpha_experiment_capacity_from_asymptotes(inf_labor_asymptote: float, inf_compute_asymptote: float, experiment_compute_exponent, current_exp_compute:float, current_serial_coding_labor:float, rho: float) -> float:
    """
    Compute the alpha parameter for the experiment capacity CES from the asymptotes and anchors.
    """
    zeta = experiment_compute_exponent
    C = current_exp_compute
    L = current_serial_coding_labor
    # return 1/(1 + k * (C**d / L)**rho)
    res =  1/(1 + ((C**zeta / L)*(inf_compute_asymptote/inf_labor_asymptote))**rho)
    return res

def compute_exp_capacity_params_from_anchors(inf_labor_asymptote: float, inf_compute_asymptote: float, compute_anchor: tuple[float, float], labor_anchor: tuple[float, float], current_exp_compute:float, current_coding_labor:float, parallel_penalty: float = 0.0) -> tuple[float, float, float, float]:
    """
    Compute the parameters for the experiment capacity CES from the asymptotes and anchors.
    """
    rho = compute_rho_from_asymptotes(inf_labor_asymptote, inf_compute_asymptote)
    experiment_compute_exponent = compute_experiment_compute_exponent_from_anchor(inf_compute_asymptote, inf_labor_asymptote, compute_anchor, rho)
    serial_coding_labor = current_coding_labor ** parallel_penalty
    alpha_experiment_capacity = compute_alpha_experiment_capacity_from_asymptotes(inf_labor_asymptote, inf_compute_asymptote, experiment_compute_exponent, current_exp_compute, serial_coding_labor, rho)
    # import pdb; pdb.set_trace()
    return rho, alpha_experiment_capacity, experiment_compute_exponent

def compute_research_effort(experiment_compute: float, serial_coding_labor: float, alpha_experiment_capacity: float, rho: float, experiment_compute_exponent: float, aggregate_research_taste: float = cfg.AGGREGATE_RESEARCH_TASTE_BASELINE) -> float:
    """
    CES combination of compute and cognitive work to determine research stock growth rate.
    This replaces the previous direct software progress calculation.
    
    Args:
        experiment_compute: Experiment compute budget
        serial_coding_labor: Output from cognitive work
        alpha_experiment_capacity: Weight on experiment compute [0,1]
        rho: Standard substitution parameter in (-inf, 1].
             rho -> 1: perfect substitutes
             rho -> 0: Cobb-Douglas
             rho -> -inf: perfect complements
        experiment_compute_exponent: Discounting factor for experiment compute (see cfg.experiment_compute_exponent_CLIP_MIN, cfg.experiment_compute_exponent_CLIP_MAX)
        aggregate_research_taste: Multiplier for research effectiveness (default 1.0)
    
    Returns:
        Research stock growth rate RS'(t)
    """
    # Input validation
    if not all(np.isfinite([experiment_compute, serial_coding_labor, alpha_experiment_capacity, rho, experiment_compute_exponent, aggregate_research_taste])):
        logger.warning("Non-finite inputs to compute_research_effort")
        return 0.0
    
    if experiment_compute < 0 or serial_coding_labor < 0:
        logger.warning("Negative inputs to compute_research_effort")
        return 0.0
    
    if aggregate_research_taste < 0:
        logger.warning(f"Negative aggregate_research_taste: {aggregate_research_taste}, setting to {cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK}")
        aggregate_research_taste = cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK
    
    if experiment_compute_exponent < cfg.experiment_compute_exponent_CLIP_MIN or experiment_compute_exponent > cfg.experiment_compute_exponent_CLIP_MAX:
        logger.warning(f"Invalid experiment_compute_exponent value {experiment_compute_exponent}, clamping to [{cfg.experiment_compute_exponent_CLIP_MIN}, {cfg.experiment_compute_exponent_CLIP_MAX}]")
        experiment_compute_exponent = np.clip(experiment_compute_exponent, cfg.experiment_compute_exponent_CLIP_MIN, cfg.experiment_compute_exponent_CLIP_MAX)
    
    # Clamp alpha_experiment_capacity to valid range
    alpha_experiment_capacity = np.clip(alpha_experiment_capacity, 0.0, 1.0)
    
    # Apply discounting factor to experiment compute
    discounted_experiment_compute = np.power(experiment_compute, experiment_compute_exponent)
    
    # Use the generic CES function for computation
    rate = _ces_function(discounted_experiment_compute, serial_coding_labor, alpha_experiment_capacity, rho)
    
    # Cap extremely large rates to prevent numerical issues
    if rate > cfg.MAX_RESEARCH_EFFORT:
        logger.warning(f"Very large research stock rate {rate}, capping to {cfg.MAX_RESEARCH_EFFORT}")
        rate = cfg.MAX_RESEARCH_EFFORT
    
    # Apply aggregate research taste multiplier
    final_rate = rate * aggregate_research_taste
    
    # Apply final cap to prevent numerical issues with the multiplied result
    if final_rate > cfg.MAX_RESEARCH_EFFORT:
        logger.warning(f"Very large final research stock rate {final_rate}, capping to {cfg.MAX_RESEARCH_EFFORT}")
        final_rate = cfg.MAX_RESEARCH_EFFORT
        
    return final_rate


def compute_software_progress_rate(research_stock: float, research_effort: float, 
                                 initial_research_stock: float, initial_research_effort: float,
                                 r_software: float) -> float:
    """
    Compute software progress rate using research stocding labor
    S(t) = RS'(t) / RS(t) * r_software
    
    Args:
        research_stock: Current research stock RS(t)
        research_effort: Current research stock rate RS'(t)
        r_software: Software progress share parameter s [0.1,10]
    
    Returns:
        Software progress rate multiplied by r_software
    """
    # Input validation
    if not all(np.isfinite([research_stock, research_effort, initial_research_stock, initial_research_effort, r_software])):
        logger.warning("Non-finite inputs to compute_software_progress_rate")
        return 0.0
    
    if research_stock <= 0 or initial_research_stock <= 0:
        logger.warning("Non-positive research stock values")
        return 0.0
    
    if initial_research_effort <= 0:
        logger.warning("Non-positive initial research stock rate")
        return 0.0
    
    if r_software < 0.01:
        logger.warning(f"Invalid r_software: {r_software}, must be in [0.1,10]")
        return 0.0
    
    # Compute software progress rate using research stock ratio formula
    try:
        numerator = research_effort
        denominator = research_stock
        
        if denominator == 0:
            logger.warning("Zero denominator in software progress rate calculation")
            return 0.0
        
        software_progress_rate = numerator / denominator
        
        if not np.isfinite(software_progress_rate) or software_progress_rate < 0:
            logger.warning(f"Invalid software progress rate: {software_progress_rate}")
            return 0.0
        # Apply software progress share multiplier: s
        final_rate = software_progress_rate * r_software
        rate_at_base_for_software_lom = final_rate * 1.0 / np.log(cfg.BASE_FOR_SOFTWARE_LOM)
        
        if not np.isfinite(final_rate) or final_rate < 0:
            logger.warning(f"Invalid final software progress rate after share multiplier: {final_rate}")
            return 0.0
        
        return rate_at_base_for_software_lom
        
    except (ZeroDivisionError, OverflowError) as e:
        logger.warning(f"Error computing software progress rate: {e}")
        return 0.0


def compute_overall_progress_rate(software_progress_rate: float, training_compute_growth_rate: float) -> float:
    """
    Sum of software and training progress rates
    
    Args:
        software_progress_rate: Rate from software development (already adjusted by software share)
        training_compute_growth_rate: Training compute budget
    
    Returns:
        Overall progress rate (sum of software and hardware contributions)
    """
    return software_progress_rate + training_compute_growth_rate


def _log_interp(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    """
    Perform log-space interpolation for exponential trends.
    
    Args:
        x: Point to interpolate at
        xp: Known x-coordinates (must be sorted)
        fp: Known y-coordinates (must be positive for log-space)
        
    Returns:
        Interpolated value
    """
    # Ensure all values are positive for log-space interpolation
    if np.any(fp <= 0):
        # Fall back to linear interpolation if any values are non-positive
        return np.interp(x, xp, fp)
    
    # Perform log-space interpolation: log(y) = log(y1) + (log(y2) - log(y1)) * (x - x1) / (x2 - x1)
    log_fp = np.log(fp)
    log_interpolated = np.interp(x, xp, log_fp)
    return np.exp(log_interpolated)


def _find_exponential_crossing_time(
    times: np.ndarray,
    values: np.ndarray,
    target: float,
) -> Optional[float]:
    """
    Return the earliest time at which an exponentially growing series crosses a target.

    Assumes the series between adjacent samples grows exponentially. When the target lies
    between two samples (t0, v0) and (t1, v1), the crossing time t* is computed in log-space:

        f = (ln(target) - ln(v0)) / (ln(v1) - ln(v0))
        t* = t0 + f * (t1 - t0)

    Falls back to linear interpolation if non-positive values prevent log-space math.

    Args:
        times: 1-D array of time points (monotonic ascending recommended)
        values: 1-D array of series values at those times
        target: Threshold to cross (typically > 0 for exponential interpolation)

    Returns:
        The crossing time as float, or None if no crossing occurs or inputs invalid.
    """
    try:
        t = np.asarray(times, dtype=float)
        v = np.asarray(values, dtype=float)
        y = float(target)

        if t.size == 0 or t.size != v.size or not np.isfinite(y):
            return None

        # Find the first index j where v[j] >= target
        crossing_indices = np.where(v >= y)[0]
        if crossing_indices.size == 0:
            return None

        j = int(crossing_indices[0])
        if j == 0:
            return float(t[0])

        v0 = v[j - 1]
        v1 = v[j]
        t0 = t[j - 1]
        t1 = t[j]

        if (np.isfinite(v0) and np.isfinite(v1) and np.isfinite(t0) and np.isfinite(t1) and t1 != t0):
            # Prefer log-space interpolation if possible
            if v0 > 0 and v1 > 0 and y > 0 and v1 != v0:
                frac = (np.log(y) - np.log(v0)) / (np.log(v1) - np.log(v0))
                frac = float(min(max(frac, 0.0), 1.0))
                return float(t0 + frac * (t1 - t0))
            # Fallback to linear interpolation
            if v1 != v0:
                frac = (y - v0) / (v1 - v0)
                frac = float(min(max(frac, 0.0), 1.0))
                return float(t0 + frac * (t1 - t0))
            # If values are equal, step to the right edge
            return float(t1)

        return None
    except Exception:
        return None

def calculate_initial_research_stock(time_series_data: TimeSeriesData, params: Parameters, 
                                   initial_progress: float = 0.0) -> float:
    """
    Calculate initial research stock using the formula: RS(0) = (RS'(0))^2 / RS''(0)
    
    Args:
        time_series_data: Input time series data
        params: Model parameters  
        initial_progress: Initial cumulative progress
        
    Returns:
        Calculated initial research stock value
    """
    try:
        start_time = time_series_data.time[0]
        dt = 1e-6  # Small time step for numerical differentiation
        
        # Get initial conditions at t=0
        
        
        L_HUMAN_0 = _log_interp(start_time, time_series_data.time, time_series_data.L_HUMAN)
        inference_compute_0 = _log_interp(start_time, time_series_data.time, time_series_data.inference_compute)
        experiment_compute_0 = _log_interp(start_time, time_series_data.time, time_series_data.experiment_compute)
        
        if params.human_only:
            coding_labor_0 = compute_coding_labor(None, inference_compute_0, L_HUMAN_0, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization, human_only=True)
            logger.info(f"HUMAN-ONLY::: coding_labor_0: {coding_labor_0}")
            initial_aggregate_research_taste = 1.0
        else:
            initial_automation = compute_automation_fraction(initial_progress, params)
            initial_ai_research_taste = compute_ai_research_taste(initial_progress, params)
            initial_aggregate_research_taste = compute_aggregate_research_taste(initial_ai_research_taste, median_to_top_gap=params.median_to_top_taste_multiplier)
            coding_labor_0 = compute_coding_labor(
                initial_automation, inference_compute_0, L_HUMAN_0, 
                params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization
            )
            logger.info(f"ACTUAL::: coding_labor_0: {coding_labor_0}")
        
        # Calculate RS'(0)
        rs_rate_0 = compute_research_effort(
            experiment_compute_0, coding_labor_0, 
            params.alpha_experiment_capacity, params.rho_experiment_capacity, params.experiment_compute_exponent, initial_aggregate_research_taste
        )
        
        # Calculate RS'(dt) for numerical differentiation
        # Use log-space interpolation for exponential trends
        L_HUMAN_dt = _log_interp(start_time + dt, time_series_data.time, time_series_data.L_HUMAN)
        inference_compute_dt = _log_interp(start_time + dt, time_series_data.time, time_series_data.inference_compute)
        experiment_compute_dt = _log_interp(start_time + dt, time_series_data.time, time_series_data.experiment_compute)
        
        # Automation fraction changes very little over small dt, so use same value
        if params.human_only:
            coding_labor_dt = compute_coding_labor(None, inference_compute_dt, L_HUMAN_dt, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization, human_only=True)
            logger.info(f"HUMAN-ONLY::: coding_labor_dt: {coding_labor_dt}")
        else:
            coding_labor_dt = compute_coding_labor(
                initial_automation, inference_compute_dt, L_HUMAN_dt,
                params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization
            )
        
        rs_rate_dt = compute_research_effort(
            experiment_compute_dt, coding_labor_dt,
            params.alpha_experiment_capacity, params.rho_experiment_capacity, params.experiment_compute_exponent, initial_aggregate_research_taste
        )
        # logger.info(f"rs_rate_dt: {rs_rate_dt}, rs_rate_0: {rs_rate_0}, dt: {dt}")
        
        # Calculate RS''(0) using forward difference
        rs_rate_second_derivative = (rs_rate_dt - rs_rate_0) / dt
        # logger.info(f"Calculated rs_rate_second_derivative: {rs_rate_second_derivative:.6f}")
        
        # Avoid division by zero or very small denominators
        if abs(rs_rate_second_derivative) < cfg.PARAM_CLIP_MIN:
            logger.warning(f"Very small research stock second derivative: {rs_rate_second_derivative}, using fallback")
            # Use a reasonable fallback value
            return max(cfg.PARAM_CLIP_MIN, rs_rate_0)
        
        # Calculate initial research stock: RS(0) = (RS'(0))^2 / RS''(0)
        initial_research_stock = (rs_rate_0 ** 2) / rs_rate_second_derivative
        
        # Ensure the result is positive and finite
        if not np.isfinite(initial_research_stock) or initial_research_stock <= 0:
            logger.warning(f"Invalid calculated initial research stock: {initial_research_stock}, using fallback")
            return max(cfg.PARAM_CLIP_MIN, rs_rate_0)
        
        # Apply reasonable bounds
        initial_research_stock = max(cfg.PARAM_CLIP_MIN, initial_research_stock)
        
        # logger.info(f"Calculated initial research stock: RS(0) = {initial_research_stock:.6f} "
        #            f"(RS'(0) = {rs_rate_0:.6f}, RS''(0) = {rs_rate_second_derivative:.6f})")
        
        return initial_research_stock
        
    except Exception as e:
        logger.error(f"Error calculating initial research stock: {e}")
        # Fallback to a reasonable default
        return 1.0


@dataclass
class InitialConditions:
    """Container for initial model conditions"""
    start_time: float
    initial_progress: float
    initial_automation: float
    L_HUMAN: float
    inference_compute: float
    experiment_compute: float
    training_compute_growth_rate: float
    coding_labor: float
    research_effort: float
    research_stock: float

def compute_initial_conditions(time_series_data: TimeSeriesData, params: Parameters, 
                             initial_progress: float = 0.0) -> InitialConditions:
    """
    Compute all initial conditions needed for model calculations.
    This eliminates duplication across evaluate_anchor_constraint, estimate_parameters, 
    compute_model, and ProgressModel.
    
    Args:
        time_series_data: Input time series
        params: Model parameters
        initial_progress: Initial cumulative progress
        
    Returns:
        InitialConditions object with all computed initial values

    TODO: maybe modify this to use the passed automation model in coding labor calculation
    """
    start_time = time_series_data.time[0]
    
    # TODO: may need to change to log-space interpolation
    L_HUMAN = np.interp(start_time, time_series_data.time, time_series_data.L_HUMAN)
    inference_compute = np.interp(start_time, time_series_data.time, time_series_data.inference_compute)
    experiment_compute = np.interp(start_time, time_series_data.time, time_series_data.experiment_compute)
    training_compute_growth_rate = np.interp(start_time, time_series_data.time, time_series_data.training_compute_growth_rate)

    if params.human_only:
        initial_automation = 1.0
        initial_ai_research_taste = 0.0
        initial_aggregate_research_taste = 1.0
        coding_labor = compute_coding_labor(None, inference_compute, L_HUMAN, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization, human_only=True)
        research_effort = compute_research_effort(experiment_compute, coding_labor, params.alpha_experiment_capacity, params.rho_experiment_capacity, params.experiment_compute_exponent, initial_aggregate_research_taste)
    else:
        initial_automation = compute_automation_fraction(initial_progress, params)
        initial_ai_research_taste = compute_ai_research_taste(initial_progress, params)
        initial_aggregate_research_taste = compute_aggregate_research_taste(initial_ai_research_taste, median_to_top_gap=params.median_to_top_taste_multiplier)
        coding_labor = compute_coding_labor(
            initial_automation, inference_compute, L_HUMAN, 
            params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization
        )
    
        research_effort = compute_research_effort(
            experiment_compute, coding_labor, 
            params.alpha_experiment_capacity, params.rho_experiment_capacity, params.experiment_compute_exponent, initial_aggregate_research_taste
        )
    
    # Validate and fallback for research stock rate
    if not np.isfinite(research_effort) or research_effort <= 0:
        logger.warning(f"Invalid initial research stock rate ({research_effort}), using fallback 1.0")
        research_effort = 1.0
    
    # Calculate initial research stock
    research_stock = calculate_initial_research_stock(time_series_data, params, initial_progress)
    
    return InitialConditions(
        start_time=start_time,
        initial_progress=initial_progress,
        initial_automation=initial_automation,
        L_HUMAN=L_HUMAN,
        inference_compute=inference_compute,
        experiment_compute=experiment_compute,
        training_compute_growth_rate=training_compute_growth_rate,
        coding_labor=coding_labor,
        research_effort=research_effort,
        research_stock=research_stock
    )



def setup_model(time_series_data: TimeSeriesData, params: Parameters,
               initial_progress: float = 0.0) -> Tuple[Parameters, InitialConditions]:
    """
    Set up model parameters and compute initial conditions.
    This replaces the duplicated setup logic in estimate_parameters and compute_model.
    
    Args:
        time_series_data: Input time series
        params: Model parameters
        initial_progress: Initial cumulative progress
        
    Returns:
        Tuple of (params, initial_conditions)
    """
    # Compute initial conditions
    initial_conditions = compute_initial_conditions(time_series_data, params, initial_progress)
    
    return params, initial_conditions

def aut_frac_from_swe_multiplier(swe_multiplier: float, L_HUMAN: float, inference_compute: float, params: Parameters) -> float:
    """
    Compute automation fraction from swe multiplier.

    Solve for A in:
      (swe_multiplier)**params.parallel_penalty * compute_coding_labor(A, inference_compute, L_HUMAN, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization, human_only=True) = compute_coding_labor(A, inference_compute, L_HUMAN, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization)
    where p = params.rho_coding_labor.
    Returns A in (0, 1). If there are multiple solutions, return the lower one.

    """
    # Input validation
    if not all(np.isfinite([swe_multiplier, L_HUMAN, inference_compute])):
        logger.warning("Non-finite inputs to aut_frac_from_swe_multiplier")
        return 0.0
    
    if swe_multiplier <= 0 or L_HUMAN <= 0 or inference_compute < 0:
        logger.warning("Invalid inputs to aut_frac_from_swe_multiplier")
        return 0.0
    
    # Target value we want to achieve
    target_output = swe_multiplier**params.parallel_penalty * compute_coding_labor(0, inference_compute, L_HUMAN, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization, human_only=True)
    
    # Define the objective function to minimize
    def objective(A_candidate):
        """Return the difference between target and actual cognitive output"""
        try:
            actual_output = compute_coding_labor(
                A_candidate, inference_compute, L_HUMAN, 
                params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization
            )
            return actual_output - target_output
        except Exception as e:
            logger.warning(f"Error in objective function: {e}")
            return float('inf')
    
    # Use bounds slightly inside (0, 1) to avoid numerical issues
    bounds = (cfg.AUTOMATION_FRACTION_CLIP_MIN, 1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN)
    
    try:
        # Use Brent's method for root finding since we have a single variable
        from scipy.optimize import brentq, minimize_scalar
        
        # Check if the function changes sign over the interval
        f_low = objective(bounds[0])
        f_high = objective(bounds[1])
        
        if f_low * f_high <= 0:
            # Sign change exists - we can find a root
            result = brentq(objective, bounds[0], bounds[1], xtol=1e-12, maxiter=100)
            result = np.clip(result, bounds[0], bounds[1])
            return float(result)
        else:
            # No sign change - target may not be achievable
            # Find the automation fraction that minimizes the absolute error
            result = minimize_scalar(
                lambda A: abs(objective(A)),
                bounds=bounds,
                method='bounded',
                options={'xatol': 1e-12, 'maxiter': 100}
            )
            
            if result.success:
                return float(np.clip(result.x, bounds[0], bounds[1]))
            else:
                raise RuntimeError("Optimization failed")
        
    except Exception as e:
        logger.warning(f"Root finding/optimization failed in aut_frac_from_swe_multiplier: {e}")
        
        # Fallback: use grid search to find the best approximation
        try:
            A_candidates = np.linspace(bounds[0], bounds[1], 1000)
            errors = [abs(objective(A)) for A in A_candidates]
            best_idx = np.argmin(errors)
            result = A_candidates[best_idx]
            
            logger.info(f"Used grid search fallback, error: {errors[best_idx]}")
            return float(result)
            
        except Exception as e2:
            logger.warning(f"Grid search fallback also failed: {e2}")
            # Return a reasonable default
            return 0.5


def solve_lower_anchor_via_automation_model(
    swe_multiplier: float,
    anchor_progress: float,
    L_HUMAN: float,
    inference_compute: float,
    params: Parameters,
) -> float:
    """
    Solve for the lower anchor automation fraction at the anchor progress such that,
    when initializing AutomationModel with anchors
        { anchor_progress: A_lower, params.progress_at_sc: params.automation_fraction_at_superhuman_coder },
    the implied coding-labor multiplier at anchor_progress matches swe_multiplier.

    The multiplier is defined on coding labor after applying parallel_penalty and normalization, i.e.
        coding_labor_with_AI = swe_multiplier**parallel_penalty * coding_labor_human_only.

    If the optimal-CES frontier is unavailable for given parameters, falls back to the simple CES formulation.
    Returns a clipped value in (cfg.AUTOMATION_FRACTION_CLIP_MIN, 1 - cfg.AUTOMATION_FRACTION_CLIP_MIN).
    """
    try:
        # Validate inputs
        if not all(np.isfinite([swe_multiplier, anchor_progress, L_HUMAN, inference_compute])):
            logger.warning("Non-finite inputs to solve_lower_anchor_via_automation_model")
            return 0.01
        if swe_multiplier <= 0 or L_HUMAN <= 0 or inference_compute < 0:
            logger.warning("Invalid inputs to solve_lower_anchor_via_automation_model")
            return 0.01

        # Ensure an upper anchor exists
        progress_at_sc = getattr(params, 'progress_at_sc', None)
        aut_at_sc = getattr(params, 'automation_fraction_at_superhuman_coder', None)
        if progress_at_sc is None or not np.isfinite(progress_at_sc) or aut_at_sc is None:
            logger.warning("Missing progress_at_sc or automation_fraction_at_superhuman_coder; falling back to direct solver")
            return aut_frac_from_swe_multiplier(swe_multiplier, L_HUMAN, inference_compute, params)

        # Target coding-labor ratio in parallel_penalty space
        target_ratio = float(np.power(swe_multiplier, params.parallel_penalty))

        # Baseline human-only coding labor (consistent with definition of multiplier)
        baseline = compute_coding_labor(
            0, inference_compute, L_HUMAN,
            params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization,
            human_only=True
        )
        if baseline <= 0 or not np.isfinite(baseline):
            logger.warning("Invalid baseline coding labor in anchor solver; using fallback")
            return aut_frac_from_swe_multiplier(swe_multiplier, L_HUMAN, inference_compute, params)

        logE = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM) * anchor_progress)

        def implied_ratio_for_anchor(a_lower: float) -> float:
            # Build temporary params with candidate anchors
            p = copy.deepcopy(params)
            # Clip candidate to safe open interval
            a_clipped = float(np.clip(a_lower, cfg.AUTOMATION_FRACTION_CLIP_MIN, 1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN))
            p.automation_anchors = {
                float(anchor_progress): a_clipped,
                float(progress_at_sc): float(np.clip(aut_at_sc, cfg.AUTOMATION_FRACTION_CLIP_MIN, 1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN)),
            }
            try:
                am = AutomationModel(p)
                H = float(L_HUMAN)
                C = float(inference_compute)
                L_opt = am.coding_labor_optimal_ces(H, C, logE, p)
                if L_opt is None or not np.isfinite(L_opt):
                    # Fallback to simple CES using the schedule's automation at the anchor (which equals a_lower)
                    A = am.get_automation_fraction(anchor_progress)
                    L_ai = compute_coding_labor(
                        A, inference_compute, L_HUMAN,
                        p.rho_coding_labor, p.parallel_penalty, p.coding_labor_normalization
                    )
                else:
                    # Match units with compute_coding_labor
                    L_ai = float((L_opt ** p.parallel_penalty) * p.coding_labor_normalization)
                if not np.isfinite(L_ai) or L_ai <= 0:
                    return 0.0
                return float(L_ai / baseline)
            except Exception as e:
                logger.warning(f"Error computing implied ratio for anchor {a_lower}: {e}")
                return 0.0

        def objective(a_lower: float) -> float:
            return implied_ratio_for_anchor(a_lower) - target_ratio

        # Bounds slightly inside (0, 1) to avoid numerical issues
        lo = float(cfg.AUTOMATION_FRACTION_CLIP_MIN)
        hi = float(1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN)

        try:
            from scipy.optimize import brentq, minimize_scalar

            f_lo = objective(lo)
            f_hi = objective(hi)
            if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi <= 0:
                root = brentq(objective, lo, hi, xtol=1e-8, maxiter=100)
                return float(np.clip(root, lo, hi))

            # If no sign change, minimize absolute error
            res = minimize_scalar(lambda x: abs(objective(x)), bounds=(lo, hi), method='bounded', options={'xatol': 1e-8, 'maxiter': 200})
            if getattr(res, 'success', False):
                return float(np.clip(res.x, lo, hi))
            else:
                raise RuntimeError("Anchor solver bounded minimization failed")
        except Exception as e:
            logger.warning(f"Anchor solver root-finding failed; using grid search fallback: {e}")
            try:
                grid = np.linspace(lo, hi, 512)
                errs = [abs(objective(a)) for a in grid]
                idx = int(np.argmin(errs))
                return float(grid[idx])
            except Exception as e2:
                logger.warning(f"Grid search fallback failed in anchor solver: {e2}")
                return 0.01
    except Exception as e:
        logger.warning(f"solve_lower_anchor_via_automation_model failed unexpectedly: {e}")
        return 0.01

def compute_automation_fraction(cumulative_progress: float, params: Parameters) -> float:
    """
    Interpolate automation fraction based on cumulative progress.

    - If params.automation_interp_type == "exponential": perform log-space interpolation between two anchors.
    - If params.automation_interp_type == "linear": perform linear interpolation between two anchors.

    Extrapolates beyond the anchors and clips the result to [0, 1].

    Args:
        cumulative_progress: Current cumulative progress.
        params: Model parameters containing automation_anchors and automation_interp_type.

    Returns:
        Automation fraction in [0, 1].
    """
    automation_model = params.automation_model
    return automation_model.get_automation_fraction(cumulative_progress)

    if params.automation_anchors is None:
        assert False, "automation_anchors must be provided"
    
    # Input validation
    if not np.isfinite(cumulative_progress):
        logger.warning(f"Non-finite cumulative_progress: {cumulative_progress}")
        cumulative_progress = 0.0
    
    # Extract anchor points (progress -> automation_fraction mapping)
    anchor_points = list(params.automation_anchors.items())
    
    if len(anchor_points) != 2:
        logger.error(f"automation_anchors must contain exactly 2 points, got {len(anchor_points)}")
        assert False, "automation_anchors must contain exactly 2 points"
    
    # Sort by progress value to ensure consistent ordering
    anchor_points.sort(key=lambda x: x[0])
    (progress_1, automation_1), (progress_2, automation_2) = anchor_points
    
    # Validate anchor values
    if automation_1 <= 0.0 or automation_1 > 1.0:
        logger.warning(f"automation_fraction at progress {progress_1} must be in (0, 1), got {automation_1}")
        automation_1 = np.clip(automation_1, 1e-6, 1.0 - 1e-6)
    
    if automation_2 <= 0.0 or automation_2 > 1.0:
        logger.warning(f"automation_fraction at progress {progress_2} must be in (0, 1), got {automation_2}")
        automation_2 = np.clip(automation_2, 1e-6, 1.0 - 1e-6)
    
    if progress_1 >= progress_2:
        logger.error(f"Progress values must be distinct and ordered: {progress_1} >= {progress_2}")
        assert False, "Progress anchor points must be distinct"
    
    interp_type = getattr(params, 'automation_interp_type', cfg.DEFAULT_PARAMETERS['automation_interp_type'])
    if interp_type == "exponential":
        try:
            # Log-space interpolation: log(automation) = a * progress + b
            log_automation_1 = np.log(automation_1)
            log_automation_2 = np.log(automation_2)

            a = (log_automation_2 - log_automation_1) / (progress_2 - progress_1)
            b = log_automation_1 - a * progress_1

            log_automation = a * cumulative_progress + b
            automation_fraction = np.exp(log_automation)
        except (ValueError, OverflowError) as e:
            logger.warning(f"Numerical error in log-space interpolation: {e}")
            # Fallback to linear behavior if log-space fails
            slope = (automation_2 - automation_1) / (progress_2 - progress_1)
            automation_fraction = (
                automation_1 + slope * (cumulative_progress - progress_1)
                if cumulative_progress <= progress_2
                else automation_2 + slope * (cumulative_progress - progress_2)
            )
    else:
        # Linear interpolation/extrapolation between anchors
        slope = (automation_2 - automation_1) / (progress_2 - progress_1)
        if progress_1 <= cumulative_progress <= progress_2:
            t = (cumulative_progress - progress_1) / (progress_2 - progress_1)
            automation_fraction = automation_1 + t * (automation_2 - automation_1)
        elif cumulative_progress < progress_1:
            automation_fraction = automation_1 + slope * (cumulative_progress - progress_1)
        else:
            automation_fraction = automation_2 + slope * (cumulative_progress - progress_2)
    
    # Clip to [0, 1] as required
    return np.clip(automation_fraction, 0.0, 1.0)


def compute_ai_research_taste(cumulative_progress: float, params: Parameters) -> float:
    """
    Compute AI research taste based on cumulative progress using either sigmoid or exponential schedule.
    
    This models how AI research taste (ability to identify promising research directions)
    improves with cumulative progress.
    
    Args:
        cumulative_progress: Current cumulative progress.
        params: Model parameters containing AI research taste parameters.
    
    Returns:
        AI research taste in [0, 1].
    """
    # Input validation
    if not np.isfinite(cumulative_progress):
        logger.warning(f"Non-finite cumulative_progress in ai_research_taste: {cumulative_progress}")
        cumulative_progress = 0.0
    
    # Choose computation method based on schedule type
    # Map UI-level schedule options to internal logic
    ui_type = params.taste_schedule_type
    if ui_type == "SDs per effective OOM" or ui_type == "SDs per progress-year":
        return _compute_ai_research_taste_sd_per_progress(cumulative_progress, params)
    if params.taste_schedule_type == "sigmoid":
        return _compute_ai_research_taste_exponential(cumulative_progress, params)
    elif params.taste_schedule_type == "exponential":
        return _compute_ai_research_taste_exponential(cumulative_progress, params)
    elif params.taste_schedule_type == "sd_per_progress":
        return _compute_ai_research_taste_sd_per_progress(cumulative_progress, params)
    else:
        logger.warning(f"Unknown taste_schedule_type: {params.taste_schedule_type}, defaulting to sd_per_progress")
        return _compute_ai_research_taste_sd_per_progress(cumulative_progress, params)


def _compute_ai_research_taste_sigmoid(cumulative_progress: float, params: Parameters) -> float:
    """
    Sigmoid function for AI research taste: f(x) = L / (1 + e^(-k*(x-x0)))
    """
    # Extract sigmoid parameters
    L = params.ai_research_taste_at_superhuman_coder  # Upper asymptote
    # Midpoint parameter removed with legacy sigmoid
    x0 = None
    k = params.ai_research_taste_slope  # Slope parameter
    
    # Legacy sigmoid path deprecated; approximate with a smooth rise capped at L
    try:
        exponent = -k * cumulative_progress
        if exponent > cfg.SIGMOID_EXPONENT_CLAMP:
            ai_research_taste = 0.0
        elif exponent < -cfg.SIGMOID_EXPONENT_CLAMP:
            ai_research_taste = L
        else:
            ai_research_taste = L * (1 - np.exp(exponent))
            ai_research_taste = np.clip(ai_research_taste, 0.0, L)
    except (OverflowError, ValueError) as e:
        logger.warning(f"Numerical error in legacy taste fallback: {e}")
        ai_research_taste = float(np.clip(L * cumulative_progress, 0.0, L))
    
    return ai_research_taste


def _compute_ai_research_taste_exponential(cumulative_progress: float, params: Parameters) -> float:
    """
    Exponential function for AI research taste that passes through 
    (progress_at_sc, ai_research_taste_at_superhuman_coder) with logplot-slope ai_research_taste_slope.
    
    Formula: taste(x) = taste_at_sc * exp(slope * (x - progress_at_sc))
    where taste_at_sc = ai_research_taste_at_superhuman_coder
    """
    taste_at_sc = params.ai_research_taste_at_superhuman_coder
    progress_at_sc = params.progress_at_sc
    slope = params.ai_research_taste_slope
    
    # Handle case where progress_at_sc is None (bootstrap failed)
    if progress_at_sc is None:
        raise ValueError("progress_at_sc is None - bootstrap process failed or was not run. Cannot compute AI research taste.")
    
    try:
        # Calculate exponential: taste(x) = taste_at_sc * exp(slope * (x - progress_at_sc))
        exponent = slope * (cumulative_progress - progress_at_sc)
        
        # Handle extreme exponents to prevent overflow/underflow
        if exponent > cfg.SIGMOID_EXPONENT_CLAMP:

            ai_research_taste = min(cfg.AI_RESEARCH_TASTE_MAX, taste_at_sc * np.exp(cfg.SIGMOID_EXPONENT_CLAMP))
        elif exponent < -cfg.SIGMOID_EXPONENT_CLAMP:
            ai_research_taste = max(0.0, taste_at_sc * np.exp(-cfg.SIGMOID_EXPONENT_CLAMP))
        else:
            ai_research_taste = taste_at_sc * np.exp(exponent)
            
    except (OverflowError, ValueError) as e:
        logger.warning(f"Numerical error in AI research taste exponential calculation: {e}")
        # Fallback: linear approximation around the anchor point
        if cumulative_progress <= progress_at_sc:
            # Use linear approximation for x <= progress_at_sc
            delta = cumulative_progress - progress_at_sc
            ai_research_taste = max(0.0, taste_at_sc + taste_at_sc * slope * delta)
        else:
            # Use saturated value for x > progress_at_sc
            ai_research_taste = min(cfg.AI_RESEARCH_TASTE_MAX, taste_at_sc * 2.0)
    
    # Clamp to valid range as a final safeguard
    return ai_research_taste


def _compute_ai_research_taste_sd_per_progress(cumulative_progress: float, params: Parameters) -> float:
    """
    Standard deviation per progress schedule for AI research taste.
    
    This schedule interprets the slope parameter as the number of standard deviations
    per progress unit in the underlying log-normal taste distribution. The curve
    passes through (progress_at_sc, ai_research_taste_at_superhuman_coder).
    
    Formula: taste(x) = taste_dist.get_taste_at_sd(slope * x + offset)
    where offset is computed to ensure the curve passes through the anchor point.
    """
    # Create taste distribution with default parameters
    try:
        taste_distribution = TasteDistribution()
    except Exception as e:
        logger.warning(f"Error creating TasteDistribution: {e}")
        # Fallback to exponential schedule
        return _compute_ai_research_taste_exponential(cumulative_progress, params)
    
    # Extract parameters
    taste_at_sc = params.ai_research_taste_at_superhuman_coder
    progress_at_sc = params.progress_at_sc
    slope = params.ai_research_taste_slope  # SD per progress unit
    
    # Handle case where progress_at_sc is None (bootstrap failed)
    if progress_at_sc is None:
        raise ValueError("progress_at_sc is None - bootstrap process failed or was not run. Cannot compute AI research taste.")
    
    try:
        # Compute offset so curve passes through (progress_at_sc, taste_at_sc)
        # We want: taste_at_sc = taste_distribution.get_taste_at_sd(slope * progress_at_sc + offset)
        # So: offset = taste_distribution.get_sd_of_taste(taste_at_sc) - slope * progress_at_sc
        
        # Clamp taste_at_sc to valid range to avoid log(0) issues
        taste_at_sc_clamped = max(1e-10, min(taste_at_sc, cfg.AI_RESEARCH_TASTE_MAX))
        
        target_sd = taste_distribution.get_sd_of_taste(taste_at_sc_clamped)
        # Adjust offset to maintain the anchor point with unpenalized slope
        offset = target_sd - slope * progress_at_sc
        
        # Compute AI research taste at current progress
        current_sd = slope * cumulative_progress + offset
        current_sd_clamped = min(current_sd, cfg.AI_RESEARCH_TASTE_MAX_SD)

        ai_research_taste = taste_distribution.get_taste_at_sd(current_sd_clamped)
        
    except Exception as e:
        logger.warning(f"Error in SD per progress calculation: {e}")
        # Fallback: use linear approximation around anchor point
        if cumulative_progress <= progress_at_sc:
            # Linear decrease for progress < anchor
            delta = cumulative_progress - progress_at_sc
            ai_research_taste = max(0.0, taste_at_sc + taste_at_sc * slope * 0.1 * delta)
        else:
            # Linear increase for progress > anchor
            delta = cumulative_progress - progress_at_sc
            ai_research_taste = min(cfg.AI_RESEARCH_TASTE_MAX, taste_at_sc * (1 + slope * 0.1 * delta))
    
    # Clamp to valid range as a final safeguard
    return np.clip(ai_research_taste, cfg.AI_RESEARCH_TASTE_MIN, cfg.AI_RESEARCH_TASTE_MAX)


def compute_aggregate_research_taste(ai_research_taste: float, 
                                   top_percentile: float = cfg.TOP_PERCENTILE,
                                   median_to_top_gap: float = cfg.MEDIAN_TO_TOP_TASTE_MULTIPLIER,
                                   baseline_mean: float = cfg.AGGREGATE_RESEARCH_TASTE_BASELINE) -> float:
    """
    Compute aggregate research taste using the log-normal distribution with a *clip-and-keep* floor.
    
    This is a wrapper around TasteDistribution.get_mean_with_floor() to maintain backward compatibility.
    
    The research taste of individuals is modeled as T ~ LogNormal(μ, σ²).  A hard floor F
    (given by `ai_research_taste`) lifts every draw below F up to F while leaving the
    upper tail unchanged, i.e. we consider Y = max(T, F).  The aggregate research taste
    is therefore the closed-form expectation
        E[Y] = F·Φ(a) + exp(μ + σ²/2) · Φ(σ − a),  a = (ln F − μ)/σ,
    where Φ is the standard normal CDF.
    
    The distribution parameters (μ, σ) are recovered from three empirical anchors:
      • top_percentile (p): fraction classified as "top" researchers
      • median_to_top_gap (G): threshold taste ÷ median taste
      • baseline_mean (M): company-wide mean taste
    following the equations σ = ln(G)/zₚ with zₚ = Φ⁻¹(1 − p) and μ = ln(M) − σ²/2.
    
    Args:
        ai_research_taste: Floor value F for research taste (any draw below this is lifted)
        top_percentile: Fraction of researchers classed as "top" (default from config)
        median_to_top_gap: Ratio of threshold taste to median taste (default from config)
        baseline_mean: Company-wide mean taste (default from config)
    
    Returns:
        Mean taste after clipping, E[max(T, F)].
    """
    # Input validation
    if not np.isfinite(ai_research_taste):
        logger.warning(f"Non-finite ai_research_taste: {ai_research_taste}")
        return cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK
    
    if ai_research_taste < 0:
        logger.warning(f"Negative ai_research_taste: {ai_research_taste}, using 0")
        ai_research_taste = 0.0
    
    # Validate distribution parameters
    if not (0 < top_percentile < 1):
        logger.warning(f"Invalid top_percentile: {top_percentile}, using default")
        top_percentile = cfg.TOP_PERCENTILE
    
    if median_to_top_gap <= 1:
        logger.warning(f"Invalid median_to_top_gap: {median_to_top_gap}, using default")
        median_to_top_gap = cfg.MEDIAN_TO_TOP_TASTE_MULTIPLIER
    
    if baseline_mean <= 0:
        logger.warning(f"Invalid baseline_mean: {baseline_mean}, using default")
        baseline_mean = cfg.AGGREGATE_RESEARCH_TASTE_BASELINE
    
    try:
        # Create taste distribution and use its get_mean_with_floor method
        taste_distribution = TasteDistribution(top_percentile, median_to_top_gap, baseline_mean)
        return taste_distribution.get_mean_with_floor(ai_research_taste)
    
    except Exception as e:
        logger.warning(f"Error computing aggregate research taste: {e}")
        # Fallback: return max of floor and baseline
        return max(ai_research_taste, cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK)


def progress_rate_at_time(t: float, state: List[float], time_series_data: TimeSeriesData, params: Parameters, 
                         initial_research_effort: Optional[float] = None, 
                         initial_research_stock: Optional[float] = None) -> List[float]:
    """
    Compute instantaneous rates for both progress and research stock.
    This is the RHS of the coupled differential equation system.
    
    Args:
        t: Current time
        state: [cumulative_progress, research_stock]
        time_series_data: Input time series
        params: Model parameters
        initial_research_effort: RS'(0) needed for software progress calculation
        initial_research_stock: RS(0) calculated initial research stock value
    
    Returns:
        [dP/dt, dRS/dt] - rates for both state variables
    """
    # Input validation
    if not np.isfinite(t):
        logger.warning(f"Non-finite time input: {t}")
        return [0.0, 0.0]
    
    if len(state) != 2:
        logger.warning(f"Invalid state vector length: {len(state)}, expected 2")
        return [0.0, 0.0]
    
    cumulative_progress, research_stock = state
    
    if not np.isfinite(cumulative_progress):
        logger.warning(f"Invalid cumulative progress: {cumulative_progress}")
        cumulative_progress = 0.0
    
    if not np.isfinite(research_stock) or research_stock <= 0:
        logger.warning(f"Invalid research stock: {research_stock}")
        if initial_research_stock is not None and initial_research_stock > 0:
            research_stock = max(cfg.PARAM_CLIP_MIN, initial_research_stock)
        else:
            # Fallback calculation if initial research stock not provided
            research_stock = calculate_initial_research_stock(time_series_data, params, cumulative_progress)
    
    # Validate time is within reasonable bounds
    time_min, time_max = time_series_data.time.min(), time_series_data.time.max()
    if t < time_min - cfg.TIME_EXTRAPOLATION_WINDOW or t > time_max + cfg.TIME_EXTRAPOLATION_WINDOW:  # Allow some extrapolation
        logger.warning(f"Time {t} far outside data range [{time_min}, {time_max}]")
    
    try:
        # Interpolate time series data to time t with validation
        # Use log-space interpolation for exponentially growing variables
        # This prevents scalloping on log plots and handles exponential growth better
        if np.all(time_series_data.L_HUMAN > 0):
            log_L_HUMAN = np.log(time_series_data.L_HUMAN)
            L_HUMAN = np.exp(np.interp(t, time_series_data.time, log_L_HUMAN))
        else:
            L_HUMAN = np.interp(t, time_series_data.time, time_series_data.L_HUMAN)
        
        if np.all(time_series_data.inference_compute > 0):
            log_inference_compute = np.log(time_series_data.inference_compute)
            inference_compute = np.exp(np.interp(t, time_series_data.time, log_inference_compute))
        else:
            inference_compute = np.interp(t, time_series_data.time, time_series_data.inference_compute)
        
        if np.all(time_series_data.experiment_compute > 0):
            log_experiment_compute = np.log(time_series_data.experiment_compute)
            experiment_compute = np.exp(np.interp(t, time_series_data.time, log_experiment_compute))
        else:
            experiment_compute = np.interp(t, time_series_data.time, time_series_data.experiment_compute)
        
        training_compute_growth_rate = np.interp(t, time_series_data.time, time_series_data.training_compute_growth_rate)
        
        # Validate interpolated values
        if not all(np.isfinite([L_HUMAN, inference_compute, experiment_compute, training_compute_growth_rate])):
            logger.warning(f"Non-finite interpolated values at t={t}")
            return [0.0, 0.0]
        
        # Ensure non-negative values
        L_HUMAN = max(0.0, L_HUMAN)
        inference_compute = max(0.0, inference_compute)
        experiment_compute = max(0.0, experiment_compute)
        training_compute_growth_rate = max(0.0, training_compute_growth_rate)
        
        if params.human_only:
            automation_fraction = 0.0
            aggregate_research_taste = 1.0
            coding_labor = compute_coding_labor(None, inference_compute, L_HUMAN, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization, human_only=True)  
        else:
            # Compute automation fraction from cumulative progress
            automation_fraction = compute_automation_fraction(cumulative_progress, params)
            if not (0 <= automation_fraction <= 1):
                logger.warning(f"Invalid automation fraction {automation_fraction} at progress {cumulative_progress}")
                automation_fraction = np.clip(automation_fraction, 0.0, 1.0)
            
            # Compute AI research taste and aggregate research taste
            ai_research_taste = compute_ai_research_taste(cumulative_progress, params)
            aggregate_research_taste = compute_aggregate_research_taste(ai_research_taste, median_to_top_gap=params.median_to_top_taste_multiplier)
            
            # Compute cognitive output with validation
            if getattr(params, 'coding_labor_mode', 'simple_ces') == 'optimal_ces':
                # Map model quantities to H, C, E (E is effective compute)
                H = float(L_HUMAN)
                C = float(inference_compute)
                logE = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM) * cumulative_progress)
                try:
                    # Build or reuse AutomationModel for frontier (uses current anchors)
                    automation_model = params.automation_model
                    L_opt = automation_model.coding_labor_optimal_ces(H, C, logE, params)
                    # Match units with compute_coding_labor: apply parallel_penalty and normalization
                    coding_labor = float((L_opt ** params.parallel_penalty) * params.coding_labor_normalization)
                except Exception as e:
                    assert False, "Falling back to simple CES due to optimal_ces error"
                    logger.warning(f"Falling back to simple CES due to optimal_ces error: {e}")
                    coding_labor = compute_coding_labor(automation_fraction, inference_compute, L_HUMAN, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization)
            else:
                coding_labor = compute_coding_labor(
                    automation_fraction, inference_compute, L_HUMAN, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization
                )
        
        if not np.isfinite(coding_labor) or coding_labor < 0:
            logger.warning(f"Invalid cognitive output: {coding_labor}")
            return [0.0, 0.0]
        
        # Compute research stock rate (dRS/dt) with validation, now named research effort
        research_effort = compute_research_effort(
            experiment_compute, coding_labor, params.alpha_experiment_capacity, params.rho_experiment_capacity, params.experiment_compute_exponent, aggregate_research_taste
        )
        
        if not np.isfinite(research_effort) or research_effort < 0:
            logger.warning(f"Invalid research stock rate: {research_effort}")
            return [0.0, 0.0]
        
        # Compute software progress rate using research stock formulation
        if initial_research_effort is None or initial_research_effort <= 0:
            logger.warning("No valid initial research stock rate provided, using fallback")
            # Fallback: use current rate as approximation
            software_progress_rate = research_effort
        else:
            software_progress_rate = compute_software_progress_rate(
                research_stock, research_effort, 
                initial_research_stock, initial_research_effort,
                params.r_software
            )
        
        if not np.isfinite(software_progress_rate) or software_progress_rate < 0:
            logger.warning(f"Invalid software progress rate: {software_progress_rate}")
            return [0.0, 0.0]
        
        # Compute overall progress rate (dP/dt)
        overall_rate = compute_overall_progress_rate(
            software_progress_rate, training_compute_growth_rate
        )
        
        # Final validation
        if not np.isfinite(overall_rate) or overall_rate < 0:
            logger.warning(f"Invalid overall progress rate: {overall_rate}")
            return [0.0, 0.0]
        
        # Cap extremely large rates to prevent numerical issues
        if overall_rate > cfg.MAX_NORMALIZED_PROGRESS_RATE:
            logger.warning(f"Very large progress rate {overall_rate}, capping to {cfg.MAX_NORMALIZED_PROGRESS_RATE}")
            overall_rate = cfg.MAX_NORMALIZED_PROGRESS_RATE
        
        logger.debug(f"t={t:.2f}, progress={cumulative_progress:.3f}, research_stock={research_stock:.3f}, "
                    f"automation={automation_fraction:.3f}, dP/dt={overall_rate:.3f}, dRS/dt={research_effort:.3f}")
        
        return [overall_rate, research_effort]
        
    except Exception as e:
        logger.error(f"Error computing rates at t={t}, state={state}: {e}")
        return [0.0, 0.0]  # Return zero rates on any error


def integrate_progress(time_range: List[float], initial_progress: float, time_series_data: TimeSeriesData, 
                      params: Parameters, direction: str = 'forward') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the coupled differential equation system with robust fallback methods:
    d(progress)/dt = progress_rate(t, progress, research_stock)
    d(research_stock)/dt = research_effort(t, progress, research_stock)
    
    Args:
        time_range: [start_time, end_time]
        initial_progress: Initial cumulative progress
        time_series_data: Input time series
        params: Model parameters
        direction: 'forward' or 'backward'
    
    Returns:
        Tuple of (times, cumulative_progress_values, research_stock_values)
    """
    # Use helper function to get initial research stock
    initial_conditions = compute_initial_conditions(time_series_data, params, initial_progress)
    initial_research_effort = initial_conditions.research_effort
    initial_research_stock = initial_conditions.research_stock
    
    def ode_func(t, y):
        try:
            rates = progress_rate_at_time(t, y, time_series_data, params, initial_research_effort, initial_research_stock)
            # Validate the rates
            if len(rates) != 2 or not all(np.isfinite(rate) and rate >= 0 for rate in rates):
                logger.warning(f"Invalid rates {rates} at time {t}, state {y}")
                return [0.0, 0.0]  # Stop integration if rates become invalid
            return rates
        except Exception as e:
            logger.warning(f"Error computing rates at t={t}, state={y}: {e}")
            return [0.0, 0.0]  # Fail gracefully
    
    def ode_func_bounded(t, y):
        """ODE function with bounds checking to prevent extreme values"""
        try:
            # Prevent progress from becoming extremely large
            if y[0] > cfg.PROGRESS_ODE_CLAMP_MAX:
                logger.warning(f"Progress {y[0]} too large at time {t}, clamping")
                y[0] = cfg.PROGRESS_ODE_CLAMP_MAX
            
            # Prevent research stock from going negative or becoming extremely large
            if y[1] <= 0:
                y[1] = max(1e-6, initial_research_stock)
            elif y[1] > cfg.RESEARCH_STOCK_ODE_CLAMP_MAX:
                logger.warning(f"Research stock {y[1]} too large at time {t}, clamping")
                y[1] = cfg.RESEARCH_STOCK_ODE_CLAMP_MAX
            
            rates = progress_rate_at_time(t, y, time_series_data, params, initial_research_effort, initial_research_stock)
            
            # Clamp rates to reasonable bounds
            for i in range(len(rates)):
                if not np.isfinite(rates[i]):
                    rates[i] = 0.0
                elif rates[i] < 0:
                    rates[i] = 0.0
            
            return rates
        except Exception as e:
            logger.warning(f"Error in bounded ODE function at t={t}: {e}")
            return [0.0, 0.0]
    
    t_start, t_end = time_range
    if direction == 'backward':
        t_start, t_end = t_end, t_start
    
    # Validate initial conditions
    if not np.isfinite(initial_progress):
        logger.warning(f"Invalid initial progress {initial_progress}, using fallback")
        initial_progress = 0.0
    
    # Use research stock from initial conditions
    initial_state = [initial_progress, initial_conditions.research_stock]
    
    # Try multiple integration methods with increasing robustness
    # Try stiff-friendly methods first to avoid excessive time spent on RK methods
    methods_to_try = [
        ('Radau', {'rtol': 1e-3, 'atol': 1e-5}),      # Implicit method for stiff problems
        ('RK23', {'rtol': 1e-4, 'atol': 1e-6}),       # Lower order explicit method
        ('RK45', {'rtol': 1e-4, 'atol': 1e-6}),       # Relaxed precision
        ('RK45', {'rtol': 1e-6, 'atol': 1e-8}),       # Higher precision
        ('DOP853', {'rtol': 1e-3, 'atol': 1e-5})      # High-order explicit method
    ]
    
    sol = None
    for method, tolerances in methods_to_try:
        try:
            logger.debug(f"Trying integration with method {method}, tolerances {tolerances}")
            
            # Use bounded ODE function for more robust integration
            sol = integrate.solve_ivp(
                ode_func_bounded,
                [t_start, t_end], 
                initial_state,
                method=method,
                dense_output=True,
                **tolerances,
                max_step=cfg.ODE_MAX_STEP  # Limit step size for stability
            )
            
            if sol.success:
                logger.debug(f"Integration succeeded with method {method}")
                
                # Log ODE step size information
                if cfg.ODE_STEP_SIZE_LOGGING and hasattr(sol, 't') and len(sol.t) > 1:
                    step_sizes = np.diff(sol.t)
                    min_step = np.min(step_sizes)
                    max_step = np.max(step_sizes)
                    mean_step = np.mean(step_sizes)
                    total_steps = len(sol.t)
                    
                    logger.info(f"ODE step size stats for {method}: "
                              f"min={min_step:.2e}, max={max_step:.2e}, "
                              f"mean={mean_step:.2e}, total_steps={total_steps}")
                    
                    # Log if step sizes are very small (potential stiffness indicator)
                    if min_step < cfg.ODE_SMALL_STEP_THRESHOLD:
                        logger.warning(f"Very small step sizes detected with {method}: "
                                     f"min_step={min_step:.2e} - this may indicate stiff ODE")
                    
                    # Log if step sizes vary significantly (potential instability)
                    step_variation = max_step / min_step if min_step > 0 else float('inf')
                    if step_variation > cfg.ODE_STEP_VARIATION_THRESHOLD:
                        logger.warning(f"Large step size variation with {method}: "
                                     f"max/min ratio={step_variation:.1f} - potential instability")
                
                break
            else:
                logger.warning(f"Integration with {method} failed: {sol.message}")
                
        except Exception as e:
            logger.warning(f"Integration method {method} raised exception: {e}")
            continue
    
    # Create dense output over time range
    try:
        times = np.linspace(min(time_range), max(time_range), cfg.DENSE_OUTPUT_POINTS)
        solution_values = sol.sol(times)
        progress_values = solution_values[0]
        research_stock_values = solution_values[1]
        
        # Validate results
        if not all(np.isfinite(progress_values)):
            logger.warning("Non-finite values in progress integration result")
            # Replace non-finite values with interpolation
            finite_mask = np.isfinite(progress_values)
            if np.any(finite_mask):
                progress_values = np.interp(times, times[finite_mask], progress_values[finite_mask])
            else:
                raise ValueError("All progress integration results are non-finite")
        
        if not all(np.isfinite(research_stock_values)):
            logger.warning("Non-finite values in research stock integration result")
            # Replace non-finite values with interpolation
            finite_mask = np.isfinite(research_stock_values)
            if np.any(finite_mask):
                research_stock_values = np.interp(times, times[finite_mask], research_stock_values[finite_mask])
            else:
                raise ValueError("All research stock integration results are non-finite")
        
        # Note: Negative progress values are allowed
        
        if np.any(research_stock_values <= 0):
            logger.warning("Non-positive research stock values detected, clamping to minimum")
            research_stock_values = np.maximum(research_stock_values, 1e-6)
        
        return times, progress_values, research_stock_values
        
    except Exception as e:
        logger.error(f"Error creating dense output: {e}")
        raise RuntimeError(f"Integration succeeded but dense output failed: {e}")


class ProgressModel:
    """Main class for AI progress modeling"""
    
    def __init__(self, params: Parameters, time_series_data: TimeSeriesData):
        """
        Time series data is from the capabilities input spreadsheet.
        """
        self.params = params
        self.data = time_series_data
        self.human_only_results = {}
        self.results = {}
        self.horizon_trajectory = None
        
        # Initialize flags for horizon trajectory anchor fix
        self._horizon_uses_shifted_form = False
        self._horizon_params = None
        
        # Initialize taste distribution for working with research taste
        self.taste_distribution = TasteDistribution(median_to_top_gap=self.params.median_to_top_taste_multiplier)
    
    def estimate_horizon_trajectory(self, human_only_times: np.ndarray, human_only_progress: np.ndarray, anchor_progress_rate: float):
        """
        Estimate horizon trajectory by fitting to log(p80_horizon_length) vs progress.
        Uses provided human-only progress trajectory to get progress values at model release dates.
        
        The functional form depends on horizon_extrapolation_type:
        - "exponential": linear regression on log(horizon) vs progress
        - "decaying doubling time": decaying doubling time functional form
        
        Args:
            human_only_times: Time array from human-only trajectory computation
            human_only_progress: Progress array from human-only trajectory computation
            anchor_progress_rate: Progress rate at anchor time (progress units per time unit),
                                 used to convert present_doubling_time from time units to progress units
            
        Returns:
            Function that maps progress to horizon length
        """
        
        # Load METR benchmark data
        try:
            with open('benchmark_results.yaml', 'r') as f:
                benchmark_data = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error("benchmark_results.yaml file not found")
            return None
        except Exception as e:
            logger.error(f"Error reading benchmark_results.yaml: {e}")
            return None
        
        # Extract (progress, horizon) pairs from METR data
        progress_horizon_pairs = []
        
        for model_name, model_info in benchmark_data['results'].items():
            # Convert release date to decimal year
            release_date_obj = model_info['release_date']
            try:
                # Handle both string and date objects
                if isinstance(release_date_obj, str):
                    release_date = datetime.strptime(release_date_obj, '%Y-%m-%d').date()
                else:
                    release_date = release_date_obj
                
                decimal_year = release_date.year + (release_date.timetuple().tm_yday - 1) / 365.25
            except (ValueError, AttributeError) as e:
                logger.warning(f"Could not parse release date for {model_name}: {release_date_obj} ({e})")
                continue
            
            # Interpolate progress value at the release date using human-only trajectory results
            if decimal_year >= human_only_times.min() and decimal_year <= human_only_times.max():
                interpolated_progress = np.interp(decimal_year, human_only_times, human_only_progress)
            elif decimal_year < human_only_times.min():
                interpolated_progress = human_only_progress[0]
            else:
                interpolated_progress = human_only_progress[-1]
            
            # Extract p80_horizon_length estimates for each agent configuration
            for agent_name, agent_data in model_info['agents'].items():
                if 'p80_horizon_length' in agent_data:
                    p80_data = agent_data['p80_horizon_length']
                    p80_estimate = p80_data.get('estimate')
                    
                    if p80_estimate is not None and p80_estimate > 0:  # Must be positive for log transform
                        progress_horizon_pairs.append((interpolated_progress, p80_estimate))
        
        if len(progress_horizon_pairs) < 2:
            logger.error("Not enough valid (progress, horizon) pairs for regression")
            return None
        
        # Convert to arrays for regression
        progress_values = np.array([pair[0] for pair in progress_horizon_pairs])
        horizon_values = np.array([pair[1] for pair in progress_horizon_pairs])
        
        # Fit functional form based on horizon_extrapolation_type
        log_horizon_values = np.log(horizon_values)
        
        # Define fitting functions
        def linear_func(x, a, b):
            return a * x + b
        
        def decaying_doubling_time_func(t, H_0, A_0, T_0):
            """Decaying doubling time function with numerical safeguards.
            Supports A_0 in (-inf, 1), excluding A_0 == 0. Handles both accelerating (A_0>0) and decelerating (A_0<0) cases.
            """
            try:
                # Handle scalar vs array inputs
                is_scalar = np.isscalar(t)
                t_arr = np.atleast_1d(t)
                
                # Ensure parameters are within valid ranges
                # A_0 must be < 1 and != 0; H_0 and T_0 must be > 0
                if T_0 <= 0 or H_0 <= 0 or A_0 >= 1 or A_0 == 0:
                    fallback = np.full_like(t_arr, np.log(1e12))
                    return fallback[0] if is_scalar else fallback
                
                # Calculate the base term (1 - A_0 * t / T_0)
                base_term = 1 - A_0 * t_arr / T_0
                # Clamp to small positive to avoid domain errors
                base_term = np.maximum(base_term, 1e-12)
                
                # Calculate the exponent
                log_denominator = np.log(1 - A_0)
                if not np.isfinite(log_denominator) or abs(log_denominator) < 1e-12:
                    fallback = np.full_like(t_arr, np.log(1e12))
                    return fallback[0] if is_scalar else fallback
                exponent = np.log(2) / log_denominator
                
                # Calculate the result
                result = H_0 * (base_term ** exponent)
                
                # Check for invalid results
                if np.any(~np.isfinite(result)) or np.any(result <= 0):
                    fallback = np.full_like(t_arr, np.log(1e12))
                    return fallback[0] if is_scalar else fallback
                
                log_result = np.log(result)
                return log_result[0] if is_scalar else log_result
                
            except (ValueError, ZeroDivisionError, OverflowError):
                t_arr = np.atleast_1d(t)
                fallback = np.full_like(t_arr, np.log(1e12))
                return fallback[0] if np.isscalar(t) else fallback
        
        try:
            if self.params.horizon_extrapolation_type == "exponential":
                # Check if manual parameters are provided
                if self.params.present_horizon is not None:
                    # Use manual fitting with anchor point
                    # Get progress at present_day
                    anchor_progress = np.interp(self.params.present_day, human_only_times, human_only_progress)
                    
                    # If present_doubling_time is provided, use it to calculate slope
                    if self.params.present_doubling_time is not None:
                        # Convert present_doubling_time from time units to progress units
                        # doubling_time_in_progress_units = doubling_time_in_time_units * progress_rate
                        doubling_time_in_progress_units = self.params.present_doubling_time * anchor_progress_rate
                        # slope = log(2) / doubling_time (in progress units)
                        slope = np.log(2) / doubling_time_in_progress_units
                    else:
                        # Optimize slope using data, but fix intercept using anchor point
                        def fit_slope_only(slope_val):
                            intercept_val = np.log(self.params.present_horizon) - slope_val * anchor_progress
                            predicted = linear_func(progress_values, slope_val, intercept_val)
                            return np.sum((log_horizon_values - predicted)**2)
                        
                        result = optimize.minimize_scalar(fit_slope_only, bounds=(-10, 10), method='bounded')
                        slope = result.x
                    
                    # Calculate intercept from anchor point: log(present_horizon) = slope * anchor_progress + intercept
                    intercept = np.log(self.params.present_horizon) - slope * anchor_progress
                    
                    logger.info(f"Manual exponential horizon trajectory: log(horizon) = {slope:.6f} * progress + {intercept:.6f}")
                    logger.info(f"Using anchor point: time={self.params.present_day}, progress={anchor_progress:.4f}, horizon={self.params.present_horizon:.4f}")
                    if self.params.present_doubling_time is not None:
                        logger.info(f"Anchor doubling time (time units): {self.params.present_doubling_time:.4f}, converted to progress units: {doubling_time_in_progress_units:.4f} (using progress rate: {anchor_progress_rate:.4f})")
                else:
                    # Use automatic curve fitting
                    popt, pcov = optimize.curve_fit(linear_func, progress_values, log_horizon_values)
                    slope, intercept = popt
                    
                    logger.info(f"Fitted exponential horizon trajectory: log(horizon) = {slope:.6f} * progress + {intercept:.6f}")
                    logger.info(f"R-squared: {1 - np.sum((log_horizon_values - linear_func(progress_values, *popt))**2) / np.sum((log_horizon_values - np.mean(log_horizon_values))**2):.4f}")
                
                # Create horizon trajectory function: progress -> horizon
                def horizon_trajectory(progress):
                    """Map progress to horizon length using fitted exponential model"""
                    return np.exp(slope * progress + intercept)
                
                # Calculate progress level where horizon reaches the target horizon
                # Target depends on include_gap (formerly benchmarks_and_gaps_mode)
                _include_gap_flag = False
                try:
                    _inc = getattr(self.params, 'include_gap', 'no gap')
                    if isinstance(_inc, str):
                        _include_gap_flag = _inc.strip().lower() == 'gap'
                    else:
                        _include_gap_flag = bool(_inc)
                except Exception:
                    _include_gap_flag = False
                target_horizon = self.params.pre_gap_sc_time_horizon if _include_gap_flag else self.params.sc_time_horizon_minutes
                if target_horizon > 0:
                    try:
                        # Solve: target_horizon = exp(slope * progress + intercept)
                        # Therefore: progress = (log(target_horizon) - intercept) / slope
                        calculated_progress_at_sc = (np.log(target_horizon) - intercept) / slope
                        # If in gap-included mode, add the gap (specified in anchor-progress-years)
                        # Convert anchor-progress-years to progress units using anchor_progress_rate
                        if _include_gap_flag:
                            try:
                                gap_anchor_years = float(self.params.gap_years)
                                gap_progress_units = float(anchor_progress_rate) * gap_anchor_years
                            except Exception:
                                gap_progress_units = float(self.params.gap_years)
                            calculated_progress_at_sc = calculated_progress_at_sc + gap_progress_units
                            try:
                                year_label = int(self.params.present_day) if getattr(self.params, 'present_day', None) is not None else 'anchor'
                            except Exception:
                                year_label = 'anchor'
                            logger.info(
                                f"Gap-included mode: using pre-gap SC horizon {self.params.pre_gap_sc_time_horizon} and "
                                f"adding gap {self.params.gap_years} {year_label}-progress-years (~{gap_progress_units:.6f} progress units)"
                            )
                        self.params.progress_at_sc = calculated_progress_at_sc
                        logger.info(f"Progress level at target horizon ({target_horizon} min): {calculated_progress_at_sc:.4f}")
                    except (ValueError, ZeroDivisionError) as e:
                        logger.warning(f"Could not calculate progress at sc_time_horizon_minutes: {e}")
                        self.params.progress_at_sc = None
            
            elif self.params.horizon_extrapolation_type == "decaying doubling time":
                # Determine approach based on whether present_doubling_time is specified
                # If present_doubling_time is specified, we MUST use the shifted function approach
                # to ensure the doubling time at the anchor point equals present_doubling_time
                if self.params.present_doubling_time is not None or self.params.present_horizon is not None:
                    # Use shifted function approach
                    self._horizon_uses_shifted_form = True
                    # Get progress at present_day
                    anchor_progress = np.interp(self.params.present_day, human_only_times, human_only_progress)
                    
                    # Determine what parameters we have and what we need to optimize
                    params_to_optimize = []
                    fixed_params = {}
                    
                    # Handle H_0 (present_horizon)
                    if self.params.present_horizon is not None:
                        fixed_params['H_0'] = self.params.present_horizon
                    else:
                        params_to_optimize.append('H_0')
                    
                    # Handle T_0 (present_doubling_time)
                    if self.params.present_doubling_time is not None:
                        # Convert present_doubling_time from time units to progress units
                        # doubling_time_in_progress_units = doubling_time_in_time_units * progress_rate
                        doubling_time_in_progress_units = self.params.present_doubling_time * anchor_progress_rate
                        fixed_params['T_0'] = doubling_time_in_progress_units
                    else:
                        params_to_optimize.append('T_0')
                    
                    # Handle A_0 (doubling_difficulty_growth_rate converted to decay_rate)
                    if self.params.doubling_difficulty_growth_rate is not None:
                        fixed_params['A_0'] = 1.0 - self.params.doubling_difficulty_growth_rate
                    else:
                        params_to_optimize.append('A_0')
                    
                    if len(params_to_optimize) == 0:
                        # All parameters specified (Case 8)
                        H_0 = self.params.present_horizon
                        T_0 = doubling_time_in_progress_units
                        A_0 = 1.0 - self.params.doubling_difficulty_growth_rate
                        logger.info(f"Manual decaying doubling time: All parameters specified")
                    elif len(params_to_optimize) == 1:
                        # Optimize one parameter
                        param_name = params_to_optimize[0]
                        
                        def fit_single_param(param_val):
                            if param_name == 'A_0' and (param_val >= 1 or abs(param_val) < 1e-6):
                                return 1e6
                            if param_name in ['H_0', 'T_0'] and param_val <= 0:
                                return 1e6
                            
                            # Reconstruct full parameter set
                            param_dict = fixed_params.copy()
                            param_dict[param_name] = param_val
                            
                            H_0_val = param_dict['H_0']
                            T_0_val = param_dict['T_0']
                            A_0_val = param_dict['A_0']
                            
                            # Use shifted function: horizon(t) = H_0 * (1 - A_0 * (t - anchor_progress) / T_0)^exponent
                            def shifted_func(t_vals):
                                try:
                                    t_shifted = t_vals - anchor_progress
                                    base_term = 1 - A_0_val * t_shifted / T_0_val
                                    # Clamp to small positive to avoid domain errors
                                    base_term = np.maximum(base_term, 1e-12)
                                    
                                    # Guard against A_0 near 0 or >=1
                                    log_denominator = np.log(1 - A_0_val)
                                    if not np.isfinite(log_denominator) or abs(log_denominator) < 1e-12:
                                        return np.full_like(t_vals, np.log(1e12))
                                    exponent = np.log(2) / log_denominator
                                    result = H_0_val * (base_term ** exponent)
                                    
                                    if np.any(~np.isfinite(result)) or np.any(result <= 0):
                                        return np.full_like(t_vals, np.log(1e12))
                                    
                                    return np.log(result)
                                except:
                                    return np.full_like(t_vals, np.log(1e12))
                            
                            predicted = shifted_func(progress_values)
                            return np.sum((log_horizon_values - predicted)**2)
                        
                        # Set bounds based on parameter type
                        if param_name == 'A_0':
                            bounds = (-0.999, 0.999)
                            initial_guess = 0.05
                        elif param_name == 'H_0':
                            bounds = (1e-6, 1e6)
                            initial_guess = 0.00001
                        elif param_name == 'T_0':
                            bounds = (1e-6, 100.0)
                            initial_guess = 1.35
                        
                        result = optimize.minimize_scalar(fit_single_param, bounds=bounds, method='bounded')
                        
                        # Extract optimized parameters
                        param_dict = fixed_params.copy()
                        param_dict[param_name] = result.x
                        
                        H_0 = param_dict['H_0']
                        T_0 = param_dict['T_0']
                        A_0 = param_dict['A_0']
                        
                        logger.info(f"Optimized {param_name}: {result.x:.6f}")
                    else:
                        # Optimize multiple parameters
                        def fit_multiple_params(params_array):
                            # Reconstruct full parameter set
                            param_dict = fixed_params.copy()
                            for i, param_name in enumerate(params_to_optimize):
                                param_dict[param_name] = params_array[i]
                            
                            H_0_val = param_dict['H_0']
                            T_0_val = param_dict['T_0']
                            A_0_val = param_dict['A_0']
                            
                            if A_0_val >= 1 or abs(A_0_val) < 1e-6 or H_0_val <= 0 or T_0_val <= 0:
                                return 1e6
                            
                            # Use shifted function
                            def shifted_func(t_vals):
                                try:
                                    t_shifted = t_vals - anchor_progress
                                    base_term = 1 - A_0_val * t_shifted / T_0_val
                                    # Clamp to small positive to avoid domain errors
                                    base_term = np.maximum(base_term, 1e-12)
                                    
                                    log_denominator = np.log(1 - A_0_val)
                                    if not np.isfinite(log_denominator) or abs(log_denominator) < 1e-12:
                                        return np.full_like(t_vals, np.log(1e12))
                                    exponent = np.log(2) / log_denominator
                                    result = H_0_val * (base_term ** exponent)
                                    
                                    if np.any(~np.isfinite(result)) or np.any(result <= 0):
                                        return np.full_like(t_vals, np.log(1e12))
                                    
                                    return np.log(result)
                                except:
                                    return np.full_like(t_vals, np.log(1e12))
                            
                            predicted = shifted_func(progress_values)
                            return np.sum((log_horizon_values - predicted)**2)
                        
                        # Set up bounds and initial guesses
                        bounds = []
                        p0 = []
                        for param_name in params_to_optimize:
                            if param_name == 'A_0':
                                bounds.append((-0.999, 0.999))
                                p0.append(0.05)
                            elif param_name == 'H_0':
                                bounds.append((1e-6, 1e6))
                                p0.append(0.00001)
                            elif param_name == 'T_0':
                                bounds.append((1e-6, 100.0))
                                p0.append(1.35)
                        
                        result = optimize.minimize(fit_multiple_params, p0, bounds=bounds, method='L-BFGS-B')
                        
                        # Extract optimized parameters
                        param_dict = fixed_params.copy()
                        for i, param_name in enumerate(params_to_optimize):
                            param_dict[param_name] = result.x[i]
                        
                        H_0 = param_dict['H_0']
                        T_0 = param_dict['T_0']
                        A_0 = param_dict['A_0']
                        
                        logger.info(f"Optimized parameters: {', '.join([f'{name}={param_dict[name]:.6f}' for name in params_to_optimize])}")
                    
                    logger.info(f"Manual decaying doubling time horizon trajectory: H_0={H_0:.6f}, A_0={A_0:.6f}, T_0={T_0:.6f}")
                    logger.info(f"Using anchor point: time={self.params.present_day}, progress={anchor_progress:.4f}")
                    if self.params.present_horizon is not None:
                        logger.info(f"Anchor horizon: {self.params.present_horizon:.4f}")
                    if self.params.present_doubling_time is not None:
                        logger.info(f"Anchor doubling time (time units): {self.params.present_doubling_time:.4f}, converted to progress units: {doubling_time_in_progress_units:.4f} (using progress rate: {anchor_progress_rate:.4f})")
                    
                    # Store anchor_progress for use in horizon_trajectory function
                    anchor_progress_for_trajectory = anchor_progress
                else:
                    # Use automatic curve fitting (Cases 1-2: no anchor parameters specified)
                    self._horizon_uses_shifted_form = False
                    # These specific values are somewhat important for the optimization
                    H_0_init = 0.00001  
                    A_0_init = 0.05
                    T_0_init = 1.35
                    
                    popt, pcov = optimize.curve_fit(
                        decaying_doubling_time_func, 
                        progress_values, 
                        log_horizon_values,
                        p0=[H_0_init, A_0_init, T_0_init],
                        bounds=([1e-6, -0.999, 1e-6], [np.inf, 0.999, np.inf])  # Allow negative A_0 but exclude 0 implicitly via fit
                    )
                    H_0, A_0, T_0 = popt
                    
                    logger.info(f"Fitted decaying doubling time horizon trajectory: H_0={H_0:.6f}, A_0={A_0:.6f}, T_0={T_0:.6f}")
                    logger.info(f"R-squared: {1 - np.sum((log_horizon_values - decaying_doubling_time_func(progress_values, *popt))**2) / np.sum((log_horizon_values - np.mean(log_horizon_values))**2):.4f}")
                    
                    # For automatic fitting, no anchor_progress shift is used
                    anchor_progress_for_trajectory = None
                
                # Create horizon trajectory function: progress -> horizon
                def horizon_trajectory(progress):
                    """Map progress to horizon length using fitted decaying doubling time model"""
                    try:
                        # Handle scalar vs array inputs
                        is_scalar = np.isscalar(progress)
                        progress_arr = np.atleast_1d(progress)
                        
                        # Ensure parameters are within valid ranges
                        if T_0 <= 0 or H_0 <= 0 or A_0 >= 1 or A_0 == 0:
                            fallback = np.full_like(progress_arr, 1e12)
                            return fallback[0] if is_scalar else fallback
                        
                        # Use shifted form if we're in the manual parameter case (anchor_progress_for_trajectory is set)
                        if anchor_progress_for_trajectory is not None:
                            # Shifted form: horizon(t) = H_0 * (1 - A_0 * (t - anchor_progress) / T_0)^exponent
                            progress_shifted = progress_arr - anchor_progress_for_trajectory
                            base_term = 1 - A_0 * progress_shifted / T_0
                        else:
                            # Original form: horizon(t) = H_0 * (1 - A_0 * t / T_0)^exponent
                            base_term = 1 - A_0 * progress_arr / T_0
                        
                        # Clamp to small positive to avoid domain errors
                        base_term = np.maximum(base_term, 1e-12)
                        
                        # Calculate the exponent
                        log_denominator = np.log(1 - A_0)
                        if not np.isfinite(log_denominator) or abs(log_denominator) < 1e-12:
                            fallback = np.full_like(progress_arr, 1e12)
                            return fallback[0] if is_scalar else fallback
                        
                        exponent = np.log(2) / log_denominator
                        
                        # Calculate the result
                        result = H_0 * (base_term ** exponent)
                        
                        # Check for invalid results
                        if np.any(~np.isfinite(result)) or np.any(result <= 0):
                            fallback = np.full_like(progress_arr, 1e12)
                            return fallback[0] if is_scalar else fallback
                        
                        return result[0] if is_scalar else result
                        
                    except (ValueError, ZeroDivisionError, OverflowError):
                        progress_arr = np.atleast_1d(progress)
                        fallback = np.full_like(progress_arr, 1e12)
                        return fallback[0] if np.isscalar(progress) else fallback
                
                # Calculate progress level where horizon reaches the target horizon
                # Target depends on include_gap (formerly benchmarks_and_gaps_mode)
                _include_gap_flag = False
                try:
                    _inc = getattr(self.params, 'include_gap', 'no gap')
                    if isinstance(_inc, str):
                        _include_gap_flag = _inc.strip().lower() == 'gap'
                    else:
                        _include_gap_flag = bool(_inc)
                except Exception:
                    _include_gap_flag = False
                target_horizon = self.params.pre_gap_sc_time_horizon if _include_gap_flag else self.params.sc_time_horizon_minutes
                if target_horizon > 0:
                    try:
                        # Add numerical safeguards
                        if T_0 <= 0 or H_0 <= 0 or A_0 >= 1 or A_0 == 0:
                            logger.warning("Invalid parameters for progress_at_sc calculation")
                            self.params.progress_at_sc = None
                        elif target_horizon <= 0:
                            logger.warning("Invalid sc_time_horizon_minutes for calculation")
                            self.params.progress_at_sc = None
                        else:
                            # Check if the ratio is valid
                            ratio = target_horizon / H_0
                            if ratio <= 0:
                                logger.warning("Invalid ratio for progress_at_sc calculation")
                                self.params.progress_at_sc = None
                            else:
                                log_ratio = np.log(1-A_0) / np.log(2)
                                if not np.isfinite(log_ratio):
                                    logger.warning("Invalid log ratio for progress_at_sc calculation")
                                    self.params.progress_at_sc = None
                                else:
                                    ratio_term = ratio ** log_ratio
                                    if not np.isfinite(ratio_term):
                                        logger.warning("Invalid ratio_term for progress_at_sc calculation")
                                        self.params.progress_at_sc = None
                                    else:
                                        # Use shifted form if we're in the manual parameter case
                                        if anchor_progress_for_trajectory is not None:
                                            # Shifted form: sc_time_horizon_minutes = H_0 * (1 - A_0 * (progress - anchor_progress) / T_0)^exponent
                                            # progress = anchor_progress + T_0 * (1 - (sc_time_horizon_minutes / H_0)^(log(1-A_0)/log(2))) / A_0
                                            calculated_progress_at_sc = anchor_progress_for_trajectory + T_0 * (1 - ratio_term) / A_0
                                        else:
                                            # Original form: sc_time_horizon_minutes = H_0 * (1 - A_0 * progress / T_0)^exponent
                                            # progress = T_0 * (1 - (sc_time_horizon_minutes / H_0)^(log(1-A_0)/log(2))) / A_0
                                            calculated_progress_at_sc = T_0 * (1 - ratio_term) / A_0
                                        
                                        # If in gap-included mode, add the gap (specified in anchor-progress-years)
                                        # Convert anchor-progress-years to progress units using anchor_progress_rate
                                        if _include_gap_flag:
                                            try:
                                                gap_anchor_years = float(self.params.gap_years)
                                                gap_progress_units = float(anchor_progress_rate) * gap_anchor_years
                                            except Exception:
                                                gap_progress_units = float(self.params.gap_years)
                                            calculated_progress_at_sc = calculated_progress_at_sc + gap_progress_units
                                            try:
                                                year_label = int(self.params.present_day) if getattr(self.params, 'present_day', None) is not None else 'anchor'
                                            except Exception:
                                                year_label = 'anchor'
                                            logger.info(
                                                f"Gap-included mode: using pre-gap SC horizon {self.params.pre_gap_sc_time_horizon} and "
                                                f"adding gap {self.params.gap_years} {year_label}-progress-years (~{gap_progress_units:.6f} progress units)"
                                            )
                                        
                                        if not np.isfinite(calculated_progress_at_sc):
                                            logger.warning("Invalid progress_at_sc result")
                                            self.params.progress_at_sc = None
                                        else:
                                            self.params.progress_at_sc = calculated_progress_at_sc
                                            logger.info(f"Progress level at target horizon ({target_horizon} min): {calculated_progress_at_sc:.4f}")
                    except (ValueError, ZeroDivisionError, OverflowError) as e:
                        logger.warning(f"Could not calculate progress at sc_time_horizon_minutes: {e}")
                        self.params.progress_at_sc = None
            
            else:
                logger.error(f"Unknown horizon_extrapolation_type: {self.params.horizon_extrapolation_type}")
                return None
            
            # Store the function and parameters for later use
            self.horizon_trajectory = horizon_trajectory
            
            # Store parameters needed for anchor update in shifted form cases
            if hasattr(self, '_horizon_uses_shifted_form') and self._horizon_uses_shifted_form:
                self._horizon_params = {
                    'H_0': H_0,
                    'A_0': A_0, 
                    'T_0': T_0,
                    'original_anchor_progress': anchor_progress_for_trajectory
                }
            
            return horizon_trajectory
            
        except Exception as e:
            logger.error(f"Error fitting horizon trajectory: {e}")
            return None
    
    def _update_horizon_trajectory_anchor(self, new_anchor_progress: float):
        """
        Update the horizon trajectory function with a new anchor progress value.
        This fixes the Case 2 issue where the fitted anchor progress differs from 
        the actual integrated progress at present_day.
        """
        if not hasattr(self, '_horizon_params') or not self._horizon_params:
            logger.warning("Cannot update horizon trajectory: no stored parameters")
            return
            
        # Extract stored parameters
        H_0 = self._horizon_params['H_0']
        A_0 = self._horizon_params['A_0'] 
        T_0 = self._horizon_params['T_0']
        
        # Update the stored anchor progress
        self._horizon_params['original_anchor_progress'] = new_anchor_progress
        
        # Create new horizon trajectory function with updated anchor progress
        def horizon_trajectory(progress):
            """Map progress to horizon length using fitted decaying doubling time model with updated anchor"""
            try:
                # Handle scalar vs array inputs
                is_scalar = np.isscalar(progress)
                progress_arr = np.atleast_1d(progress)
                
                # Ensure parameters are within valid ranges
                # Allow A_0 < 0, disallow A_0 == 0 and A_0 >= 1
                if T_0 <= 0 or H_0 <= 0 or A_0 >= 1 or A_0 == 0:
                    fallback = np.full_like(progress_arr, 1e12)
                    return fallback[0] if is_scalar else fallback
                
                # Use shifted form with updated anchor progress
                progress_shifted = progress_arr - new_anchor_progress
                base_term = 1 - A_0 * progress_shifted / T_0
                # Clamp to small positive to avoid domain errors
                base_term = np.maximum(base_term, 1e-12)
                
                # Calculate the exponent
                log_denominator = np.log(1 - A_0)
                if not np.isfinite(log_denominator) or abs(log_denominator) < 1e-12:
                    fallback = np.full_like(progress_arr, 1e12)
                    return fallback[0] if is_scalar else fallback
                
                exponent = np.log(2) / log_denominator
                
                # Calculate the result
                result = H_0 * (base_term ** exponent)
                
                # Check for invalid results
                if np.any(~np.isfinite(result)) or np.any(result <= 0):
                    fallback = np.full_like(progress_arr, 1e12)
                    return fallback[0] if is_scalar else fallback
                
                return result[0] if is_scalar else result
                
            except (ValueError, ZeroDivisionError, OverflowError):
                progress_arr = np.atleast_1d(progress)
                fallback = np.full_like(progress_arr, 1e12)
                return fallback[0] if np.isscalar(progress) else fallback
        
        # Replace the horizon trajectory function
        self.horizon_trajectory = horizon_trajectory
        logger.info(f"Updated horizon trajectory anchor progress to {new_anchor_progress:.6f}")
    
    def compute_human_only_trajectory(self, time_range: List[float], initial_progress: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute human-only progress over specified time range
        """
        if initial_progress is None:
            initial_progress = 0.0  # Use a reasonable default value

        # Calls the human-only version of integrate_progress
        human_only_params = copy.deepcopy(self.params)
        human_only_params.human_only = True
        times, progress_values, research_stock_values = integrate_progress(time_range, initial_progress, self.data, human_only_params)

        # human-only initial conditions
        initial_conditions = compute_initial_conditions(self.data, human_only_params, initial_progress)
        initial_research_effort_val = initial_conditions.research_effort
        initial_research_stock_val = initial_conditions.research_stock
        logger.info(f"HUMAN-ONLY::: initial_research_stock_val: {initial_research_stock_val}, initial_research_effort_val: {initial_research_effort_val}")
        progress_rates = []
        research_efforts = []
        sw_progress_rates = []

        # human-only metrics
        for i, (t, p, rs) in enumerate(zip(times, progress_values, research_stock_values)):
            state = [p, rs]
            progress_rate, research_effort = progress_rate_at_time(t, state, self.data, human_only_params, initial_research_effort_val, initial_research_stock_val)
            progress_rates.append(progress_rate)
            research_efforts.append(research_effort)
            sw_progress_rates.append(compute_software_progress_rate(rs, research_effort, initial_research_stock_val, initial_research_effort_val, human_only_params.r_software))
        
        # Anchor stats at params.present_day
        present_day = human_only_params.present_day
        present_day_progress = np.interp(present_day, times, progress_values)
        present_day_progress_rate = np.interp(present_day, times, progress_rates)
        reference_sw_progress_rate = np.interp(cfg.SOFTWARE_PROGRESS_SCALE_REFERENCE_YEAR, times, sw_progress_rates)
        present_day_sw_progress_rate = np.interp(present_day, times, sw_progress_rates)
        present_day_research_effort = np.interp(present_day, times, research_efforts)
        present_day_research_stock = np.interp(present_day, times, research_stock_values)
        # Interpolate human and AI labor at anchor time using log-space when positive
        if np.all(self.data.L_HUMAN > 0):
            present_day_human_labor = _log_interp(present_day, self.data.time, self.data.L_HUMAN)
        else:
            present_day_human_labor = np.interp(present_day, self.data.time, self.data.L_HUMAN)
        if np.all(self.data.inference_compute > 0):
            present_day_inference_compute = _log_interp(present_day, self.data.time, self.data.inference_compute)
        else:
            present_day_inference_compute = np.interp(present_day, self.data.time, self.data.inference_compute)
        if np.all(self.data.experiment_compute > 0):
            present_day_experiment_compute = _log_interp(present_day, self.data.time, self.data.experiment_compute)
        else:
            present_day_experiment_compute = np.interp(present_day, self.data.time, self.data.experiment_compute)
        self.human_only_results = {
            'times': times,
            'progress': progress_values,
            'research_stock': research_stock_values,
            'progress_rates': progress_rates,
            'research_efforts': research_efforts,
            'reference_sw_progress_rate': reference_sw_progress_rate,
            'anchor_stats': {
                'progress': present_day_progress,
                'progress_rate': present_day_progress_rate,
                'sw_progress_rate': present_day_sw_progress_rate,
                'experiment_compute': present_day_experiment_compute,
                'human_labor': present_day_human_labor,
                'inference_compute': present_day_inference_compute,
                'research_effort': present_day_research_effort,
                'research_stock': present_day_research_stock
            },
            'input_time_series': {
                'time': self.data.time,
                'L_HUMAN': self.data.L_HUMAN,
                'inference_compute': self.data.inference_compute,
                'experiment_compute': self.data.experiment_compute,
                'training_compute_growth_rate': self.data.training_compute_growth_rate
            }
        }
        
        return times, progress_values, research_stock_values
        
    def compute_progress_trajectory(self, time_range: List[float], initial_progress: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute progress over specified time range with comprehensive metrics
        
        Args:
            time_range: [start_time, end_time]
            initial_progress: Initial progress (defaults to 0.0)
        
        Returns:
            Tuple of (times, cumulative_progress_values, research_stock_values)
        """
        _fn_start_time = time.perf_counter()
        # first, compute exp capacity params from asymptotes and anchors
        if not self.params.direct_input_exp_cap_ces_params:
            ref_exp_compute = _log_interp(cfg.REFERENCE_YEAR, self.data.time, self.data.experiment_compute)
            ref_coding_labor = _log_interp(cfg.REFERENCE_YEAR, self.data.time, self.data.L_HUMAN)
            logger.info(f"ref_exp_compute: {ref_exp_compute}, ref_coding_labor: {ref_coding_labor}")
            rho, alpha_experiment_capacity, experiment_compute_exponent = compute_exp_capacity_params_from_anchors(
                self.params.inf_labor_asymptote, 
                self.params.inf_compute_asymptote, 
                (cfg.REFERENCE_COMPUTE_CHANGE, self.params.compute_anchor_exp_cap), 
                (cfg.REFERENCE_LABOR_CHANGE, self.params.labor_anchor_exp_cap), 
                ref_exp_compute, ref_coding_labor,
                self.params.parallel_penalty
            )
            self.params.rho_experiment_capacity = rho
            self.params.alpha_experiment_capacity = alpha_experiment_capacity
            self.params.experiment_compute_exponent = experiment_compute_exponent
            logger.info(f"computed exp capacity params: rho: {rho}, alpha: {alpha_experiment_capacity}, experiment_compute_exponent: {experiment_compute_exponent}")
        else:
            logger.info(f"using direct input exp capacity params: rho: {self.params.rho_experiment_capacity}, alpha: {self.params.alpha_experiment_capacity}, experiment_compute_exponent: {self.params.experiment_compute_exponent}")
        
        # hackily handle doubling_difficulty_growth_rate = 1 case (equivalent to decay_rate = 0)
        if self.params.doubling_difficulty_growth_rate == 1.0:
            logger.info(f"doubling_difficulty_growth_rate is 1.0 (decay_rate = 0), setting to exponential")
            self.params.horizon_extrapolation_type = "exponential"

        # next compute human-only trajectory
        # Need to do this for various reasons, e.g. to auto-fit the METR trajectory you need to the (effective compute, horizon) pairs (we assume no automation)
        # Also if we want to specify current doubling time or gap size in years rather than progress units, need to know how much EC was increased in present day
        _t_human_only_start = time.perf_counter()
        # First, rum a human-only trajectory to get anchor stats
        self.compute_human_only_trajectory(time_range, initial_progress)
        # Then, scale r_software so that sw_progress_rate at anchor time is 1
        logger.info(f"reference year: {cfg.SOFTWARE_PROGRESS_SCALE_REFERENCE_YEAR}")
        logger.info(f"reference sw_progress_rate: {self.human_only_results['reference_sw_progress_rate']}")
        logger.info(f"desired software_progress_rate_at_reference_year: {self.params.software_progress_rate_at_reference_year}")
        self.params.r_software = self.params.software_progress_rate_at_reference_year * self.params.r_software/self.human_only_results['reference_sw_progress_rate']
        logger.info(f"new r_software: {self.params.r_software}")
        #Finally, recompute the human-only trajectory with the new r_software
        human_only_times, human_only_progress, _ = self.compute_human_only_trajectory(time_range, initial_progress)
        logger.info(f"new reference sw_progress_rate: {self.human_only_results['reference_sw_progress_rate']}")
        _dt_human_only = time.perf_counter() - _t_human_only_start
        logger.info(f"Timing: human-only trajectory computed in {_dt_human_only:.3f}s (elapsed {time.perf_counter() - _fn_start_time:.3f}s)")
        
        # Store the various facts about the present day, assuming no automation
        present_day = self.params.present_day
        present_day_progress = self.human_only_results['anchor_stats']['progress']
        present_day_progress_rate = self.human_only_results['anchor_stats']['progress_rate']
        present_day_sw_progress_rate = self.human_only_results['anchor_stats']['sw_progress_rate']
        present_day_research_effort = self.human_only_results['anchor_stats']['research_effort']
        present_day_research_stock = self.human_only_results['anchor_stats']['research_stock']
        present_day_human_labor = self.human_only_results['anchor_stats']['human_labor']
        present_day_inference_compute = self.human_only_results['anchor_stats']['inference_compute']
        present_day_experiment_compute = self.human_only_results['anchor_stats']['experiment_compute']
        logger.info(f"present_day_human_labor: {present_day_human_labor}, present_day_inference_compute: {present_day_inference_compute}")

        # estimate horizon trajectory from METR data using human-only trajectory
        _t_horizon_est_start = time.perf_counter()
        try:
            anchor_progress_rate = self.human_only_results['anchor_stats']['progress_rate']
            self.estimate_horizon_trajectory(human_only_times, human_only_progress, anchor_progress_rate)
            _dt_horizon_est = time.perf_counter() - _t_horizon_est_start
            logger.info(f"Timing: horizon trajectory estimation completed in {_dt_horizon_est:.3f}s (elapsed {time.perf_counter() - _fn_start_time:.3f}s)")
        except Exception as e:
            logger.warning(f"Failed to estimate horizon trajectory: {e}")
            _dt_horizon_est = time.perf_counter() - _t_horizon_est_start
            logger.info(f"Timing: horizon trajectory estimation failed after {_dt_horizon_est:.3f}s (elapsed {time.perf_counter() - _fn_start_time:.3f}s)")
            self.horizon_trajectory = None

        # Convert AI research taste slope if using "SDs per progress-year" mode
        # This must be done after computing anchor_progress_rate but before the main trajectory
        # Store original slope for display purposes
        self._original_taste_slope = self.params.ai_research_taste_slope
        if self.params.taste_schedule_type == "SDs per progress-year":
            try:
                anchor_progress_rate = self.human_only_results['anchor_stats']['progress_rate']
                if anchor_progress_rate is not None and np.isfinite(anchor_progress_rate) and anchor_progress_rate > 0:
                    # Convert from SD/progress-year to SD/progress-unit by multiplying by (1 year / anchor_progress_rate progress-units)
                    # which simplifies to dividing by anchor_progress_rate
                    original_slope = self.params.ai_research_taste_slope
                    converted_slope = original_slope / anchor_progress_rate
                    logger.info(f"Converting taste slope from {original_slope:.6f} SD/progress-year to {converted_slope:.6f} SD/progress-unit (anchor rate: {anchor_progress_rate:.6f})")
                    self.params.ai_research_taste_slope = converted_slope
                else:
                    logger.warning(f"Invalid anchor_progress_rate for taste slope conversion: {anchor_progress_rate}")
            except Exception as e:
                logger.warning(f"Failed to convert taste slope for progress-year mode: {e}")

        # compute automation fraction at anchor time by solving for lower anchor via AutomationModel
        _t_anchor_solver_start = time.perf_counter()
        anchor_aut_frac = solve_lower_anchor_via_automation_model(
            self.params.swe_multiplier_at_present_day,
            float(present_day_progress),
            float(present_day_human_labor),
            float(present_day_inference_compute),
            self.params,
        )
        _dt_anchor_solver = time.perf_counter() - _t_anchor_solver_start
        logger.info(f"Timing: lower anchor via AutomationModel solved in {_dt_anchor_solver:.3f}s (elapsed {time.perf_counter() - _fn_start_time:.3f}s)")
        logger.info(f"calculated anchor automation fraction (via AM solver): {anchor_aut_frac} from swe_multiplier_at_present_day: {self.params.swe_multiplier_at_present_day} and present_day: {present_day}")
        automation_anchors = {
            present_day_progress: anchor_aut_frac,
            self.params.progress_at_sc: self.params.automation_fraction_at_superhuman_coder
        }
        logger.info(f"Automation anchors: {automation_anchors}")
        self.params.automation_anchors = automation_anchors

        # STORE AUTOMATION MODEL INSTANCE IN PARAMS HERE
        self.params.automation_model = AutomationModel(self.params)

        # Below gives you at each time what is the effectige compute value and what is the research stock value. It runs the whole model.
        # With just time -> effective compute and research stock, you can compute all the other metrics.
        _t_integrate_start = time.perf_counter()
        times, progress_values, research_stock_values = integrate_progress(time_range, initial_progress, self.data, self.params)
        _dt_integrate = time.perf_counter() - _t_integrate_start
        logger.info(f"Timing: integrate_progress completed in {_dt_integrate:.3f}s (elapsed {time.perf_counter() - _fn_start_time:.3f}s)")
        
        # Fix for Case 2 anchor horizon blowup: Update anchor_progress after ODE integration
        # This ensures that the horizon at present_day matches the specified present_horizon value
        if (self.horizon_trajectory is not None and 
            hasattr(self, '_horizon_uses_shifted_form') and self._horizon_uses_shifted_form and
            self.params.present_horizon is not None):
            
            # Recompute the actual progress at present_day from the integrated trajectory
            actual_present_day_progress = np.interp(self.params.present_day, times, progress_values)
            
            # Update the horizon trajectory function with the corrected anchor progress
            logger.info(f"Updating present_day progress from fitted value ({present_day_progress:.6f}) to actual integrated value: {actual_present_day_progress:.6f}")
            self._update_horizon_trajectory_anchor(actual_present_day_progress)
        
        # Use utility function to compute initial conditions with the correct parameters
        initial_conditions = compute_initial_conditions(self.data, self.params, initial_progress)
        initial_research_effort_val = initial_conditions.research_effort
        initial_research_stock_val = initial_conditions.research_stock
        logger.info(f"ACTUAL::: initial_research_stock_val: {initial_research_stock_val}, initial_research_effort_val: {initial_research_effort_val}")

        # Calculate all metrics in a single pass to avoid redundancy
        progress_rates = []
        research_efforts = []
        automation_fractions = []
        ai_research_tastes = []
        ai_research_taste_sds = []
        ai_research_taste_quantiles = []
        aggregate_research_tastes = []
        coding_labors = []
        coding_labors_with_present_resources = []
        software_progress_rates = []
        software_progress_rates_present_resources = []
        software_efficiency = []  # Integral of software_progress_rate
        human_only_research_efforts = []
        human_only_software_progress_rates = []
        human_only_progress_rates = []
        ai_labor_contributions = []
        human_labor_contributions = []
        ai_coding_labor_multipliers = []
        ai_coding_labor_mult_ref_present_day = []
        ai_research_stock_multipliers = []
        ai_software_progress_multipliers = []
        ai_sw_progress_mult_ref_present_day = []
        ai_overall_progress_multipliers = []
        discounted_exp_compute = []
        horizon_lengths = []
        effective_compute = []
        training_compute = []
        experiment_capacity = []
        
        # logger.info(f"Computing comprehensive metrics for {len(times)} time points")
        
        _t_metrics_loop_start = time.perf_counter()
        for i, (t, progress, rs) in enumerate(zip(times, progress_values, research_stock_values)):
            try:
                state = [progress, rs]
                rates = progress_rate_at_time(t, state, self.data, self.params, initial_research_effort_val, initial_research_stock_val)
                progress_rates.append(rates[0])
                research_efforts.append(rates[1])

                # INPUT TIME SERIES
                L_HUMAN = _log_interp(t, self.data.time, self.data.L_HUMAN)
                inference_compute = _log_interp(t, self.data.time, self.data.inference_compute)
                experiment_compute = _log_interp(t, self.data.time, self.data.experiment_compute)
                training_compute_growth_rate = _log_interp(t, self.data.time, self.data.training_compute_growth_rate)
                # Compute discounted experiment compute
                discounted_exp_compute_val = experiment_compute ** self.params.experiment_compute_exponent
                discounted_exp_compute.append(discounted_exp_compute_val if np.isfinite(discounted_exp_compute_val) else 0.0)
                
                
                # AUTOMATION FRACTION
                automation_fraction = compute_automation_fraction(progress, self.params)
                automation_fractions.append(automation_fraction)
                
                # RESEARCH TASTE
                ai_research_taste = compute_ai_research_taste(progress, self.params)
                ai_research_taste_sd = self.taste_distribution.get_sd_of_taste(ai_research_taste)
                ai_research_taste_quantile = self.taste_distribution.get_quantile_of_taste(ai_research_taste)
                aggregate_research_taste = compute_aggregate_research_taste(ai_research_taste, median_to_top_gap=self.params.median_to_top_taste_multiplier)
                ai_research_tastes.append(ai_research_taste)
                ai_research_taste_sds.append(ai_research_taste_sd if np.isfinite(ai_research_taste_sd) else 0.0)
                ai_research_taste_quantiles.append(ai_research_taste_quantile if np.isfinite(ai_research_taste_quantile) else 0.0)
                aggregate_research_tastes.append(aggregate_research_taste)
                
                # CODING LABOR
                if getattr(self.params, 'coding_labor_mode', 'simple_ces') == 'optimal_ces':
                    H = float(L_HUMAN)
                    C = float(inference_compute)
                    logE = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM) * progress)
                    try:
                        automation_model = self.params.automation_model
                        L_opt = automation_model.coding_labor_optimal_ces(H, C, logE, self.params)
                        L_opt_present_resources = automation_model.coding_labor_optimal_ces(present_day_human_labor, present_day_inference_compute, logE, self.params)
                        if L_opt is None or not np.isfinite(L_opt):
                            assert False, "L_opt is None or not np.isfinite(L_opt)"
                        else:
                            # TODO: why is this being converted to serial-equivalent?
                            coding_labor = L_opt
                            coding_labor_with_present_resources = L_opt_present_resources
                            serial_coding_labor = float((L_opt ** self.params.parallel_penalty) * self.params.coding_labor_normalization)
                            serial_coding_labor_with_present_resources = float((L_opt_present_resources ** self.params.parallel_penalty) * self.params.coding_labor_normalization)
                    except Exception as e:
                        assert False, f"Falling back to simple CES in metrics due to optimal_ces error: {e}"
                else:
                    serial_coding_labor = compute_coding_labor(
                        automation_fraction, inference_compute, L_HUMAN, 
                        self.params.rho_coding_labor, self.params.parallel_penalty, self.params.coding_labor_normalization
                    )
                coding_labors.append(coding_labor if np.isfinite(coding_labor) else 0.0)
                coding_labors_with_present_resources.append(coding_labor_with_present_resources if np.isfinite(coding_labor_with_present_resources) else 0.0)
                

                # EXPERIMENT CAPACITY
                current_research_effort = research_efforts[i]
                exp_capacity = current_research_effort / aggregate_research_taste if aggregate_research_taste > 0 else 0.0
                experiment_capacity.append(exp_capacity if np.isfinite(exp_capacity) else 0.0)

                # RESEARCH EFFORT
                research_effort_present_resources = compute_research_effort(
                    present_day_experiment_compute, serial_coding_labor_with_present_resources, 
                    self.params.alpha_experiment_capacity, self.params.rho_experiment_capacity, self.params.experiment_compute_exponent, aggregate_research_taste
                )

                # SOFTWARE PROGRESS RATE
                assert current_research_effort == compute_research_effort(
                    experiment_compute, serial_coding_labor, 
                    self.params.alpha_experiment_capacity, self.params.rho_experiment_capacity, self.params.experiment_compute_exponent, aggregate_research_taste
                )
                current_research_effort = research_efforts[i]
                software_rate = compute_software_progress_rate(
                    rs, current_research_effort, 
                    initial_research_stock_val, 
                    initial_research_effort_val,
                    self.params.r_software
                )
                software_progress_rates.append(software_rate if np.isfinite(software_rate) else 0.0)
                software_rate_present_resources = compute_software_progress_rate(
                    present_day_research_stock, research_effort_present_resources, 
                    initial_research_stock_val,
                    initial_research_effort_val,
                    self.params.r_software
                )
                software_progress_rates_present_resources.append(software_rate_present_resources if np.isfinite(software_rate_present_resources) else 0.0)
                
                # SOFTWARE EFFICIENCY (OOMS)
                if i == 0:
                    # Initialize software efficiency at 0
                    software_efficiency_val = 0.0
                else:
                    # Trapezoidal integration: add area of trapezoid from previous time step
                    dt = times[i] - times[i-1]
                    avg_rate = (software_progress_rates[i] + software_progress_rates[i-1]) / 2.0
                    software_efficiency_val = software_efficiency[i-1] + avg_rate * dt
                software_efficiency.append(software_efficiency_val if np.isfinite(software_efficiency_val) else 0.0)
                
                # TRAINING COMPUTE (OOMS)
                if i == 0:
                    # Initialize training compute at 0
                    training_compute_val = 0.0
                else:
                    # Trapezoidal integration: add area of trapezoid from previous time step
                    dt = times[i] - times[i-1]
                    # Get the previous training_compute_growth_rate for trapezoidal rule
                    prev_training_compute_growth_rate = _log_interp(times[i-1], self.data.time, self.data.training_compute_growth_rate)
                    avg_growth_rate = (training_compute_growth_rate + prev_training_compute_growth_rate) / 2.0
                    training_compute_val = training_compute[i-1] + avg_growth_rate * dt
                
                training_compute.append(training_compute_val if np.isfinite(training_compute_val) else 0.0)

                # EFFECTIVE COMPUTE (OOMS)
                effective_compute_val = training_compute_val + software_efficiency_val
                effective_compute.append(effective_compute_val)

                # TIME HORIZON
                horizon_length = 0.0  # Default fallback
                if self.horizon_trajectory is not None:
                    try:
                        horizon_length = self.horizon_trajectory(progress)
                        if not np.isfinite(horizon_length) or horizon_length < 0:
                            horizon_length = 0.0
                    except Exception as horizon_e:
                        assert False, f"Error computing horizon at progress {progress}: {horizon_e}"
                        logger.warning(f"Error computing horizon at progress {progress}: {horizon_e}")
                        horizon_length = 0.0
                horizon_lengths.append(horizon_length)

                # INSTANTANEOUS HUMAN-ONLY METRICS
                human_only_coding_labor = L_HUMAN
                human_only_serial_coding_labor = L_HUMAN**self.params.parallel_penalty
                human_only_aggregate_research_taste = compute_aggregate_research_taste(0, median_to_top_gap=self.params.median_to_top_taste_multiplier) # No AI research taste
                human_only_research_effort = compute_research_effort(
                    experiment_compute, human_only_serial_coding_labor, 
                    self.params.alpha_experiment_capacity, self.params.rho_experiment_capacity, self.params.experiment_compute_exponent, human_only_aggregate_research_taste
                )
                human_only_research_efforts.append(human_only_research_effort if np.isfinite(human_only_research_effort) else 0.0)
                human_only_software_rate = compute_software_progress_rate(
                    rs, human_only_research_effort,
                    initial_research_stock_val,
                    initial_research_effort_val,
                    self.params.r_software
                )
                human_only_software_progress_rates.append(human_only_software_rate if np.isfinite(human_only_software_rate) else 0.0)
                human_only_overall_rate = compute_overall_progress_rate(
                    human_only_software_rate, training_compute_growth_rate
                )
                human_only_progress_rates.append(
                    human_only_overall_rate if np.isfinite(human_only_overall_rate) else 0.0
                )
                
                # Calculate labor contributions to cognitive output
                human_contrib = L_HUMAN
                ai_contrib = max(0.0, coding_labor - human_contrib)  # Ensure non-negative
                
                human_labor_contributions.append(human_contrib)
                ai_labor_contributions.append(ai_contrib)

                # AUTOMATION MULTIPLIERS
                if self.params.parallel_penalty and self.params.parallel_penalty != 0:
                    ai_coding_labor_multipliers.append(coding_labor / human_contrib if ai_contrib > 0 else 1.0)
                    ai_coding_labor_mult_ref_present_day.append(coding_labor_with_present_resources / present_day_human_labor if ai_contrib > 0 else 1.0)
                else:
                    ai_coding_labor_multipliers.append(0.0)
                ai_research_stock_multipliers.append(research_efforts[i] / human_only_research_effort if human_only_research_effort > 0 else 0.0)
                ai_software_progress_multipliers.append(software_rate / human_only_software_progress_rates[i] if human_only_software_progress_rates[i] > 0 else 0.0)
                ai_sw_progress_mult_ref_present_day.append(software_rate_present_resources / present_day_sw_progress_rate if present_day_sw_progress_rate > 0 else 0.0)
                ai_overall_progress_multipliers.append(progress_rates[i] / human_only_progress_rates[i] if human_only_progress_rates[i] > 0 else 0.0)

                
            except Exception as e:
                assert False, f"Error calculating metrics at t={t}: {e}"
                logger.warning(f"Error calculating metrics at t={t}: {e}")
                # Use safe fallback values
                if len(progress_rates) <= i:
                    progress_rates.append(0.0)
                if len(research_efforts) <= i:
                    research_efforts.append(0.0)
                automation_fractions.append(0.0)
                ai_research_tastes.append(0.0)
                ai_research_taste_sds.append(0.0)
                ai_research_taste_quantiles.append(0.0)
                aggregate_research_tastes.append(cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK)  # Default to no enhancement
                coding_labors.append(0.0)
                software_progress_rates.append(0.0)
                software_efficiency.append(0.0)
                human_only_progress_rates.append(0.0)
                human_only_research_efforts.append(0.0)
                human_only_software_progress_rates.append(0.0)
                human_labor_contributions.append(0.0)
                ai_labor_contributions.append(0.0)
                ai_coding_labor_multipliers.append(0.0)
                ai_research_stock_multipliers.append(0.0)
                ai_software_progress_multipliers.append(0.0)
                ai_overall_progress_multipliers.append(0.0)
                discounted_exp_compute.append(0.0)
                horizon_lengths.append(0.0)
                effective_compute.append(0.0)
                training_compute.append(0.0)
                experiment_capacity.append(0.0)
        _dt_metrics_loop = time.perf_counter() - _t_metrics_loop_start
        _num_iters = len(times) if 'times' in locals() and times is not None else 0
        _avg_iter_ms = (1000.0 * _dt_metrics_loop / _num_iters) if _num_iters > 0 else float('nan')
        logger.info(f"Timing: metrics loop processed {_num_iters} points in {_dt_metrics_loop:.3f}s (avg {_avg_iter_ms:.3f} ms/iter, elapsed {time.perf_counter() - _fn_start_time:.3f}s)")
        # Calculate time when superhuman coder level is reached
        sc_time = None
        if self.params.progress_at_sc is not None:
            # Find the time when progress reaches progress_at_sc
            sc_progress_target = self.params.progress_at_sc
            
            # Check if SC is reached within the trajectory
            if progress_values[-1] >= sc_progress_target:
                # Find the exact time by interpolation
                if progress_values[0] >= sc_progress_target:
                    # SC level already reached at start
                    sc_time = times[0]
                else:
                    # Interpolate to find when progress crosses sc_progress_target
                    try:
                        sc_time = np.interp(sc_progress_target, progress_values, times)
                    except Exception as e:
                        logger.warning(f"Error interpolating SC time: {e}")
                        sc_time = None
                        
                logger.info(f"Superhuman Coder level ({sc_progress_target:.3f}) reached at time {sc_time:.3f}")
            else:
                logger.info(f"Superhuman Coder level ({sc_progress_target:.3f}) not reached within trajectory (final progress: {progress_values[-1]:.3f})")
        
        # Calculate software progress multiplier at SC
        if sc_time is not None:
            self.sc_sw_multiplier = _log_interp(sc_time, times, np.asarray(ai_sw_progress_mult_ref_present_day, dtype=float))
        else:
            sc_sw_multiplier = None
        
        # Compute the time when ai_coding_labor_mult_ref_present_day first reaches the required threshold
        # using exponential (log-space) interpolation between adjacent samples.
        ai2027_sc_time = None
        try:
            if self.params.parallel_penalty is not None and self.params.parallel_penalty != 0:
                ai2027_sc_required_mult = (
                    cfg.LABOR_MULT_EXTRA_FOR_AI2027_SC * 30 * 30 ** (1 / self.params.parallel_penalty)
                )
                ai2027_sc_time = _find_exponential_crossing_time(
                    np.asarray(times, dtype=float),
                    np.asarray(ai_coding_labor_mult_ref_present_day, dtype=float),
                    float(ai2027_sc_required_mult),
                )
        except Exception as e:
            logger.warning(f"Error computing ai2027_sc_time: {e}")
            
        # Calculate progress rate at anchor time
        present_day = self.params.present_day
        anchor_progress_rate = None
        if present_day is not None:
            # Check if anchor time is within our trajectory
            if times[0] <= present_day <= times[-1]:
                anchor_progress_rate = np.interp(present_day, times, progress_rates)
                logger.info(f"Progress rate at anchor time ({present_day:.3f}): {anchor_progress_rate:.6f}")
            else:
                logger.warning(f"Anchor time {present_day:.3f} is outside trajectory range [{times[0]:.3f}, {times[-1]:.3f}]")

        # Compute instantaneous doubling time at the anchor (years)
        instantaneous_anchor_doubling_time_years = None
        try:
            if (self.horizon_trajectory is not None and
                anchor_progress_rate is not None and np.isfinite(anchor_progress_rate) and anchor_progress_rate > 0):
                anchor_progress_value = self.human_only_results['anchor_stats']['progress']
                progress = float(anchor_progress_value)
                # Numerical derivative of ln(horizon) with respect to progress at anchor
                eps = 1e-6 * max(1.0, abs(progress))
                if eps == 0:
                    eps = 1e-6
                H_p = self.horizon_trajectory(progress)
                H_p_eps = self.horizon_trajectory(progress + eps)
                if (np.isfinite(H_p) and np.isfinite(H_p_eps) and H_p > 0 and H_p_eps > 0):
                    dlnH_dprogress = (np.log(H_p_eps) - np.log(H_p)) / eps
                    if np.isfinite(dlnH_dprogress) and dlnH_dprogress > 0:
                        instantaneous_anchor_doubling_time_years = float(np.log(2) / (dlnH_dprogress * anchor_progress_rate))
        except Exception as e:
            logger.warning(f"Failed computing instantaneous doubling time at anchor: {e}")

        # Compute AI research taste slope in SD per anchor-progress-year (SD/year at anchor)
        ai_taste_slope_per_anchor_progress_year = None
        ai_taste_slope_per_effective_oom = None
        try:
            if anchor_progress_rate is not None and np.isfinite(anchor_progress_rate):
                if self.params.taste_schedule_type == "SDs per progress-year":
                    # For progress-year mode, use original input value for progress-year display
                    ai_taste_slope_per_anchor_progress_year = float(self._original_taste_slope)
                    # And compute effective OOM value using converted slope
                    ai_taste_slope_per_effective_oom = float(self.params.ai_research_taste_slope)
                else:
                    # For effective OOM mode, compute progress-year display from effective OOM input
                    ai_taste_slope_per_effective_oom = float(self.params.ai_research_taste_slope)
                    ai_taste_slope_per_anchor_progress_year = float(self.params.ai_research_taste_slope) * float(anchor_progress_rate)
        except Exception as e:
            logger.warning(f"Failed computing taste slope conversions: {e}")
        
        # RESCALE SOFTWARE EFFICIENCY TO REFLECT PRESENT-DAY BASELINE
        software_efficiency = software_efficiency - np.interp(cfg.TRAINING_COMPUTE_REFERENCE_YEAR, times, software_efficiency)
        training_compute = training_compute - np.interp(cfg.TRAINING_COMPUTE_REFERENCE_YEAR, times, training_compute) + cfg.TRAINING_COMPUTE_REFERENCE_OOMS
        print(software_efficiency)
        print(training_compute)
        effective_compute = effective_compute - np.interp(cfg.TRAINING_COMPUTE_REFERENCE_YEAR, times, effective_compute) + cfg.TRAINING_COMPUTE_REFERENCE_OOMS

        # Store comprehensive results
        self.results = {
            'times': times,
            'progress': progress_values,
            'research_stock': research_stock_values,
            'automation_fraction': automation_fractions,
            'ai_research_taste': ai_research_tastes,
            'ai_research_taste_sd': ai_research_taste_sds,
            'ai_research_taste_quantile': ai_research_taste_quantiles,
            'aggregate_research_taste': aggregate_research_tastes,
            'progress_rates': progress_rates,
            'research_efforts': research_efforts,
            'coding_labors': coding_labors,
            'coding_labors_with_present_resources': coding_labors_with_present_resources,
            'software_progress_rates': software_progress_rates,
            'software_efficiency': software_efficiency,
            'human_only_progress_rates': human_only_progress_rates,
            'ai_labor_contributions': ai_labor_contributions,
            'human_labor_contributions': human_labor_contributions,
            'ai_coding_labor_multipliers': ai_coding_labor_multipliers,
            'ai_coding_labor_mult_ref_present_day': ai_coding_labor_mult_ref_present_day,
            'ai_research_stock_multipliers': ai_research_stock_multipliers,
            'ai_software_progress_multipliers': ai_software_progress_multipliers,
            'ai_sw_progress_mult_ref_present_day': ai_sw_progress_mult_ref_present_day,
            'ai_overall_progress_multipliers': ai_overall_progress_multipliers,
            'discounted_exp_compute': discounted_exp_compute,
            'horizon_lengths': horizon_lengths,
            'effective_compute': effective_compute,
            'training_compute': training_compute,
            'experiment_capacity': experiment_capacity,
            'sc_time': sc_time,  # Time when superhuman coder level is reached
            'sc_progress_level': self.params.progress_at_sc,  # Progress level for SC
            'sc_sw_multiplier': self.sc_sw_multiplier if hasattr(self, 'sc_sw_multiplier') else None,  # Software progress multiplier at SC
            'ai2027_sc_time': ai2027_sc_time,  # Time when @AI2027 SC condition is met
            'present_day': present_day,  # Anchor time for manual horizon fitting
            'anchor_progress_rate': anchor_progress_rate,  # Progress rate at anchor time
            'instantaneous_anchor_doubling_time_years': instantaneous_anchor_doubling_time_years,  # Instantaneous doubling time of horizon at anchor (years)
            'ai_research_taste_slope_per_anchor_progress_year': ai_taste_slope_per_anchor_progress_year,  # SD per anchor-progress-year
            'ai_research_taste_slope_per_effective_oom': ai_taste_slope_per_effective_oom,  # SD per effective OOM
            'input_time_series': {
                'time': self.data.time,
                'L_HUMAN': self.data.L_HUMAN,
                'inference_compute': self.data.inference_compute,
                'experiment_compute': self.data.experiment_compute,
                'training_compute_growth_rate': self.data.training_compute_growth_rate
            },
            'exp_capacity_params': {
                'rho': self.params.rho_experiment_capacity,
                'alpha': self.params.alpha_experiment_capacity,
                'experiment_compute_exponent': self.params.experiment_compute_exponent,
            }
        }
        self.results['milestones'] = self.compute_milestones()
        # logger.info(f"Computed trajectory from {time_range[0]} to {time_range[1]}")
        # logger.info(f"Progress: {progress_values[0]:.3f} -> {progress_values[-1]:.3f}")
        # logger.info(f"Research Stock: {research_stock_values[0]:.3f} -> {research_stock_values[-1]:.3f}")
        # logger.info(f"Automation: {automation_fractions[0]:.3f} -> {automation_fractions[-1]:.3f}")
        
        return times, progress_values, research_stock_values
    
    def compute_milestones(self):
        """
        Compute milestones for the model.
        """
        milestones = {
            'TCD-AI': {
                'metric': 'progress',
                'target': self.results['sc_progress_level'],
                'interpolation_type': 'linear',            },
            'AI2027-SC': {
                'metric': 'ai_coding_labor_mult_ref_present_day',
                'target': cfg.LABOR_MULT_EXTRA_FOR_AI2027_SC * 30 * 30 ** (1 / self.params.parallel_penalty),
                'interpolation_type': 'exponential',
            },
            '5x-AI': {
                'metric': 'ai_sw_progress_mult_ref_present_day',
                'target': 5,
                'interpolation_type': 'exponential',
                'progress_multiplier': 5,
            },
            '25x-AI': {
                'metric': 'ai_sw_progress_mult_ref_present_day',
                'target': 25,
                'interpolation_type': 'exponential',
                'progress_multiplier': 25
            },
            '250x-AI': {
                'metric': 'ai_sw_progress_mult_ref_present_day',
                'target': 250,
                'interpolation_type': 'exponential',
                'progress_multiplier': 250
            },
            '2000x-AI': {
                'metric': 'ai_sw_progress_mult_ref_present_day',
                'target': 2000,
                'interpolation_type': 'exponential',
                'progress_multiplier': 2000
            },
            '(Expensive) SAR': {
                'metric': 'ai_research_taste',
                'target': cfg.MEDIAN_TO_TOP_TASTE_MULTIPLIER,
                'interpolation_type': 'exponential'
            },
            '(Expensive) SIAR': {
                'metric': 'ai_research_taste',
                'target': cfg.MEDIAN_TO_TOP_TASTE_MULTIPLIER**3,
                'interpolation_type': 'exponential'
            }
        }
        for milestone in milestones.values():
            if milestone['interpolation_type'] == 'exponential':
                time = _find_exponential_crossing_time(
                    np.asarray(self.results['times'], dtype=float),
                    np.asarray(self.results[milestone['metric']], dtype=float),
                    float(milestone['target'])
                )
            else:
                time = np.interp(milestone['target'], self.results[milestone['metric']], self.results['times'])
                if np.interp(time, self.results['times'], self.results[milestone['metric']]) < milestone['target'] - 1e-4:
                    time = None
            if time is not None:
                milestone['time'] = time
                if 'progress_multiplier' not in milestone:
                    milestone['progress_multiplier'] = _log_interp(milestone['time'], self.results['times'], np.asarray(self.results['ai_sw_progress_mult_ref_present_day'], dtype=float))
                if 'effective_compute_ooms' not in milestone:
                    milestone['effective_compute_ooms'] = np.interp(milestone['time'], self.results['times'], np.asarray(self.results['effective_compute'], dtype=float))
        return milestones
    
    def evaluate_anchor_constraint(self, constraint: AnchorConstraint) -> float:
        """
        Evaluate an anchor constraint using pre-computed metrics.
        
        Args:
            constraint: Constraint specification
            
        Returns:
            Error (model_value - target_value), normalized if target != 0
        """
        if not self.results:
            raise ValueError("No results available. Run compute_progress_trajectory first.")
        
        # Cache frequently accessed arrays to avoid repeated dictionary lookups
        times = self.results['times']
        
        # Extract condition (only one allowed)
        conditions = constraint.conditions
        if len(conditions) != 1:
            raise ValueError("Only one condition is allowed")
        
        condition_key, condition_value = next(iter(conditions.items()))
        
        # Optimize condition processing with direct lookup table
        if condition_key == 'time':
            # Already optimized with binary search
            if condition_value <= times[0]:
                time_idx = 0
            elif condition_value >= times[-1]:
                time_idx = len(times) - 1
            else:
                time_idx = np.searchsorted(times, condition_value)
                if time_idx > 0:
                    # Choose closer of the two neighboring points
                    if abs(times[time_idx] - condition_value) > abs(times[time_idx-1] - condition_value):
                        time_idx = time_idx - 1
        else:
            # For all other conditions, use optimized search
            condition_array = self.results.get(condition_key)
            if condition_array is None:
                # Handle input time series conditions
                if condition_key in ['L_HUMAN', 'inference_compute', 'experiment_compute']:
                    input_series = self.results['input_time_series'][condition_key]
                    # Interpolate to model times for comparison
                    condition_array = np.interp(times, self.results['input_time_series']['time'], input_series)
                else:
                    raise ValueError(f"Unknown condition key: {condition_key}")
            
            # Use optimized closest value search
            time_idx = self._find_closest_index(condition_array, condition_value)
            
            # Check if condition value is reachable (with tolerance)
            if abs(condition_array[time_idx] - condition_value) > 0.01:
                logger.warning(f"{condition_key} never reaches condition value")
                return 0.0

        # Direct array access for target variable (avoid dictionary lookup)
        target_key = constraint.target_variable
        if target_key == 'progress_rate':
            model_value = self.results['progress_rates'][time_idx]
        elif target_key == 'automation_fraction':
            model_value = self.results['automation_fraction'][time_idx]
        elif target_key == 'coding_labor':
            model_value = self.results['coding_labors'][time_idx]
        else:
            raise ValueError(f"Unknown target variable: {target_key}")
        
        # Optimized error calculation
        if constraint.target_value != 0:
            error = (model_value - constraint.target_value) / abs(constraint.target_value)
            # Use faster clipping if available
            error = max(-cfg.RELATIVE_ERROR_CLIP, min(cfg.RELATIVE_ERROR_CLIP, error))
        else:
            error = model_value - constraint.target_value
        
        logger.debug(f"Constraint evaluation: target={target_key}, model_value={model_value:.6f}, target_value={constraint.target_value:.6f}, error={error:.6f}")
        
        return error
    
    def _find_closest_index(self, array: np.ndarray, target_value: float) -> int:
        """
        Optimized method to find closest array index to target value.
        
        Uses binary search when array is sorted, otherwise falls back to linear search.
        Could be further optimized by checking monotonicity once and caching the result.
        """
        # Quick check if array is sorted (monotonic)
        if len(array) > 1:
            is_increasing = np.all(array[1:] >= array[:-1])
            is_decreasing = np.all(array[1:] <= array[:-1])
            
            if is_increasing:
                # Use binary search for increasing array
                idx = np.searchsorted(array, target_value)
                if idx == 0:
                    return 0
                elif idx == len(array):
                    return len(array) - 1
                else:
                    # Choose closer of two neighboring points
                    if abs(array[idx] - target_value) < abs(array[idx-1] - target_value):
                        return idx
                    else:
                        return idx - 1
            elif is_decreasing:
                # Use binary search for decreasing array (search on reversed)
                reversed_array = array[::-1]
                reversed_idx = np.searchsorted(reversed_array, target_value)
                if reversed_idx == 0:
                    return len(array) - 1
                elif reversed_idx == len(reversed_array):
                    return 0
                else:
                    # Convert back to original index and choose closer point
                    idx1 = len(array) - 1 - reversed_idx
                    idx2 = len(array) - reversed_idx
                    if abs(array[idx1] - target_value) < abs(array[idx2] - target_value):
                        return idx1
                    else:
                        return idx2
        
        # Fall back to linear search for non-monotonic arrays
        return np.argmin(np.abs(array - target_value))
    
    def evaluate_all_constraints(self, constraints: List[AnchorConstraint]) -> float:
        """
        Evaluate all anchor constraints and return total weighted error.
        
        Args:
            constraints: List of anchor constraints
            
        Returns:
            Total weighted squared error
        """
        total_error = 0.0
        for i, constraint in enumerate(constraints):
            try:
                error = self.evaluate_anchor_constraint(constraint)
                weighted_error = constraint.weight * (error ** 2)
                total_error += weighted_error
                logger.debug(f"Constraint {i+1}: error={error:.6f}, weighted_error={weighted_error:.6f}")
            except Exception as e:
                logger.warning(f"Error evaluating constraint {i+1}: {e}")
                # Return high penalty for constraint evaluation failures
                return cfg.OBJECTIVE_FUNCTION_CONFIG['high_penalty']
        
        return total_error
    
    
    def get_progress_at_time(self, time: float) -> float:
        """
        Get progress value at a specific time by interpolating computed results.
        
        Args:
            time: Time in decimal years
            
        Returns:
            Interpolated progress value
            
        Raises:
            ValueError: If no results are available or time is outside computed range
        """
        if 'times' not in self.results or 'progress' not in self.results:
            raise ValueError("No results available. Run compute_progress_trajectory first.")
        
        times = self.results['times']
        progress_values = self.results['progress']
        
        # Check if time is within the computed range
        if time < times[0] or time > times[-1]:
            raise ValueError(f"Time {time} is outside computed range [{times[0]:.3f}, {times[-1]:.3f}]")
        
        # Use numpy interpolation
        interpolated_progress = np.interp(time, times, progress_values)
        
        logger.debug(f"Interpolated progress at time {time}: {interpolated_progress:.6f}")
        
        return float(interpolated_progress)
    
    def compute_aggregate_taste_with_sd_schedule(self, progress: float, sd_per_progress_unit: float = 0.5) -> float:
        """
        Compute aggregate research taste with AI research taste specified in standard deviations.
        
        This is a convenience method that implements the pattern:
        ai_research_taste = taste_dist.get_taste_at_sd(sd_per_progress_unit * progress)
        aggregate_taste = taste_dist.get_mean_with_floor(ai_research_taste)
        
        Args:
            progress: Current progress level
            sd_per_progress_unit: How many standard deviations AI research taste grows per progress unit
            
        Returns:
            Aggregate research taste with the AI floor applied
            
        Example:
            # AI research taste grows at 0.5 SD per progress unit
            model = ProgressModel(params, data) 
            progress = 10.0
            aggregate_taste = model.compute_aggregate_taste_with_sd_schedule(progress, 0.5)
        """
        ai_research_taste = self.taste_distribution.get_taste_at_sd(sd_per_progress_unit * progress)
        return self.taste_distribution.get_mean_with_floor(ai_research_taste)


def load_time_series_data(filename: str) -> TimeSeriesData:
    """Load time series data from CSV file"""
    import csv
    
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)
    
    time = np.array([float(row['time']) for row in data])
    L_HUMAN = np.array([float(row['L_HUMAN']) for row in data])
    inference_compute = np.array([float(row['inference_compute']) for row in data])
    experiment_compute = np.array([float(row['experiment_compute']) for row in data])
    training_compute_growth_rate = np.array([float(row['training_compute_growth_rate']) for row in data])
    
    return TimeSeriesData(time, L_HUMAN, inference_compute, experiment_compute, training_compute_growth_rate)