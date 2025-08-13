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
from typing import List, Tuple, Optional, Dict, Any, Union
from scipy import optimize, integrate, interpolate
import logging
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
    L_AI: np.ndarray  # AI labor supply (human-equivalents)
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
                 median_to_top_gap: float = cfg.MEDIAN_TO_TOP_TASTE_GAP,
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


@dataclass
class Parameters:
    """Model parameters with validation"""

    human_only: bool = field(default_factory=lambda: False)
    
    # Production function parameters
    rho_cognitive: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['rho_cognitive'])
    rho_progress: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['rho_progress'])
    alpha: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['alpha'])
    software_scale: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['software_scale'])
    zeta: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['zeta'])
    
    # Automation sigmoid parameters
    automation_fraction_at_superhuman_coder: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_fraction_at_superhuman_coder'])
    progress_at_half_sc_automation: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['progress_at_half_sc_automation'])
    automation_slope: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_slope'])
    automation_anchors: Optional[Dict[float, float]] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_anchors'])
    swe_multiplier_at_anchor_time: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['swe_multiplier_at_anchor_time'])
    # AI Research Taste sigmoid parameters
    ai_research_taste_at_superhuman_coder: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['ai_research_taste_at_superhuman_coder'])
    # Optional: allow specifying the superhuman-coder taste as SD within the human range
    ai_research_taste_at_superhuman_coder_sd: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('ai_research_taste_at_superhuman_coder_sd'))
    progress_at_half_ai_research_taste: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['progress_at_half_ai_research_taste'])
    ai_research_taste_slope: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['ai_research_taste_slope'])
    taste_schedule_type: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['taste_schedule_type'])
    progress_at_sc: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('progress_at_sc'))
    sc_time_horizon_minutes: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['sc_time_horizon_minutes'])
    horizon_extrapolation_type: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['horizon_extrapolation_type'])
    
    # Manual horizon fitting parameters
    anchor_time: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['anchor_time'])
    anchor_horizon: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['anchor_horizon'])
    anchor_doubling_time: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['anchor_doubling_time'])
    doubling_decay_rate: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['doubling_decay_rate'])
    
    # Normalization
    cognitive_output_normalization: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['cognitive_output_normalization'])
    
    # Baseline Annual Compute Multiplier
    baseline_annual_compute_multiplier: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['baseline_annual_compute_multiplier'])
    
    # Lambda parameter for CES output transformation
    lambda_param: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['lambda'])
    
    def __post_init__(self):
        """Validate and sanitize parameters after initialization"""
        # Sanitize elasticity parameters
        if not np.isfinite(self.rho_cognitive):
            logger.warning(f"Non-finite rho_cognitive: {self.rho_cognitive}, setting to 0")
            self.rho_cognitive = 0.0
        else:
            # Standard CES rho is in (-inf, 1]. We clamp to a reasonable range.
            # rho -> 0 is Cobb-Douglas, rho -> 1 is perfect substitutes.
            self.rho_cognitive = np.clip(self.rho_cognitive, cfg.RHO_CLIP_MIN, 1.0)
        
        if not np.isfinite(self.rho_progress):
            logger.warning(f"Non-finite rho_progress: {self.rho_progress}, setting to 0")
            self.rho_progress = 0.0
        else:
            # Standard CES rho is in (-inf, 1]. We clamp to a reasonable range.
            # rho -> 0 is Cobb-Douglas, rho -> 1 is perfect substitutes.
            self.rho_progress = np.clip(self.rho_progress, cfg.RHO_CLIP_MIN, 1.0)
        
        # Sanitize weights
        if not np.isfinite(self.alpha):
            logger.warning(f"Non-finite alpha: {self.alpha}, setting to 0.5")
            self.alpha = 0.5
        else:
            self.alpha = np.clip(self.alpha, cfg.PARAM_CLIP_MIN, 1.0 - cfg.PARAM_CLIP_MIN)
        
        if not np.isfinite(self.software_scale):
            logger.warning(f"Non-finite software_scale: {self.software_scale}, setting to 1.0")
            self.software_scale = 1.0
        else:
            self.software_scale = np.clip(self.software_scale, 0.1, 10.0)
        
        if not np.isfinite(self.zeta):
            logger.warning(f"Non-finite zeta: {self.zeta}, setting to {cfg.DEFAULT_PARAMETERS['zeta']}")
            self.zeta = cfg.DEFAULT_PARAMETERS['zeta']
        else:
            self.zeta = np.clip(self.zeta, cfg.ZETA_CLIP_MIN, cfg.ZETA_CLIP_MAX)
        
        # Sanitize automation parameters
        if not np.isfinite(self.automation_fraction_at_superhuman_coder):
            logger.warning(f"Non-finite automation_fraction_at_superhuman_coder: {self.automation_fraction_at_superhuman_coder}, setting to {cfg.DEFAULT_PARAMETERS['automation_fraction_at_superhuman_coder']}")
            self.automation_fraction_at_superhuman_coder = cfg.DEFAULT_PARAMETERS['automation_fraction_at_superhuman_coder']
        else:
            self.automation_fraction_at_superhuman_coder = np.clip(self.automation_fraction_at_superhuman_coder, cfg.PARAM_CLIP_MIN, 1.0 - cfg.PARAM_CLIP_MIN)
        
        if not np.isfinite(self.progress_at_half_sc_automation) or self.progress_at_half_sc_automation <= 0:
            logger.warning(f"Invalid progress_at_half_sc_automation: {self.progress_at_half_sc_automation}, setting to {cfg.DEFAULT_PARAMETERS['progress_at_half_sc_automation']}")
            self.progress_at_half_sc_automation = cfg.DEFAULT_PARAMETERS['progress_at_half_sc_automation']
        else:
            self.progress_at_half_sc_automation = max(cfg.PARAM_CLIP_MIN, self.progress_at_half_sc_automation)
        
        if not np.isfinite(self.automation_slope):
            logger.warning(f"Non-finite automation_slope: {self.automation_slope}, setting to 1.0")
            self.automation_slope = 1.0
        else:
            # Clamp slope to reasonable range to prevent numerical instability
            self.automation_slope = np.clip(self.automation_slope, cfg.AUTOMATION_SLOPE_CLIP_MIN, cfg.AUTOMATION_SLOPE_CLIP_MAX)
        
        # Sanitize AI research taste parameters
        # If SD-specification is provided, convert to raw taste via TasteDistribution
        if self.ai_research_taste_at_superhuman_coder_sd is not None and np.isfinite(self.ai_research_taste_at_superhuman_coder_sd):
            try:
                taste_distribution_tmp = TasteDistribution()
                converted_taste = taste_distribution_tmp.get_taste_at_sd(float(self.ai_research_taste_at_superhuman_coder_sd))
                if np.isfinite(converted_taste):
                    self.ai_research_taste_at_superhuman_coder = float(converted_taste)
            except Exception as e:
                logger.warning(f"Failed converting ai_research_taste_at_superhuman_coder_sd to taste: {e}")
        if not np.isfinite(self.ai_research_taste_at_superhuman_coder):
            logger.warning(f"Non-finite ai_research_taste_at_superhuman_coder: {self.ai_research_taste_at_superhuman_coder}, setting to {cfg.DEFAULT_PARAMETERS['ai_research_taste_at_superhuman_coder']}")
            self.ai_research_taste_at_superhuman_coder = cfg.DEFAULT_PARAMETERS['ai_research_taste_at_superhuman_coder']
        else:
            self.ai_research_taste_at_superhuman_coder = np.clip(self.ai_research_taste_at_superhuman_coder, cfg.PARAM_CLIP_MIN, cfg.AI_RESEARCH_TASTE_MAX)
        
        if not np.isfinite(self.progress_at_half_ai_research_taste) or self.progress_at_half_ai_research_taste <= 0:
            logger.warning(f"Invalid progress_at_half_ai_research_taste: {self.progress_at_half_ai_research_taste}, setting to {cfg.DEFAULT_PARAMETERS['progress_at_half_ai_research_taste']}")
            self.progress_at_half_ai_research_taste = cfg.DEFAULT_PARAMETERS['progress_at_half_ai_research_taste']
        else:
            self.progress_at_half_ai_research_taste = max(cfg.PARAM_CLIP_MIN, self.progress_at_half_ai_research_taste)
        
        if not np.isfinite(self.ai_research_taste_slope):
            logger.warning(f"Non-finite ai_research_taste_slope: {self.ai_research_taste_slope}, setting to 1.0")
            self.ai_research_taste_slope = 1.0
        else:
            # Clamp slope to reasonable range to prevent numerical instability
            self.ai_research_taste_slope = np.clip(self.ai_research_taste_slope, cfg.AUTOMATION_SLOPE_CLIP_MIN, cfg.AUTOMATION_SLOPE_CLIP_MAX)
        
        # Sanitize time horizon parameter
        if not np.isfinite(self.sc_time_horizon_minutes) or self.sc_time_horizon_minutes <= 0:
            logger.warning(f"Invalid sc_time_horizon_minutes: {self.sc_time_horizon_minutes}, setting to {cfg.DEFAULT_PARAMETERS['sc_time_horizon_minutes']}")
            self.sc_time_horizon_minutes = cfg.DEFAULT_PARAMETERS['sc_time_horizon_minutes']
        else:
            # Clamp to reasonable bounds
            self.sc_time_horizon_minutes = np.clip(self.sc_time_horizon_minutes, cfg.PARAMETER_BOUNDS['sc_time_horizon_minutes'][0], cfg.PARAMETER_BOUNDS['sc_time_horizon_minutes'][1])
        
        # Sanitize categorical parameters
        if self.horizon_extrapolation_type not in cfg.HORIZON_EXTRAPOLATION_TYPES:
            logger.warning(f"Invalid horizon_extrapolation_type: {self.horizon_extrapolation_type}, setting to default")
            self.horizon_extrapolation_type = cfg.DEFAULT_HORIZON_EXTRAPOLATION_TYPE
        
        # Sanitize manual horizon fitting parameters
        if not np.isfinite(self.anchor_time):
            logger.warning(f"Non-finite anchor_time: {self.anchor_time}, setting to default")
            self.anchor_time = cfg.DEFAULT_ANCHOR_TIME
        else:
            # Clamp to reasonable time range
            self.anchor_time = np.clip(self.anchor_time, 2020.0, 2030.0)
        
        # Validate optional parameters - if provided, ensure they're finite and positive
        if self.anchor_horizon is not None:
            if not np.isfinite(self.anchor_horizon) or self.anchor_horizon <= 0:
                logger.warning(f"Invalid anchor_horizon: {self.anchor_horizon}, setting to None for optimization")
                self.anchor_horizon = None
            else:
                self.anchor_horizon = np.clip(self.anchor_horizon, 0.01, 10000.0)
        
        if self.anchor_doubling_time is not None:
            if not np.isfinite(self.anchor_doubling_time) or self.anchor_doubling_time <= 0:
                logger.warning(f"Invalid anchor_doubling_time: {self.anchor_doubling_time}, setting to None for optimization")
                self.anchor_doubling_time = None
            else:
                self.anchor_doubling_time = np.clip(self.anchor_doubling_time, 0.01, 100.0)
        
        if self.doubling_decay_rate is not None:
            if not np.isfinite(self.doubling_decay_rate) or self.doubling_decay_rate <= 0:
                logger.warning(f"Invalid doubling_decay_rate: {self.doubling_decay_rate}, setting to None for optimization")
                self.doubling_decay_rate = None
            else:
                self.doubling_decay_rate = np.clip(self.doubling_decay_rate, 0.001, 1.0)

        # Sanitize normalization parameters
        if not np.isfinite(self.cognitive_output_normalization) or self.cognitive_output_normalization <= 0:
            logger.warning(f"Invalid cognitive_output_normalization: {self.cognitive_output_normalization}, setting to 1.0")
            self.cognitive_output_normalization = 1.0
        else:
            self.cognitive_output_normalization = max(cfg.NORMALIZATION_MIN, self.cognitive_output_normalization)
        
        # Validate baseline annual compute multiplier
        if not np.isfinite(self.baseline_annual_compute_multiplier) or self.baseline_annual_compute_multiplier <= 0:
            logger.warning(f"Invalid baseline_annual_compute_multiplier: {self.baseline_annual_compute_multiplier}, setting to default")
            self.baseline_annual_compute_multiplier = cfg.BASELINE_ANNUAL_COMPUTE_MULTIPLIER_DEFAULT
        else:
            # Ensure it's within reasonable bounds
            bounds = cfg.PARAMETER_BOUNDS.get('baseline_annual_compute_multiplier', (1.0, 20.0))
            self.baseline_annual_compute_multiplier = np.clip(self.baseline_annual_compute_multiplier, bounds[0], bounds[1])
        
        # Validate lambda parameter
        if not np.isfinite(self.lambda_param) or self.lambda_param < 0:
            logger.warning(f"Invalid lambda: {self.lambda_param}, setting to default")
            self.lambda_param = cfg.DEFAULT_PARAMETERS['lambda']
        else:
            # Ensure it's within bounds
            bounds = cfg.PARAMETER_BOUNDS.get('lambda', (0.0, 2.0))
            self.lambda_param = np.clip(self.lambda_param, bounds[0], bounds[1])


@dataclass
class AnchorConstraint:
    """Specifies a constraint for parameter estimation"""
    # Dict mapping variable names to values (can be partial)
    conditions: Dict[str, float]  # e.g., {"automation_fraction": 0.9, "L_AI": 1e6}
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


def compute_cognitive_output(automation_fraction: float, L_AI: float, L_HUMAN: float, rho: float, lambda_param: float, cognitive_normalization: float = 1.0, human_only: bool = False) -> float:
    """
    CES combination of AI and human labor using an alternative formulation inspired by
    the structure in FORMULAS.md: Y = ( (A^(1-rho) * L_AI^rho) + ((1-A)^(1-rho) * L_HUMAN^rho) )^(1/rho)
    This can be seen as a standard CES function on effective labor inputs L_AI/A and L_HUMAN/(1-A).
    
    Args:
        automation_fraction: Fraction of work automated (A) [0,1]
        L_AI: AI labor supply
        L_HUMAN: Human labor supply
        rho: Standard substitution parameter in (-inf, 1].
             rho -> 1: perfect substitutes (Y = L_AI + L_HUMAN)
             rho -> 0: Cobb-Douglas (Y = (L_AI/A)^A * (L_HUMAN/(1-A))^(1-A))
             rho -> -inf: perfect complements (Y = min(L_AI/A, L_HUMAN/(1-A)))
        lambda_param: Power transformation parameter applied to CES output before normalization
        cognitive_normalization: Normalization constant for cognitive output
    
    Returns:
        Cognitive output
    """
    if human_only:
        return L_HUMAN ** lambda_param * cognitive_normalization 
    
    # Input validation
    if not all(np.isfinite([automation_fraction, L_AI, L_HUMAN, rho, lambda_param, cognitive_normalization])):
        logger.warning("Non-finite inputs to compute_cognitive_output")
        return 0.0
    
    if L_AI < 0 or L_HUMAN < 0:
        logger.warning("Negative labor inputs")
        return 0.0
    
    # Clamp automation fraction to valid range, avoiding 0 and 1 for division.
    a = np.clip(automation_fraction, cfg.AUTOMATION_FRACTION_CLIP_MIN, 1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN)

    # Handle edge cases for rho
    if abs(rho) < cfg.RHO_COBB_DOUGLAS_THRESHOLD:  # Cobb-Douglas case
        if L_AI > 0 and L_HUMAN > 0:
            try:
                # Effective inputs are L_AI/a and L_HUMAN/(1-a)
                # Cobb-Douglas form is (L_AI/a)^a * (L_HUMAN/(1-a))^(1-a)
                term1 = a * (np.log(L_AI) - np.log(a))
                term2 = (1 - a) * (np.log(L_HUMAN) - np.log(1 - a))
                log_result = term1 + term2
                result = np.exp(log_result)
            except (ValueError, OverflowError):
                logger.warning("Overflow in Cobb-Douglas case, using linear fallback")
                result = a * L_AI + (1-a) * L_HUMAN
        else:
            result = 0.0

    elif rho == 1.0:  # Perfect substitutes
        # Formula simplifies to L_AI + L_HUMAN
        result = L_AI + L_HUMAN

    elif rho < cfg.RHO_LEONTIEF_THRESHOLD:  # Nearly perfect complements (Leontief)
        # Limit is min(L_AI/a, L_HUMAN/(1-a))
        result = min(L_AI / a, L_HUMAN / (1 - a))

    else: # Standard CES formula for the alternative model
        try:
            term1 = np.power(a, 1 - rho) * np.power(L_AI, rho) if L_AI > 0 else 0
            term2 = np.power(1 - a, 1 - rho) * np.power(L_HUMAN, rho) if L_HUMAN > 0 else 0
            
            total = term1 + term2
            if total <= 0:
                logger.warning(f"Alternative CES inner term is non-positive ({total}) with rho={rho}. Fallback to min.")
                result = min(L_AI / a, L_HUMAN / (1 - a))
            else:
                result = np.power(total, 1 / rho)

            if not np.isfinite(result):
                logger.warning(f"Non-finite result in Alternative CES (rho={rho}), using linear fallback.")
                result = a * L_AI + (1-a) * L_HUMAN
        
        except (OverflowError, ZeroDivisionError, ValueError) as e:
            logger.warning(f"Numerical error in Alternative CES (rho={rho}): {e}, using linear fallback.")
            result = a * L_AI + (1-a) * L_HUMAN
            
    # Apply lambda transformation before normalization
    try:
        result_with_lambda = np.power(result, lambda_param)
        if not np.isfinite(result_with_lambda):
            logger.warning(f"Non-finite result after lambda transformation (lambda={lambda_param}), using original result")
            result_with_lambda = result
    except (OverflowError, ValueError) as e:
        logger.warning(f"Error applying lambda transformation (lambda={lambda_param}): {e}, using original result")
        result_with_lambda = result
    
    return result_with_lambda * cognitive_normalization


def compute_research_stock_rate(experiment_compute: float, cognitive_output: float, alpha: float, rho: float, zeta: float, aggregate_research_taste: float = cfg.AGGREGATE_RESEARCH_TASTE_BASELINE) -> float:
    """
    CES combination of compute and cognitive work to determine research stock growth rate.
    This replaces the previous direct software progress calculation.
    
    Args:
        experiment_compute: Experiment compute budget
        cognitive_output: Output from cognitive work
        alpha: Weight on experiment compute [0,1]
        rho: Standard substitution parameter in (-inf, 1].
             rho -> 1: perfect substitutes
             rho -> 0: Cobb-Douglas
             rho -> -inf: perfect complements
        zeta: Discounting factor for experiment compute (see cfg.ZETA_CLIP_MIN, cfg.ZETA_CLIP_MAX)
        aggregate_research_taste: Multiplier for research effectiveness (default 1.0)
    
    Returns:
        Research stock growth rate RS'(t)
    """
    # Input validation
    if not all(np.isfinite([experiment_compute, cognitive_output, alpha, rho, zeta, aggregate_research_taste])):
        logger.warning("Non-finite inputs to compute_research_stock_rate")
        return 0.0
    
    if experiment_compute < 0 or cognitive_output < 0:
        logger.warning("Negative inputs to compute_research_stock_rate")
        return 0.0
    
    if aggregate_research_taste < 0:
        logger.warning(f"Negative aggregate_research_taste: {aggregate_research_taste}, setting to {cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK}")
        aggregate_research_taste = cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK
    
    if zeta < cfg.ZETA_CLIP_MIN or zeta > cfg.ZETA_CLIP_MAX:
        logger.warning(f"Invalid zeta value {zeta}, clamping to [{cfg.ZETA_CLIP_MIN}, {cfg.ZETA_CLIP_MAX}]")
        zeta = np.clip(zeta, cfg.ZETA_CLIP_MIN, cfg.ZETA_CLIP_MAX)
    
    # Clamp alpha to valid range
    alpha = np.clip(alpha, 0.0, 1.0)
    
    # Apply discounting factor to experiment compute
    discounted_experiment_compute = np.power(experiment_compute, zeta)
    
    # Use the generic CES function for computation
    rate = _ces_function(discounted_experiment_compute, cognitive_output, alpha, rho)
    
    # Cap extremely large rates to prevent numerical issues
    if rate > cfg.MAX_RESEARCH_STOCK_RATE:
        logger.warning(f"Very large research stock rate {rate}, capping to {cfg.MAX_RESEARCH_STOCK_RATE}")
        rate = cfg.MAX_RESEARCH_STOCK_RATE
    
    # Apply aggregate research taste multiplier
    final_rate = rate * aggregate_research_taste
    
    # Apply final cap to prevent numerical issues with the multiplied result
    if final_rate > cfg.MAX_RESEARCH_STOCK_RATE:
        logger.warning(f"Very large final research stock rate {final_rate}, capping to {cfg.MAX_RESEARCH_STOCK_RATE}")
        final_rate = cfg.MAX_RESEARCH_STOCK_RATE
        
    return final_rate


def compute_software_progress_rate(research_stock: float, research_stock_rate: float, 
                                 initial_research_stock: float, initial_research_stock_rate: float,
                                 software_scale: float) -> float:
    """
    Compute software progress rate using research stock formulation:
    S(t) = RS'(t) / RS(t) * s
    
    Args:
        research_stock: Current research stock RS(t)
        research_stock_rate: Current research stock rate RS'(t)
        software_scale: Software progress share parameter s [0.1,10]
    
    Returns:
        Software progress rate multiplied by s
    """
    # Input validation
    if not all(np.isfinite([research_stock, research_stock_rate, initial_research_stock, initial_research_stock_rate, software_scale])):
        logger.warning("Non-finite inputs to compute_software_progress_rate")
        return 0.0
    
    if research_stock <= 0 or initial_research_stock <= 0:
        logger.warning("Non-positive research stock values")
        return 0.0
    
    if initial_research_stock_rate <= 0:
        logger.warning("Non-positive initial research stock rate")
        return 0.0
    
    if software_scale < 0.1 or software_scale > 10:
        logger.warning(f"Invalid software_scale: {software_scale}, must be in [0.1,10]")
        return 0.0
    
    # Compute software progress rate using research stock ratio formula
    try:
        numerator = research_stock_rate
        denominator = research_stock
        
        if denominator == 0:
            logger.warning("Zero denominator in software progress rate calculation")
            return 0.0
        
        software_progress_rate = numerator / denominator
        
        if not np.isfinite(software_progress_rate) or software_progress_rate < 0:
            logger.warning(f"Invalid software progress rate: {software_progress_rate}")
            return 0.0
        # Apply software progress share multiplier: s
        final_rate = software_progress_rate * software_scale
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
        L_AI_0 = _log_interp(start_time, time_series_data.time, time_series_data.L_AI)
        experiment_compute_0 = _log_interp(start_time, time_series_data.time, time_series_data.experiment_compute)
        
        if params.human_only:
            cognitive_output_0 = compute_cognitive_output(None, L_AI_0, L_HUMAN_0, params.rho_cognitive, params.lambda_param, params.cognitive_output_normalization, human_only=True)
            logger.info(f"HUMAN-ONLY::: cognitive_output_0: {cognitive_output_0}")
            initial_aggregate_research_taste = 1.0
        else:
            initial_automation = compute_automation_fraction(initial_progress, params)
            initial_ai_research_taste = compute_ai_research_taste(initial_progress, params)
            initial_aggregate_research_taste = compute_aggregate_research_taste(initial_ai_research_taste)
            cognitive_output_0 = compute_cognitive_output(
                initial_automation, L_AI_0, L_HUMAN_0, 
                params.rho_cognitive, params.lambda_param, params.cognitive_output_normalization
            )
            logger.info(f"ACTUAL::: cognitive_output_0: {cognitive_output_0}")
        
        # Calculate RS'(0)
        rs_rate_0 = compute_research_stock_rate(
            experiment_compute_0, cognitive_output_0, 
            params.alpha, params.rho_progress, params.zeta, initial_aggregate_research_taste
        )
        
        # Calculate RS'(dt) for numerical differentiation
        # Use log-space interpolation for exponential trends
        L_HUMAN_dt = _log_interp(start_time + dt, time_series_data.time, time_series_data.L_HUMAN)
        L_AI_dt = _log_interp(start_time + dt, time_series_data.time, time_series_data.L_AI)
        experiment_compute_dt = _log_interp(start_time + dt, time_series_data.time, time_series_data.experiment_compute)
        
        # Automation fraction changes very little over small dt, so use same value
        if params.human_only:
            cognitive_output_dt = compute_cognitive_output(None, L_AI_dt, L_HUMAN_dt, params.rho_cognitive, params.lambda_param, params.cognitive_output_normalization, human_only=True)
            logger.info(f"HUMAN-ONLY::: cognitive_output_dt: {cognitive_output_dt}")
        else:
            cognitive_output_dt = compute_cognitive_output(
                initial_automation, L_AI_dt, L_HUMAN_dt,
                params.rho_cognitive, params.lambda_param, params.cognitive_output_normalization
            )
        
        rs_rate_dt = compute_research_stock_rate(
            experiment_compute_dt, cognitive_output_dt,
            params.alpha, params.rho_progress, params.zeta, initial_aggregate_research_taste
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
    L_AI: float
    experiment_compute: float
    training_compute_growth_rate: float
    cognitive_output: float
    research_stock_rate: float
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
    """
    start_time = time_series_data.time[0]
    
    # TODO: may need to change to log-space interpolation
    L_HUMAN = np.interp(start_time, time_series_data.time, time_series_data.L_HUMAN)
    L_AI = np.interp(start_time, time_series_data.time, time_series_data.L_AI)
    experiment_compute = np.interp(start_time, time_series_data.time, time_series_data.experiment_compute)
    training_compute_growth_rate = np.interp(start_time, time_series_data.time, time_series_data.training_compute_growth_rate)

    if params.human_only:
        initial_automation = 1.0
        initial_ai_research_taste = 0.0
        initial_aggregate_research_taste = 1.0
        cognitive_output = compute_cognitive_output(None, L_AI, L_HUMAN, params.rho_cognitive, params.lambda_param, params.cognitive_output_normalization, human_only=True)
        research_stock_rate = compute_research_stock_rate(experiment_compute, cognitive_output, params.alpha, params.rho_progress, params.zeta, initial_aggregate_research_taste)
    else:
        initial_automation = compute_automation_fraction(initial_progress, params)
        initial_ai_research_taste = compute_ai_research_taste(initial_progress, params)
        initial_aggregate_research_taste = compute_aggregate_research_taste(initial_ai_research_taste)
        cognitive_output = compute_cognitive_output(
            initial_automation, L_AI, L_HUMAN, 
            params.rho_cognitive, params.lambda_param, params.cognitive_output_normalization
        )
    
        research_stock_rate = compute_research_stock_rate(
            experiment_compute, cognitive_output, 
            params.alpha, params.rho_progress, params.zeta, initial_aggregate_research_taste
        )
    
    # Validate and fallback for research stock rate
    if not np.isfinite(research_stock_rate) or research_stock_rate <= 0:
        logger.warning(f"Invalid initial research stock rate ({research_stock_rate}), using fallback 1.0")
        research_stock_rate = 1.0
    
    # Calculate initial research stock
    research_stock = calculate_initial_research_stock(time_series_data, params, initial_progress)
    
    return InitialConditions(
        start_time=start_time,
        initial_progress=initial_progress,
        initial_automation=initial_automation,
        L_HUMAN=L_HUMAN,
        L_AI=L_AI,
        experiment_compute=experiment_compute,
        training_compute_growth_rate=training_compute_growth_rate,
        cognitive_output=cognitive_output,
        research_stock_rate=research_stock_rate,
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

def aut_frac_from_swe_multiplier(swe_multiplier: float, L_HUMAN: float, L_AI: float, params: Parameters) -> float:
    """
    Compute automation fraction from swe multiplier.

    Solve for A in:
      (swe_multiplier)**params.lambda_param * compute_cognitive_output(A, L_AI, L_HUMAN, params.rho_cognitive, params.lambda_param, params.cognitive_output_normalization, human_only=True) = compute_cognitive_output(A, L_AI, L_HUMAN, params.rho_cognitive, params.lambda_param, params.cognitive_output_normalization)
    where p = params.rho_cognitive.
    Returns A in (0, 1). If there are multiple solutions, return the lower one.

    """
    # Input validation
    if not all(np.isfinite([swe_multiplier, L_HUMAN, L_AI])):
        logger.warning("Non-finite inputs to aut_frac_from_swe_multiplier")
        return 0.0
    
    if swe_multiplier <= 0 or L_HUMAN <= 0 or L_AI < 0:
        logger.warning("Invalid inputs to aut_frac_from_swe_multiplier")
        return 0.0
    
    # Target value we want to achieve
    target_output = swe_multiplier**params.lambda_param * compute_cognitive_output(0, L_AI, L_HUMAN, params.rho_cognitive, params.lambda_param, params.cognitive_output_normalization, human_only=True)
    
    # Define the objective function to minimize
    def objective(A_candidate):
        """Return the difference between target and actual cognitive output"""
        try:
            actual_output = compute_cognitive_output(
                A_candidate, L_AI, L_HUMAN, 
                params.rho_cognitive, params.lambda_param, params.cognitive_output_normalization
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


def compute_automation_fraction(cumulative_progress: float, params: Parameters) -> float:
    """
    Exponential (log-space) interpolation for automation fraction based on cumulative progress.
    
    Uses log-space interpolation between two anchor points in automation_anchors.
    Extrapolates beyond these points but clips at 1.0.
    
    Args:
        cumulative_progress: Current cumulative progress.
        params: Model parameters containing automation_anchors.
    
    Returns:
        Automation fraction in [0, 1].
    """
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
    if automation_1 <= 0.0 or automation_1 >= 1.0:
        logger.warning(f"automation_fraction at progress {progress_1} must be in (0, 1), got {automation_1}")
        automation_1 = np.clip(automation_1, 1e-6, 1.0 - 1e-6)
    
    if automation_2 <= 0.0 or automation_2 >= 1.0:
        logger.warning(f"automation_fraction at progress {progress_2} must be in (0, 1), got {automation_2}")
        automation_2 = np.clip(automation_2, 1e-6, 1.0 - 1e-6)
    
    if progress_1 >= progress_2:
        logger.error(f"Progress values must be distinct and ordered: {progress_1} >= {progress_2}")
        assert False, "Progress anchor points must be distinct"
    
    try:
        # Log-space interpolation: log(automation) = a * progress + b
        # Solve for a and b using the two anchor points
        log_automation_1 = np.log(automation_1)
        log_automation_2 = np.log(automation_2)
        
        # Linear interpolation in log space: log(y) = a*x + b
        a = (log_automation_2 - log_automation_1) / (progress_2 - progress_1)
        b = log_automation_1 - a * progress_1
        
        # Calculate log(automation) at current progress
        log_automation = a * cumulative_progress + b
        
        # Convert back from log space
        automation_fraction = np.exp(log_automation)
        
    except (ValueError, OverflowError) as e:
        logger.warning(f"Numerical error in log-space interpolation: {e}")
        # Fallback: linear interpolation
        if progress_1 <= cumulative_progress <= progress_2:
            # Linear interpolation between anchor points
            t = (cumulative_progress - progress_1) / (progress_2 - progress_1)
            automation_fraction = automation_1 + t * (automation_2 - automation_1)
        elif cumulative_progress < progress_1:
            # Extrapolate below first anchor
            slope = (automation_2 - automation_1) / (progress_2 - progress_1)
            automation_fraction = automation_1 + slope * (cumulative_progress - progress_1)
        else:  # cumulative_progress > progress_2
            # Extrapolate above second anchor
            slope = (automation_2 - automation_1) / (progress_2 - progress_1)
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
        return _compute_ai_research_taste_sigmoid(cumulative_progress, params)
    elif params.taste_schedule_type == "exponential":
        return _compute_ai_research_taste_exponential(cumulative_progress, params)
    elif params.taste_schedule_type == "sd_per_progress":
        return _compute_ai_research_taste_sd_per_progress(cumulative_progress, params)
    else:
        logger.warning(f"Unknown taste_schedule_type: {params.taste_schedule_type}, defaulting to sigmoid")
        return _compute_ai_research_taste_sigmoid(cumulative_progress, params)


def _compute_ai_research_taste_sigmoid(cumulative_progress: float, params: Parameters) -> float:
    """
    Sigmoid function for AI research taste: f(x) = L / (1 + e^(-k*(x-x0)))
    """
    # Extract sigmoid parameters
    L = params.ai_research_taste_at_superhuman_coder  # Upper asymptote
    x0 = params.progress_at_half_ai_research_taste  # Midpoint (where taste = L/2)
    k = params.ai_research_taste_slope  # Slope parameter
    
    # Calculate sigmoid: f(x) = L / (1 + e^(-k*(x-x0)))
    try:
        exponent = -k * (cumulative_progress - x0)
        
        # Handle extreme exponents to prevent overflow/underflow
        if exponent > cfg.SIGMOID_EXPONENT_CLAMP:  # e^100 is very large, sigmoid ≈ 0
            ai_research_taste = 0.0
        elif exponent < -cfg.SIGMOID_EXPONENT_CLAMP:  # e^(-100) is very small, sigmoid ≈ L
            ai_research_taste = L
        else:
            ai_research_taste = L / (1 + np.exp(exponent))
            
    except (OverflowError, ValueError) as e:
        logger.warning(f"Numerical error in AI research taste sigmoid calculation: {e}")
        # Fallback: linear interpolation between 0 and L
        if cumulative_progress <= x0:
            ai_research_taste = L * 0.5 * (cumulative_progress / x0)
        else:
            ai_research_taste = L * (0.5 + 0.5 * (cumulative_progress - x0) / x0)
    
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
        offset = target_sd - slope * progress_at_sc
        
        # Compute AI research taste at current progress
        penalty = (cfg.AI_RESEARCH_TASTE_MAX_SD - target_sd) / cfg.AI_RESEARCH_TASTE_MAX_SD
        current_sd = slope * penalty * cumulative_progress + offset
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
                                   median_to_top_gap: float = cfg.MEDIAN_TO_TOP_TASTE_GAP,
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
        median_to_top_gap = cfg.MEDIAN_TO_TOP_TASTE_GAP
    
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
                         initial_research_stock_rate: Optional[float] = None, 
                         initial_research_stock: Optional[float] = None) -> List[float]:
    """
    Compute instantaneous rates for both progress and research stock.
    This is the RHS of the coupled differential equation system.
    
    Args:
        t: Current time
        state: [cumulative_progress, research_stock]
        time_series_data: Input time series
        params: Model parameters
        initial_research_stock_rate: RS'(0) needed for software progress calculation
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
        
        if np.all(time_series_data.L_AI > 0):
            log_L_AI = np.log(time_series_data.L_AI)
            L_AI = np.exp(np.interp(t, time_series_data.time, log_L_AI))
        else:
            L_AI = np.interp(t, time_series_data.time, time_series_data.L_AI)
        
        if np.all(time_series_data.experiment_compute > 0):
            log_experiment_compute = np.log(time_series_data.experiment_compute)
            experiment_compute = np.exp(np.interp(t, time_series_data.time, log_experiment_compute))
        else:
            experiment_compute = np.interp(t, time_series_data.time, time_series_data.experiment_compute)
        
        training_compute_growth_rate = np.interp(t, time_series_data.time, time_series_data.training_compute_growth_rate)
        
        # Validate interpolated values
        if not all(np.isfinite([L_HUMAN, L_AI, experiment_compute, training_compute_growth_rate])):
            logger.warning(f"Non-finite interpolated values at t={t}")
            return [0.0, 0.0]
        
        # Ensure non-negative values
        L_HUMAN = max(0.0, L_HUMAN)
        L_AI = max(0.0, L_AI)
        experiment_compute = max(0.0, experiment_compute)
        training_compute_growth_rate = max(0.0, training_compute_growth_rate)
        
        if params.human_only:
            automation_fraction = 0.0
            aggregate_research_taste = 1.0
            cognitive_output = compute_cognitive_output(None, L_AI, L_HUMAN, params.rho_cognitive, params.lambda_param, params.cognitive_output_normalization, human_only=True)  
        else:
            # Compute automation fraction from cumulative progress
            automation_fraction = compute_automation_fraction(cumulative_progress, params)
            if not (0 <= automation_fraction <= 1):
                logger.warning(f"Invalid automation fraction {automation_fraction} at progress {cumulative_progress}")
                automation_fraction = np.clip(automation_fraction, 0.0, 1.0)
            
            # Compute AI research taste and aggregate research taste
            ai_research_taste = compute_ai_research_taste(cumulative_progress, params)
            aggregate_research_taste = compute_aggregate_research_taste(ai_research_taste)
            
            # Compute cognitive output with validation
            cognitive_output = compute_cognitive_output(
                automation_fraction, L_AI, L_HUMAN, params.rho_cognitive, params.lambda_param, params.cognitive_output_normalization
            )
        
        if not np.isfinite(cognitive_output) or cognitive_output < 0:
            logger.warning(f"Invalid cognitive output: {cognitive_output}")
            return [0.0, 0.0]
        
        # Compute research stock rate (dRS/dt) with validation
        research_stock_rate = compute_research_stock_rate(
            experiment_compute, cognitive_output, params.alpha, params.rho_progress, params.zeta, aggregate_research_taste
        )
        
        if not np.isfinite(research_stock_rate) or research_stock_rate < 0:
            logger.warning(f"Invalid research stock rate: {research_stock_rate}")
            return [0.0, 0.0]
        
        # Compute software progress rate using research stock formulation
        if initial_research_stock_rate is None or initial_research_stock_rate <= 0:
            logger.warning("No valid initial research stock rate provided, using fallback")
            # Fallback: use current rate as approximation
            software_progress_rate = research_stock_rate
        else:
            software_progress_rate = compute_software_progress_rate(
                research_stock, research_stock_rate, 
                initial_research_stock, initial_research_stock_rate,
                params.software_scale
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
                    f"automation={automation_fraction:.3f}, dP/dt={overall_rate:.3f}, dRS/dt={research_stock_rate:.3f}")
        
        return [overall_rate, research_stock_rate]
        
    except Exception as e:
        logger.error(f"Error computing rates at t={t}, state={state}: {e}")
        return [0.0, 0.0]  # Return zero rates on any error


def integrate_progress(time_range: List[float], initial_progress: float, time_series_data: TimeSeriesData, 
                      params: Parameters, direction: str = 'forward') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the coupled differential equation system with robust fallback methods:
    d(progress)/dt = progress_rate(t, progress, research_stock)
    d(research_stock)/dt = research_stock_rate(t, progress, research_stock)
    
    Args:
        time_range: [start_time, end_time]
        initial_progress: Initial cumulative progress
        time_series_data: Input time series
        params: Model parameters
        direction: 'forward' or 'backward'
    
    Returns:
        Tuple of (times, cumulative_progress_values, research_stock_values)
    """
    # Use utility function to compute initial conditions
    initial_conditions = compute_initial_conditions(time_series_data, params, initial_progress)
    initial_research_stock_rate = initial_conditions.research_stock_rate
    initial_research_stock = initial_conditions.research_stock
    
    def ode_func(t, y):
        try:
            rates = progress_rate_at_time(t, y, time_series_data, params, initial_research_stock_rate, initial_research_stock)
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
            
            rates = progress_rate_at_time(t, y, time_series_data, params, initial_research_stock_rate, initial_research_stock)
            
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
    methods_to_try = [
        ('RK45', {'rtol': 1e-6, 'atol': 1e-8}),      # Default high precision
        ('RK45', {'rtol': 1e-4, 'atol': 1e-6}),      # Relaxed precision
        ('RK23', {'rtol': 1e-4, 'atol': 1e-6}),      # Lower order method
        ('DOP853', {'rtol': 1e-3, 'atol': 1e-5}),    # Explicit method
        ('Radau', {'rtol': 1e-3, 'atol': 1e-5})      # Implicit method for stiff problems
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
    
    # If all methods fail, try simple Euler method as ultimate fallback
    if sol is None or not sol.success:
        assert False, "All scipy integration methods failed, ignoring Euler fallback and quitting"        
        try:
            # Simple Euler integration as last resort
            n_steps = max(cfg.EULER_FALLBACK_MIN_STEPS, int(abs(t_end - t_start) * cfg.EULER_FALLBACK_STEPS_PER_YEAR))  # Adaptive step count
            times = np.linspace(t_start, t_end, n_steps)
            dt = (t_end - t_start) / (n_steps - 1)
            
            # Log Euler step size information
            if cfg.ODE_STEP_SIZE_LOGGING:
                logger.info(f"Euler fallback integration: dt={dt:.2e}, n_steps={n_steps}, "
                           f"time_range=[{t_start:.2f}, {t_end:.2f}]")
            
            progress_values = np.zeros(n_steps)
            research_stock_values = np.zeros(n_steps)
            progress_values[0] = initial_progress
            research_stock_values[0] = initial_research_stock
            
            for i in range(1, n_steps):
                try:
                    state = [progress_values[i-1], research_stock_values[i-1]]
                    rates = progress_rate_at_time(times[i-1], state, time_series_data, params, initial_research_stock_rate, initial_research_stock)
                    
                    # Validate and clamp rates
                    for j in range(len(rates)):
                        if not np.isfinite(rates[j]) or rates[j] < 0:
                            rates[j] = 0.0
                    
                    progress_values[i] = progress_values[i-1] + rates[0] * dt
                    research_stock_values[i] = research_stock_values[i-1] + rates[1] * dt
                    
                    # Ensure values don't become too large
                    if progress_values[i] > cfg.PROGRESS_ODE_CLAMP_MAX:
                        progress_values[i] = cfg.PROGRESS_ODE_CLAMP_MAX
                    
                    if research_stock_values[i] <= 0:
                        research_stock_values[i] = max(research_stock_values[i-1], 1e-6)
                    elif research_stock_values[i] > cfg.RESEARCH_STOCK_ODE_CLAMP_MAX:
                        research_stock_values[i] = cfg.RESEARCH_STOCK_ODE_CLAMP_MAX
                        
                except Exception as e:
                    logger.warning(f"Euler step failed at i={i}: {e}")
                    progress_values[i] = progress_values[i-1]  # Keep previous values
                    research_stock_values[i] = research_stock_values[i-1]
            
            # Convert back to original time range
            final_times = np.linspace(min(time_range), max(time_range), cfg.DENSE_OUTPUT_POINTS)
            final_progress = np.interp(final_times, times, progress_values)
            final_research_stock = np.interp(final_times, times, research_stock_values)
            
            return final_times, final_progress, final_research_stock
            
        except Exception as e:
            logger.error(f"Even Euler fallback failed: {e}")
            raise RuntimeError(f"All integration methods failed. Last error: {e}")
    
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


def estimate_parameters(anchor_constraints: List[AnchorConstraint], time_series_data: TimeSeriesData, 
                       initial_params: Parameters, initial_progress: float = 0.0, fixed_params: Optional[List[str]] = None) -> Tuple[Parameters, List[Dict[str, Any]]]:
    """
    Find parameters that best satisfy anchor constraints.
    
    Args:
        anchor_constraints: List of constraints to satisfy
        time_series_data: Input time series
        initial_params: Initial parameter guess
        fixed_params: List of parameter names to keep fixed
    
    Returns:
        Optimized parameters
    """
    if fixed_params is None:
        fixed_params = []
    
    # Parameter bounds - tight bounds for numerical stability and physical constraints
    bounds = cfg.PARAMETER_BOUNDS
    
    def validate_parameter_combination(params_dict: Dict[str, float]) -> bool:
        """Validate that parameter combinations are physically meaningful and numerically stable"""
        try:
            # Check automation fraction is reasonable
            if 'automation_fraction_at_superhuman_coder' in params_dict:
                if params_dict['automation_fraction_at_superhuman_coder'] <= cfg.PARAM_VALIDATION_THRESHOLDS['automation_fraction_superhuman_coder_min'] or params_dict['automation_fraction_at_superhuman_coder'] >= cfg.PARAM_VALIDATION_THRESHOLDS['automation_fraction_superhuman_coder_max']:
                    logger.warning(f"automation_fraction_at_superhuman_coder should be in reasonable range ({cfg.PARAM_VALIDATION_THRESHOLDS['automation_fraction_superhuman_coder_min']}, {cfg.PARAM_VALIDATION_THRESHOLDS['automation_fraction_superhuman_coder_max']})")
                    return False
            
            # Check that progress at half automation is positive
            if 'progress_at_half_sc_automation' in params_dict:
                if params_dict['progress_at_half_sc_automation'] <= cfg.PARAM_VALIDATION_THRESHOLDS['progress_at_half_automation_min']:
                    logger.warning("progress_at_half_sc_automation must be positive")
                    return False
            
            # Check automation slope is reasonable
            if 'automation_slope' in params_dict:
                if params_dict['automation_slope'] <= cfg.PARAM_VALIDATION_THRESHOLDS['automation_slope_min'] or params_dict['automation_slope'] > cfg.PARAM_VALIDATION_THRESHOLDS['automation_slope_max']:
                    logger.warning(f"automation_slope should be in reasonable range ({cfg.PARAM_VALIDATION_THRESHOLDS['automation_slope_min']}, {cfg.PARAM_VALIDATION_THRESHOLDS['automation_slope_max']})")
                    return False
            
            # Check for extreme elasticity combinations that cause numerical instability
            if 'rho_cognitive' in params_dict and 'rho_progress' in params_dict:
                if abs(params_dict['rho_cognitive']) > cfg.PARAM_VALIDATION_THRESHOLDS['rho_extreme_abs'] and abs(params_dict['rho_progress']) > cfg.PARAM_VALIDATION_THRESHOLDS['rho_extreme_abs']:
                    if params_dict['rho_cognitive'] * params_dict['rho_progress'] > cfg.PARAM_VALIDATION_THRESHOLDS['rho_product_max']:
                        logger.warning("Extreme elasticity combination may cause numerical instability")
                        return False
            
            # Check normalization values are reasonable
            if 'cognitive_output_normalization' in params_dict:
                if params_dict['cognitive_output_normalization'] > cfg.PARAM_VALIDATION_THRESHOLDS['cognitive_output_normalization_max']:
                    logger.warning("Cognitive output normalization too large, may cause instability")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating parameter combination: {e}")
            return False
    
    # Use utility function to compute initial conditions
    initial_params, initial_conditions = setup_model(time_series_data, initial_params, initial_progress)
    
    # Pre-validate constraints for feasibility
    def check_constraint_feasibility(constraint: AnchorConstraint) -> bool:
        """Check if a constraint is mathematically feasible"""
        try:
            conditions = constraint.conditions
            
            # Check if constraint conditions are physically reasonable
            if 'automation_fraction' in conditions:
                if not (0 <= conditions['automation_fraction'] <= 1):
                    logger.warning(f"Constraint has invalid automation fraction: {conditions['automation_fraction']}")
                    return False
            
            if 'time' in conditions:
                time_range = [time_series_data.time.min(), time_series_data.time.max()]
                if not (time_range[0] <= conditions['time'] <= time_range[1]):
                    logger.warning(f"Constraint time {conditions['time']} outside data range {time_range}")
                    return False
            
            # Check if target values are reasonable
            if constraint.target_variable == 'progress_rate':
                if constraint.target_value <= 0 or constraint.target_value > cfg.FEASIBILITY_CHECK_THRESHOLDS['progress_rate_target_max']:
                    logger.warning(f"Unreasonable progress rate target: {constraint.target_value}")
                    return False
            elif constraint.target_variable == 'automation_fraction':
                if not (0 <= constraint.target_value <= 1):
                    logger.warning(f"Invalid automation fraction target: {constraint.target_value}")
                    return False
            
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking constraint feasibility: {e}")
            return False
    
    # Check all constraints for feasibility
    feasible_constraints = []
    for i, constraint in enumerate(anchor_constraints):
        if check_constraint_feasibility(constraint):
            feasible_constraints.append(constraint)
            logger.info(f"Constraint {i+1} is feasible")
        else:
            logger.warning(f"Constraint {i+1} appears infeasible and will be excluded from optimization")
    
    if not feasible_constraints:
        logger.error("No feasible constraints found for optimization")
        raise ValueError("All constraints appear to be infeasible")
    
    if len(feasible_constraints) < len(anchor_constraints):
        logger.warning(f"Using {len(feasible_constraints)} out of {len(anchor_constraints)} constraints")
    
    # Use only feasible constraints
    anchor_constraints = feasible_constraints
    
    # Get variable parameter names
    param_names = [name for name in bounds.keys() if name not in fixed_params]
    
    logger.info(f"Parameters being optimized: {param_names}")
    logger.info(f"Fixed parameters: {fixed_params}")
    logger.info(f"Initial cognitive_output_normalization: {initial_params.cognitive_output_normalization}")
    logger.info(f"Using {len(anchor_constraints)} feasible constraints")
    
    def objective(x):
        # Create parameters object
        params_dict = initial_params.__dict__.copy()
        for i, name in enumerate(param_names):
            params_dict[name] = x[i]
        
        # Validate parameter combination before proceeding
        if not validate_parameter_combination(params_dict):
            return cfg.OBJECTIVE_FUNCTION_CONFIG['high_penalty']  # High penalty for invalid combinations
        
        params = Parameters(**params_dict)
        
        try:
            # Set up model with proper normalization
            normalized_params, initial_conditions = setup_model(
                time_series_data, params, initial_progress
            )
            
            # Determine time range needed for all constraints
            constraint_times = []
            for constraint in anchor_constraints:
                t = constraint.conditions.get('time', 2025.0)
                constraint_times.append(t)
            
            start_time = time_series_data.time[0]
            end_time = max(max(constraint_times), time_series_data.time[-1])
            time_range = [start_time, end_time]
            
            # Use ProgressModel to compute trajectory and all metrics in one pass
            model = ProgressModel(normalized_params, time_series_data)
            model.compute_progress_trajectory(time_range, initial_progress)
            
            # Evaluate all constraints using the model's built-in methods
            total_error = model.evaluate_all_constraints(anchor_constraints)
            
            # Add regularization to prevent extreme parameter values
            regularization = 0.0
            for i, (name, value) in enumerate(zip(param_names, x)):
                if name in ['rho_cognitive', 'rho_progress']:
                    # Penalize extreme elasticity values more strongly
                    regularization += cfg.OBJECTIVE_FUNCTION_CONFIG['elasticity_regularization_weight'] * (value ** 4)  # Quartic penalty for elasticities
                elif name in ['alpha', 'software_scale']:
                    # Penalize values near boundaries (0 or 1)
                    distance_from_center = abs(value - 0.5)
                    if distance_from_center > cfg.OBJECTIVE_FUNCTION_CONFIG['boundary_avoidance_threshold']:  # Tighter boundary avoidance
                        regularization += cfg.OBJECTIVE_FUNCTION_CONFIG['boundary_avoidance_regularization_weight'] * ((distance_from_center - cfg.OBJECTIVE_FUNCTION_CONFIG['boundary_avoidance_threshold']) ** 2)
            
            # return total_error + regularization
            return total_error
            
        except Exception as e:
            logger.warning(f"Error in objective function: {e}")
            return cfg.OBJECTIVE_FUNCTION_CONFIG['high_penalty']
    
    # Initial values and bounds for optimization
    x0 = [getattr(initial_params, name) for name in param_names]
    opt_bounds = [bounds[name] for name in param_names]
    
    # Clamp initial values to bounds to avoid starting outside valid range
    for i, (name, bound) in enumerate(zip(param_names, opt_bounds)):
        min_bound, max_bound = bound
        if x0[i] < min_bound or x0[i] > max_bound:
            logger.warning(f"Initial value for {name} ({x0[i]}) outside bounds {bound}, clamping")
            x0[i] = np.clip(x0[i], min_bound, max_bound)
    
    # Generate diverse and strategic starting points for robust optimization
    best_result = None
    best_objective = float('inf')
    
    def generate_strategic_starting_points():
        """Generate diverse starting points using multiple strategies"""
        points = []
        
        # 1. Original UI parameters
        points.append(x0.copy())
        
        # 2. Parameter space extremes (conservative)
        for param_idx in range(len(param_names)):
            # Try min and max for each parameter individually
            for factor in [cfg.STRATEGIC_STARTING_POINTS_CONFIG['extreme_factor_min'], cfg.STRATEGIC_STARTING_POINTS_CONFIG['extreme_factor_max']]:  # Near min and max
                extreme_point = x0.copy()
                min_bound, max_bound = opt_bounds[param_idx]
                extreme_point[param_idx] = min_bound + factor * (max_bound - min_bound)
                points.append(extreme_point)
        
        # 3. Random Latin hypercube sampling for better space coverage
        for i in range(cfg.STRATEGIC_STARTING_POINTS_CONFIG['lhs_points']):
            lhs_point = []
            for j, (param_name, bound) in enumerate(zip(param_names, opt_bounds)):
                min_bound, max_bound = bound
                # Use Latin hypercube sampling
                segment = (i + np.random.random()) / cfg.STRATEGIC_STARTING_POINTS_CONFIG['lhs_points']  # Divide [0,1] into 5 segments
                value = min_bound + segment * (max_bound - min_bound)
                lhs_point.append(value)
            points.append(lhs_point)
        
        # 4. Constraint-informed starting points
        # Generate points that might satisfy constraints better
        for constraint in anchor_constraints:
            try:
                constraint_point = x0.copy()
                # Adjust parameters based on constraint type
                if constraint.target_variable == 'progress_rate':
                    if constraint.target_value > cfg.STRATEGIC_STARTING_POINTS_CONFIG['high_progress_rate_threshold']:  # High progress rate desired
                        # Favor less negative elasticities
                        for j, name in enumerate(param_names):
                            if name in ['rho_cognitive', 'rho_progress']:
                                min_bound, max_bound = opt_bounds[j]
                                constraint_point[j] = max_bound * cfg.STRATEGIC_STARTING_POINTS_CONFIG['rho_adjustment_factor']  # Near positive end
                elif constraint.target_variable == 'automation_fraction':
                    if constraint.target_value > cfg.STRATEGIC_STARTING_POINTS_CONFIG['high_automation_threshold']:  # High automation desired
                        # This requires higher progress values
                        for j, name in enumerate(param_names):
                            if 'progress_at' in name:
                                min_bound, max_bound = opt_bounds[j]
                                constraint_point[j] = max_bound * cfg.STRATEGIC_STARTING_POINTS_CONFIG['progress_at_half_automation_adjustment_factor']
                points.append(constraint_point)
            except Exception as e:
                logger.debug(f"Skipping constraint-informed point: {e}")
        
        # 5. Perturbed versions of the original point
        for i in range(cfg.STRATEGIC_STARTING_POINTS_CONFIG['perturbed_points']):
            perturbed = []
            for j, (param_name, bound) in enumerate(zip(param_names, opt_bounds)):
                val = x0[j]
                min_bound, max_bound = bound
                # Smaller perturbations for critical parameters
                if param_name in ['rho_cognitive', 'rho_progress']:
                    perturbation_factor = cfg.STRATEGIC_STARTING_POINTS_CONFIG['critical_param_perturbation_factor']  # 10% of range
                else:
                    perturbation_factor = cfg.STRATEGIC_STARTING_POINTS_CONFIG['other_param_perturbation_factor']  # 20% of range
                
                range_size = max_bound - min_bound
                perturbation = np.random.uniform(-perturbation_factor * range_size, 
                                               perturbation_factor * range_size)
                perturbed_val = np.clip(val + perturbation, min_bound, max_bound)
                perturbed.append(perturbed_val)
            points.append(perturbed)
        
        return points
    
    starting_points = generate_strategic_starting_points()
    logger.info(f"Generated {len(starting_points)} strategic starting points")
    for starting_point in starting_points:
        logger.info(f"Starting point: {starting_point}")
    # Calculate initial objective value
    initial_objective = objective(x0)
    
    # Try multiple optimization methods for robustness
    optimization_methods = [
        ('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-9}),  # Primary method
        ('TNC', {'maxiter': 500, 'ftol': 1e-8}),        # Alternative constrained method
        ('SLSQP', {'maxiter': 500, 'ftol': 1e-8}),      # Sequential Least Squares
    ]
    
    # Try optimization from each starting point with each method
    for method_name, method_options in optimization_methods:
        for i, start_point in enumerate(starting_points):
            try:
                logger.debug(f"Trying {method_name} from starting point {i+1}")
                
                # Validate starting point
                if not all(np.isfinite(start_point)):
                    logger.warning(f"Non-finite starting point {i+1}, skipping")
                    continue
                
                result = optimize.minimize(
                    objective, 
                    start_point, 
                    method=method_name, 
                    bounds=opt_bounds,
                    options=method_options
                )
                
                if result.success and result.fun < best_objective:
                    best_result = result
                    best_objective = result.fun
                    logger.info(f"Better solution found with {method_name} from point {i+1}: objective = {result.fun:.6f}")
                    
                # Early termination if we find a very good solution
                if result.fun < cfg.OPTIMIZATION_CONFIG['early_termination_fun_threshold_excellent']:
                    logger.info(f"Found excellent solution (objective < {cfg.OPTIMIZATION_CONFIG['early_termination_fun_threshold_excellent']}), stopping search")
                    break
                    
            except Exception as e:
                logger.debug(f"Optimization with {method_name} from point {i+1} failed: {e}")
                continue
        
        # If we found a good solution, no need to try other methods
        if best_objective < cfg.OPTIMIZATION_CONFIG['early_termination_fun_threshold_good']:
            logger.info(f"Good solution found with {method_name}, skipping remaining methods")
            break
    
    # Use best result or fallback to last attempted result
    if best_result is None:
        logger.warning("All optimization attempts failed, using last result")
        best_result = result
    else:
        logger.info(f"Best optimization result: objective = {best_objective:.6f}")
    
    result = best_result
    
    if not result.success:
        logger.warning(f"Optimization failed: {result.message}")
    
    # Create optimized parameters and ensure they're within bounds
    optimized_dict = initial_params.__dict__.copy()
    for i, name in enumerate(param_names):
        optimized_value = result.x[i]
        min_bound, max_bound = bounds[name]
        
        # Double-check bounds enforcement
        if optimized_value < min_bound or optimized_value > max_bound:
            logger.warning(f"Optimized value for {name} ({optimized_value}) outside bounds {bounds[name]}, clamping")
            optimized_value = np.clip(optimized_value, min_bound, max_bound)
        
        optimized_dict[name] = optimized_value
        logger.debug(f"Parameter {name}: {getattr(initial_params, name):.6f} -> {optimized_value:.6f}")
    
    optimized_params = Parameters(**optimized_dict)
    
    # Add objective values as attributes for the API to use
    optimized_params._initial_objective = initial_objective
    optimized_params._final_objective = result.fun
    
    # Evaluate final constraint satisfaction using ProgressModel
    constraint_evaluations = []
    
    try:
        # Determine time range needed for all constraints
        constraint_times = []
        for constraint in anchor_constraints:
            t = constraint.conditions.get('time', 2025.0)
            constraint_times.append(t)
        
        start_time = time_series_data.time[0]
        end_time = max(max(constraint_times), time_series_data.time[-1])
        time_range = [start_time, end_time]
        
        # Set up model with proper normalization
        normalized_params, initial_conditions = setup_model(
            time_series_data, optimized_params, initial_progress
        )
        
        # Create ProgressModel and compute trajectory
        model = ProgressModel(normalized_params, time_series_data)
        model.compute_progress_trajectory(time_range, initial_progress)
        
        # Evaluate each constraint using the model's method
        for i, constraint in enumerate(anchor_constraints):
            try:
                error = model.evaluate_anchor_constraint(constraint)
                # Convert relative error to satisfaction percentage
                # error = (model - target) / target, so satisfaction = 1 - |error|
                satisfaction = max(0, 1 - abs(error))
                constraint_evaluations.append({
                    'constraint_index': i,
                    'error': error,
                    'satisfaction': satisfaction,
                    'target_variable': constraint.target_variable,
                    'target_value': constraint.target_value
                })
            except Exception as e:
                logger.warning(f"Error evaluating final constraint {i+1}: {e}")
                constraint_evaluations.append({
                    'constraint_index': i,
                    'error': float('inf'),
                    'satisfaction': 0.0,
                    'target_variable': constraint.target_variable,
                    'target_value': constraint.target_value
                })
                
    except Exception as e:
        logger.warning(f"Error setting up ProgressModel for final constraint evaluation: {e}")
        # Fallback: create empty evaluations for all constraints
        for i, constraint in enumerate(anchor_constraints):
            constraint_evaluations.append({
                'constraint_index': i,
                'error': float('inf'),
                'satisfaction': 0.0,
                'target_variable': constraint.target_variable,
                'target_value': constraint.target_value
            })
    
    optimized_params._constraint_evaluations = constraint_evaluations
    
    return optimized_params, constraint_evaluations


class ProgressModel:
    """Main class for AI progress modeling"""
    
    def __init__(self, params: Parameters, time_series_data: TimeSeriesData):
        self.params = params
        self.data = time_series_data
        self.human_only_results = {}
        self.results = {}
        self.horizon_trajectory = None
        
        # Initialize flags for horizon trajectory anchor fix
        self._horizon_uses_shifted_form = False
        self._horizon_params = None
        
        # Initialize taste distribution for working with research taste
        self.taste_distribution = TasteDistribution()
    
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
                                 used to convert anchor_doubling_time from time units to progress units
            
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
            """Decaying doubling time function with numerical safeguards"""
            try:
                # Handle scalar vs array inputs
                is_scalar = np.isscalar(t)
                t_arr = np.atleast_1d(t)
                
                # Ensure parameters are within valid ranges
                if T_0 <= 0 or A_0 <= 0 or A_0 >= 1 or H_0 <= 0:
                    fallback = np.full_like(t_arr, np.log(1e12))
                    return fallback[0] if is_scalar else fallback
                
                # Calculate the base term (1 - A_0 * t / T_0)
                base_term = 1 - A_0 * t_arr / T_0
                
                # Check for negative or zero base terms
                if np.any(base_term <= 0):
                    fallback = np.full_like(t_arr, np.log(1e12))
                    return fallback[0] if is_scalar else fallback
                
                # Calculate the exponent
                log_denominator = np.log(1 - A_0)
                if log_denominator >= 0:  # This should be negative for valid A_0
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
                if self.params.anchor_horizon is not None:
                    # Use manual fitting with anchor point
                    # Get progress at anchor_time
                    anchor_progress = np.interp(self.params.anchor_time, human_only_times, human_only_progress)
                    
                    # If anchor_doubling_time is provided, use it to calculate slope
                    if self.params.anchor_doubling_time is not None:
                        # Convert anchor_doubling_time from time units to progress units
                        # doubling_time_in_progress_units = doubling_time_in_time_units * progress_rate
                        doubling_time_in_progress_units = self.params.anchor_doubling_time * anchor_progress_rate
                        # slope = log(2) / doubling_time (in progress units)
                        slope = np.log(2) / doubling_time_in_progress_units
                    else:
                        # Optimize slope using data, but fix intercept using anchor point
                        def fit_slope_only(slope_val):
                            intercept_val = np.log(self.params.anchor_horizon) - slope_val * anchor_progress
                            predicted = linear_func(progress_values, slope_val, intercept_val)
                            return np.sum((log_horizon_values - predicted)**2)
                        
                        result = optimize.minimize_scalar(fit_slope_only, bounds=(-10, 10), method='bounded')
                        slope = result.x
                    
                    # Calculate intercept from anchor point: log(anchor_horizon) = slope * anchor_progress + intercept
                    intercept = np.log(self.params.anchor_horizon) - slope * anchor_progress
                    
                    logger.info(f"Manual exponential horizon trajectory: log(horizon) = {slope:.6f} * progress + {intercept:.6f}")
                    logger.info(f"Using anchor point: time={self.params.anchor_time}, progress={anchor_progress:.4f}, horizon={self.params.anchor_horizon:.4f}")
                    if self.params.anchor_doubling_time is not None:
                        logger.info(f"Anchor doubling time (time units): {self.params.anchor_doubling_time:.4f}, converted to progress units: {doubling_time_in_progress_units:.4f} (using progress rate: {anchor_progress_rate:.4f})")
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
                
                # Calculate progress level where horizon reaches sc_time_horizon_minutes
                if self.params.sc_time_horizon_minutes > 0:
                    try:
                        # Solve: sc_time_horizon_minutes = exp(slope * progress + intercept)
                        # Therefore: progress = (log(sc_time_horizon_minutes) - intercept) / slope
                        calculated_progress_at_sc = (np.log(self.params.sc_time_horizon_minutes) - intercept) / slope
                        self.params.progress_at_sc = calculated_progress_at_sc
                        logger.info(f"Progress level at sc_time_horizon_minutes ({self.params.sc_time_horizon_minutes} min): {calculated_progress_at_sc:.4f}")
                    except (ValueError, ZeroDivisionError) as e:
                        logger.warning(f"Could not calculate progress at sc_time_horizon_minutes: {e}")
                        self.params.progress_at_sc = None
            
            elif self.params.horizon_extrapolation_type == "decaying doubling time":
                # Determine approach based on whether anchor_doubling_time is specified
                # If anchor_doubling_time is specified, we MUST use the shifted function approach
                # to ensure the doubling time at the anchor point equals anchor_doubling_time
                if self.params.anchor_doubling_time is not None or self.params.anchor_horizon is not None:
                    # Use shifted function approach
                    self._horizon_uses_shifted_form = True
                    # Get progress at anchor_time
                    anchor_progress = np.interp(self.params.anchor_time, human_only_times, human_only_progress)
                    
                    # Determine what parameters we have and what we need to optimize
                    params_to_optimize = []
                    fixed_params = {}
                    
                    # Handle H_0 (anchor_horizon)
                    if self.params.anchor_horizon is not None:
                        fixed_params['H_0'] = self.params.anchor_horizon
                    else:
                        params_to_optimize.append('H_0')
                    
                    # Handle T_0 (anchor_doubling_time)
                    if self.params.anchor_doubling_time is not None:
                        # Convert anchor_doubling_time from time units to progress units
                        # doubling_time_in_progress_units = doubling_time_in_time_units * progress_rate
                        doubling_time_in_progress_units = self.params.anchor_doubling_time * anchor_progress_rate
                        fixed_params['T_0'] = doubling_time_in_progress_units
                    else:
                        params_to_optimize.append('T_0')
                    
                    # Handle A_0 (doubling_decay_rate)
                    if self.params.doubling_decay_rate is not None:
                        fixed_params['A_0'] = self.params.doubling_decay_rate
                    else:
                        params_to_optimize.append('A_0')
                    
                    if len(params_to_optimize) == 0:
                        # All parameters specified (Case 8)
                        H_0 = self.params.anchor_horizon
                        T_0 = doubling_time_in_progress_units
                        A_0 = self.params.doubling_decay_rate
                        logger.info(f"Manual decaying doubling time: All parameters specified")
                    elif len(params_to_optimize) == 1:
                        # Optimize one parameter
                        param_name = params_to_optimize[0]
                        
                        def fit_single_param(param_val):
                            if param_name == 'A_0' and (param_val <= 0 or param_val >= 1):
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
                                    
                                    # Check for negative or zero base terms
                                    if np.any(base_term <= 0):
                                        return np.full_like(t_vals, np.log(1e12))
                                    
                                    exponent = np.log(2) / np.log(1 - A_0_val)
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
                            bounds = (1e-6, 0.999)
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
                            
                            if A_0_val <= 0 or A_0_val >= 1 or H_0_val <= 0 or T_0_val <= 0:
                                return 1e6
                            
                            # Use shifted function
                            def shifted_func(t_vals):
                                try:
                                    t_shifted = t_vals - anchor_progress
                                    base_term = 1 - A_0_val * t_shifted / T_0_val
                                    
                                    if np.any(base_term <= 0):
                                        return np.full_like(t_vals, np.log(1e12))
                                    
                                    exponent = np.log(2) / np.log(1 - A_0_val)
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
                                bounds.append((1e-6, 0.999))
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
                    logger.info(f"Using anchor point: time={self.params.anchor_time}, progress={anchor_progress:.4f}")
                    if self.params.anchor_horizon is not None:
                        logger.info(f"Anchor horizon: {self.params.anchor_horizon:.4f}")
                    if self.params.anchor_doubling_time is not None:
                        logger.info(f"Anchor doubling time (time units): {self.params.anchor_doubling_time:.4f}, converted to progress units: {doubling_time_in_progress_units:.4f} (using progress rate: {anchor_progress_rate:.4f})")
                    
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
                        bounds=([1e-6, 1e-6, 1e-6], [np.inf, 0.999, np.inf])  # Reasonable bounds
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
                        if T_0 <= 0 or A_0 <= 0 or A_0 >= 1 or H_0 <= 0:
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
                        
                        # Check for negative or zero base terms
                        if np.any(base_term <= 0):
                            fallback = np.full_like(progress_arr, 1e12)
                            return fallback[0] if is_scalar else fallback
                        
                        # Calculate the exponent
                        log_denominator = np.log(1 - A_0)
                        if log_denominator >= 0:  # This should be negative for valid A_0
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
                
                # Calculate progress level where horizon reaches sc_time_horizon_minutes
                if self.params.sc_time_horizon_minutes > 0:
                    try:
                        # Add numerical safeguards
                        if T_0 <= 0 or A_0 <= 0 or A_0 >= 1 or H_0 <= 0:
                            logger.warning("Invalid parameters for progress_at_sc calculation")
                            self.params.progress_at_sc = None
                        elif self.params.sc_time_horizon_minutes <= 0:
                            logger.warning("Invalid sc_time_horizon_minutes for calculation")
                            self.params.progress_at_sc = None
                        else:
                            # Check if the ratio is valid
                            ratio = self.params.sc_time_horizon_minutes / H_0
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
                                        
                                        if not np.isfinite(calculated_progress_at_sc):
                                            logger.warning("Invalid progress_at_sc result")
                                            self.params.progress_at_sc = None
                                        else:
                                            self.params.progress_at_sc = calculated_progress_at_sc
                                            logger.info(f"Progress level at sc_time_horizon_minutes ({self.params.sc_time_horizon_minutes} min): {calculated_progress_at_sc:.4f}")
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
        the actual integrated progress at anchor_time.
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
                if T_0 <= 0 or A_0 <= 0 or A_0 >= 1 or H_0 <= 0:
                    fallback = np.full_like(progress_arr, 1e12)
                    return fallback[0] if is_scalar else fallback
                
                # Use shifted form with updated anchor progress
                progress_shifted = progress_arr - new_anchor_progress
                base_term = 1 - A_0 * progress_shifted / T_0
                
                # Check for negative or zero base terms
                if np.any(base_term <= 0):
                    fallback = np.full_like(progress_arr, 1e12)
                    return fallback[0] if is_scalar else fallback
                
                # Calculate the exponent
                log_denominator = np.log(1 - A_0)
                if log_denominator >= 0:  # This should be negative for valid A_0
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
        initial_research_stock_rate_val = initial_conditions.research_stock_rate
        initial_research_stock_val = initial_conditions.research_stock
        logger.info(f"HUMAN-ONLY::: initial_research_stock_val: {initial_research_stock_val}, initial_research_stock_rate_val: {initial_research_stock_rate_val}")
        progress_rates = []
        research_stock_rates = []

        # human-only metrics
        for i, (t, p, rs) in enumerate(zip(times, progress_values, research_stock_values)):
            state = [p, rs]
            rates = progress_rate_at_time(t, state, self.data, human_only_params, initial_research_stock_rate_val, initial_research_stock_val)
            progress_rates.append(rates[0])
            research_stock_rates.append(rates[1])
        
        # Anchor stats at params.anchor_time
        anchor_time = human_only_params.anchor_time
        anchor_progress = np.interp(anchor_time, times, progress_values)
        anchor_progress_rate = np.interp(anchor_time, times, progress_rates)
        # Interpolate human and AI labor at anchor time using log-space when positive
        if np.all(self.data.L_HUMAN > 0):
            anchor_human_labor = _log_interp(anchor_time, self.data.time, self.data.L_HUMAN)
        else:
            anchor_human_labor = np.interp(anchor_time, self.data.time, self.data.L_HUMAN)
        if np.all(self.data.L_AI > 0):
            anchor_ai_labor = _log_interp(anchor_time, self.data.time, self.data.L_AI)
        else:
            anchor_ai_labor = np.interp(anchor_time, self.data.time, self.data.L_AI)
        
        self.human_only_results = {
            'times': times,
            'progress': progress_values,
            'research_stock': research_stock_values,
            'progress_rates': progress_rates,
            'research_stock_rates': research_stock_rates,
            'anchor_stats': {
                'progress': anchor_progress,
                'progress_rate': anchor_progress_rate,
                'human_labor': anchor_human_labor,
                'ai_labor': anchor_ai_labor
            },
            'input_time_series': {
                'time': self.data.time,
                'L_HUMAN': self.data.L_HUMAN,
                'L_AI': self.data.L_AI,
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

        # first compute human-only trajectory
        human_only_times, human_only_progress, _ = self.compute_human_only_trajectory(time_range, initial_progress)
        
        # estimate horizon trajectory from METR data using human-only trajectory
        try:
            anchor_progress_rate = self.human_only_results['anchor_stats']['progress_rate']
            self.estimate_horizon_trajectory(human_only_times, human_only_progress, anchor_progress_rate)
        except Exception as e:
            logger.warning(f"Failed to estimate horizon trajectory: {e}")
            self.horizon_trajectory = None

        # compute automation fraction at anchor time
        anchor_time = self.params.anchor_time
        anchor_progress = self.human_only_results['anchor_stats']['progress']
        anchor_human_labor = self.human_only_results['anchor_stats']['human_labor']
        anchor_ai_labor = self.human_only_results['anchor_stats']['ai_labor']
        # compute automation fraction at anchor time
        anchor_aut_frac = aut_frac_from_swe_multiplier(self.params.swe_multiplier_at_anchor_time, anchor_human_labor, anchor_ai_labor, self.params)
        # anchor_aut_frac = 0.01
        logger.info(f"calculated anchor automation fraction: {anchor_aut_frac} from swe_multiplier_at_anchor_time: {self.params.swe_multiplier_at_anchor_time} and anchor_time: {anchor_time}")
        automation_anchors = {
            anchor_progress: anchor_aut_frac,
            self.params.progress_at_sc: self.params.automation_fraction_at_superhuman_coder
        }
        logger.info(f"Automation anchors: {automation_anchors}")
        self.params.automation_anchors = automation_anchors
        times, progress_values, research_stock_values = integrate_progress(time_range, initial_progress, self.data, self.params)
        
        # Fix for Case 2 anchor horizon blowup: Update anchor_progress after ODE integration
        # This ensures that the horizon at anchor_time matches the specified anchor_horizon value
        if (self.horizon_trajectory is not None and 
            hasattr(self, '_horizon_uses_shifted_form') and self._horizon_uses_shifted_form and
            self.params.anchor_horizon is not None):
            
            # Recompute the actual progress at anchor_time from the integrated trajectory
            actual_anchor_progress = np.interp(self.params.anchor_time, times, progress_values)
            
            # Update the horizon trajectory function with the corrected anchor progress
            logger.info(f"Updating anchor progress from fitted value ({anchor_progress:.6f}) to actual integrated value: {actual_anchor_progress:.6f}")
            self._update_horizon_trajectory_anchor(actual_anchor_progress)
        
        # Use utility function to compute initial conditions with the correct parameters
        initial_conditions = compute_initial_conditions(self.data, self.params, initial_progress)
        initial_research_stock_rate_val = initial_conditions.research_stock_rate
        initial_research_stock_val = initial_conditions.research_stock
        logger.info(f"ACTUAL::: initial_research_stock_val: {initial_research_stock_val}, initial_research_stock_rate_val: {initial_research_stock_rate_val}")

        # Calculate all metrics in a single pass to avoid redundancy
        progress_rates = []
        research_stock_rates = []
        automation_fractions = []
        ai_research_tastes = []
        ai_research_taste_sds = []
        ai_research_taste_quantiles = []
        aggregate_research_tastes = []
        cognitive_outputs = []
        software_progress_rates = []
        software_efficiency = []  # Integral of software_progress_rate
        human_only_research_stock_rates = []
        human_only_software_progress_rates = []
        human_only_progress_rates = []
        ai_labor_contributions = []
        human_labor_contributions = []
        ai_cognitive_output_multipliers = []
        ai_research_stock_multipliers = []
        ai_software_progress_multipliers = []
        ai_overall_progress_multipliers = []
        discounted_exp_compute = []
        horizon_lengths = []
        effective_compute = []
        training_compute = []
        experiment_capacity = []
        
        # logger.info(f"Computing comprehensive metrics for {len(times)} time points")
        
        for i, (t, p, rs) in enumerate(zip(times, progress_values, research_stock_values)):
            try:
                state = [p, rs]
                rates = progress_rate_at_time(t, state, self.data, self.params, initial_research_stock_rate_val, initial_research_stock_val)
                progress_rates.append(rates[0])
                research_stock_rates.append(rates[1])
                
                # Compute automation fraction
                automation_fraction = compute_automation_fraction(p, self.params)
                automation_fractions.append(automation_fraction)
                
                # Compute AI research taste and aggregate research taste
                ai_research_taste = compute_ai_research_taste(p, self.params)
                ai_research_taste_sd = self.taste_distribution.get_sd_of_taste(ai_research_taste)
                ai_research_taste_quantile = self.taste_distribution.get_quantile_of_taste(ai_research_taste)
                aggregate_research_taste = compute_aggregate_research_taste(ai_research_taste)
                ai_research_tastes.append(ai_research_taste)
                ai_research_taste_sds.append(ai_research_taste_sd if np.isfinite(ai_research_taste_sd) else 0.0)
                ai_research_taste_quantiles.append(ai_research_taste_quantile if np.isfinite(ai_research_taste_quantile) else 0.0)
                aggregate_research_tastes.append(aggregate_research_taste)
                
                # Compute experiment capacity (research_stock_rate / aggregate_research_taste)
                current_research_stock_rate = research_stock_rates[i]
                exp_capacity = current_research_stock_rate / aggregate_research_taste if aggregate_research_taste > 0 else 0.0
                experiment_capacity.append(exp_capacity if np.isfinite(exp_capacity) else 0.0)
                
                # Interpolate input time series to current time
                L_HUMAN = _log_interp(t, self.data.time, self.data.L_HUMAN)
                L_AI = _log_interp(t, self.data.time, self.data.L_AI)
                experiment_compute = _log_interp(t, self.data.time, self.data.experiment_compute)
                training_compute_growth_rate = _log_interp(t, self.data.time, self.data.training_compute_growth_rate)
                
                # Compute discounted experiment compute
                discounted_exp_compute_val = experiment_compute ** self.params.zeta
                discounted_exp_compute.append(discounted_exp_compute_val if np.isfinite(discounted_exp_compute_val) else 0.0)
                
                # Compute cognitive output
                cognitive_output = compute_cognitive_output(
                    automation_fraction, L_AI, L_HUMAN, 
                    self.params.rho_cognitive, self.params.lambda_param, self.params.cognitive_output_normalization
                )
                cognitive_outputs.append(cognitive_output if np.isfinite(cognitive_output) else 0.0)
                
                # Compute software progress rate
                current_research_stock_rate = research_stock_rates[i]
                software_rate = compute_software_progress_rate(
                    rs, current_research_stock_rate, 
                    initial_research_stock_val, 
                    initial_research_stock_rate_val,
                    self.params.software_scale
                )
                software_progress_rates.append(software_rate if np.isfinite(software_rate) else 0.0)
                
                # Compute software efficiency (integral of software_progress_rate)
                if i == 0:
                    # Initialize software efficiency at 0
                    software_efficiency_val = 0.0
                else:
                    # Trapezoidal integration: add area of trapezoid from previous time step
                    dt = times[i] - times[i-1]
                    avg_rate = (software_progress_rates[i] + software_progress_rates[i-1]) / 2.0
                    software_efficiency_val = software_efficiency[i-1] + avg_rate * dt
                
                software_efficiency.append(software_efficiency_val if np.isfinite(software_efficiency_val) else 0.0)
                
                # Calculate human-only progress rate (with automation fraction = 0)
                human_only_cognitive_output = compute_cognitive_output(0, L_AI, L_HUMAN, self.params.rho_cognitive, self.params.lambda_param, self.params.cognitive_output_normalization, human_only=True)
                human_only_aggregate_research_taste = compute_aggregate_research_taste(0) # No AI research taste
                human_only_research_stock_rate = compute_research_stock_rate(
                    experiment_compute, human_only_cognitive_output, 
                    self.params.alpha, self.params.rho_progress, self.params.zeta, human_only_aggregate_research_taste
                )
                human_only_research_stock_rates.append(human_only_research_stock_rate if np.isfinite(human_only_research_stock_rate) else 0.0)
                human_only_software_rate = compute_software_progress_rate(
                    rs, human_only_research_stock_rate,
                    initial_research_stock_val,
                    initial_research_stock_rate_val,
                    self.params.software_scale
                )
                human_only_software_progress_rates.append(human_only_software_rate if np.isfinite(human_only_software_rate) else 0.0)
                human_only_overall_rate = compute_overall_progress_rate(
                    human_only_software_rate, training_compute_growth_rate
                )
                
                human_only_progress_rates.append(
                    human_only_overall_rate if np.isfinite(human_only_overall_rate) else 0.0
                )
                
                # Calculate labor contributions to cognitive output
                human_contrib = compute_cognitive_output(0, L_AI, L_HUMAN, self.params.rho_cognitive, self.params.lambda_param, self.params.cognitive_output_normalization, human_only=True)
                ai_contrib = max(0.0, cognitive_output - human_contrib)  # Ensure non-negative
                
                human_labor_contributions.append(human_contrib)
                ai_labor_contributions.append(ai_contrib)

                # Calculate automation multipliers on various quantities
                ai_cognitive_output_multipliers.append((cognitive_output / human_contrib)**(1.0/self.params.lambda_param) if ai_contrib > 0 else 0.0)
                ai_research_stock_multipliers.append(research_stock_rates[i] / human_only_research_stock_rate if human_only_research_stock_rate > 0 else 0.0)
                ai_software_progress_multipliers.append(software_rate / human_only_software_progress_rates[i] if human_only_software_progress_rates[i] > 0 else 0.0)
                ai_overall_progress_multipliers.append(progress_rates[i] / human_only_progress_rates[i] if human_only_progress_rates[i] > 0 else 0.0)

                # Compute horizon length using the fitted trajectory function
                horizon_length = 0.0  # Default fallback
                if self.horizon_trajectory is not None:
                    try:
                        horizon_length = self.horizon_trajectory(p)
                        if not np.isfinite(horizon_length) or horizon_length < 0:
                            horizon_length = 0.0
                    except Exception as horizon_e:
                        logger.warning(f"Error computing horizon at progress {p}: {horizon_e}")
                        horizon_length = 0.0
                
                horizon_lengths.append(horizon_length)
                
                # Compute effective compute as baseline_annual_compute_multiplier^progress
                effective_compute_val = 0.0  # Default fallback
                try:
                    effective_compute_val = self.params.baseline_annual_compute_multiplier ** p
                    if not np.isfinite(effective_compute_val) or effective_compute_val < 0:
                        effective_compute_val = 0.0
                except Exception as compute_e:
                    logger.warning(f"Error computing effective compute at progress {p}: {compute_e}")
                    effective_compute_val = 0.0
                
                effective_compute.append(effective_compute_val)
                
                # Compute training compute (integral of training_compute_growth_rate)
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

                
            except Exception as e:
                logger.warning(f"Error calculating metrics at t={t}: {e}")
                # Use safe fallback values
                if len(progress_rates) <= i:
                    progress_rates.append(0.0)
                if len(research_stock_rates) <= i:
                    research_stock_rates.append(0.0)
                automation_fractions.append(0.0)
                ai_research_tastes.append(0.0)
                ai_research_taste_sds.append(0.0)
                ai_research_taste_quantiles.append(0.0)
                aggregate_research_tastes.append(cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK)  # Default to no enhancement
                cognitive_outputs.append(0.0)
                software_progress_rates.append(0.0)
                software_efficiency.append(0.0)
                human_only_progress_rates.append(0.0)
                human_only_research_stock_rates.append(0.0)
                human_only_software_progress_rates.append(0.0)
                human_labor_contributions.append(0.0)
                ai_labor_contributions.append(0.0)
                ai_cognitive_output_multipliers.append(0.0)
                ai_research_stock_multipliers.append(0.0)
                ai_software_progress_multipliers.append(0.0)
                ai_overall_progress_multipliers.append(0.0)
                discounted_exp_compute.append(0.0)
                horizon_lengths.append(0.0)
                effective_compute.append(0.0)
                training_compute.append(0.0)
                experiment_capacity.append(0.0)
        
            
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
            self.sc_sw_multiplier = np.interp(sc_time, times, ai_software_progress_multipliers)
        else:
            sc_sw_multiplier = None
        
        # Calculate progress rate at anchor time
        anchor_time = self.params.anchor_time
        anchor_progress_rate = None
        if anchor_time is not None:
            # Check if anchor time is within our trajectory
            if times[0] <= anchor_time <= times[-1]:
                anchor_progress_rate = np.interp(anchor_time, times, progress_rates)
                logger.info(f"Progress rate at anchor time ({anchor_time:.3f}): {anchor_progress_rate:.6f}")
            else:
                logger.warning(f"Anchor time {anchor_time:.3f} is outside trajectory range [{times[0]:.3f}, {times[-1]:.3f}]")

        # Compute AI research taste slope in SD per anchor-progress-year (SD/year at anchor)
        ai_taste_slope_per_anchor_progress_year = None
        try:
            if anchor_progress_rate is not None and np.isfinite(anchor_progress_rate):
                ai_taste_slope_per_anchor_progress_year = float(self.params.ai_research_taste_slope) * float(anchor_progress_rate)
        except Exception as e:
            logger.warning(f"Failed computing taste slope per anchor-progress-year: {e}")
        
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
            'research_stock_rates': research_stock_rates,
            'cognitive_outputs': cognitive_outputs,
            'software_progress_rates': software_progress_rates,
            'software_efficiency': software_efficiency,
            'human_only_progress_rates': human_only_progress_rates,
            'ai_labor_contributions': ai_labor_contributions,
            'human_labor_contributions': human_labor_contributions,
            'ai_cognitive_output_multipliers': ai_cognitive_output_multipliers,
            'ai_research_stock_multipliers': ai_research_stock_multipliers,
            'ai_software_progress_multipliers': ai_software_progress_multipliers,
            'ai_overall_progress_multipliers': ai_overall_progress_multipliers,
            'discounted_exp_compute': discounted_exp_compute,
            'horizon_lengths': horizon_lengths,
            'effective_compute': effective_compute,
            'training_compute': training_compute,
            'experiment_capacity': experiment_capacity,
            'sc_time': sc_time,  # Time when superhuman coder level is reached
            'sc_progress_level': self.params.progress_at_sc,  # Progress level for SC
            'sc_sw_multiplier': self.sc_sw_multiplier if hasattr(self, 'sc_sw_multiplier') else None,  # Software progress multiplier at SC
            'anchor_time': anchor_time,  # Anchor time for manual horizon fitting
            'anchor_progress_rate': anchor_progress_rate,  # Progress rate at anchor time
            'ai_research_taste_slope_per_anchor_progress_year': ai_taste_slope_per_anchor_progress_year,  # SD per anchor-progress-year
            'input_time_series': {
                'time': self.data.time,
                'L_HUMAN': self.data.L_HUMAN,
                'L_AI': self.data.L_AI,
                'experiment_compute': self.data.experiment_compute,
                'training_compute_growth_rate': self.data.training_compute_growth_rate
            }
        }
        
        # logger.info(f"Computed trajectory from {time_range[0]} to {time_range[1]}")
        # logger.info(f"Progress: {progress_values[0]:.3f} -> {progress_values[-1]:.3f}")
        # logger.info(f"Research Stock: {research_stock_values[0]:.3f} -> {research_stock_values[-1]:.3f}")
        # logger.info(f"Automation: {automation_fractions[0]:.3f} -> {automation_fractions[-1]:.3f}")
        
        return times, progress_values, research_stock_values
    
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
                if condition_key in ['L_HUMAN', 'L_AI', 'experiment_compute']:
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
        elif target_key == 'cognitive_output':
            model_value = self.results['cognitive_outputs'][time_idx]
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
    L_AI = np.array([float(row['L_AI']) for row in data])
    experiment_compute = np.array([float(row['experiment_compute']) for row in data])
    training_compute_growth_rate = np.array([float(row['training_compute_growth_rate']) for row in data])
    
    return TimeSeriesData(time, L_HUMAN, L_AI, experiment_compute, training_compute_growth_rate)