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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesData:
    """Input time series data"""
    time: np.ndarray  # Decimal years
    L_HUMAN: np.ndarray  # Human labor supply
    L_AI: np.ndarray  # AI labor supply (human-equivalents)
    experiment_compute: np.ndarray  # Experiment compute budget
    training_compute: np.ndarray  # Training compute budget


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
    # Production function parameters
    rho_cognitive: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['rho_cognitive'])
    rho_progress: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['rho_progress'])
    alpha: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['alpha'])
    software_progress_share: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['software_progress_share'])
    zeta: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['zeta'])
    
    # Automation sigmoid parameters
    automation_fraction_at_superhuman_coder: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_fraction_at_superhuman_coder'])
    progress_at_half_sc_automation: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['progress_at_half_sc_automation'])
    automation_slope: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_slope'])
    
    # AI Research Taste sigmoid parameters
    ai_research_taste_at_superhuman_coder: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['ai_research_taste_at_superhuman_coder'])
    progress_at_half_ai_research_taste: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['progress_at_half_ai_research_taste'])
    ai_research_taste_slope: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['ai_research_taste_slope'])
    taste_schedule_type: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['taste_schedule_type'])
    progress_at_sc: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['progress_at_sc'])
    
    # Normalization
    progress_rate_normalization: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['progress_rate_normalization'])
    cognitive_output_normalization: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['cognitive_output_normalization'])
    
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
        
        if not np.isfinite(self.software_progress_share):
            logger.warning(f"Non-finite software_progress_share: {self.software_progress_share}, setting to 0.5")
            self.software_progress_share = 0.5
        else:
            self.software_progress_share = np.clip(self.software_progress_share, cfg.PARAM_CLIP_MIN, 1.0 - cfg.PARAM_CLIP_MIN)
        
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
        if not np.isfinite(self.ai_research_taste_at_superhuman_coder):
            logger.warning(f"Non-finite ai_research_taste_at_superhuman_coder: {self.ai_research_taste_at_superhuman_coder}, setting to {cfg.DEFAULT_PARAMETERS['ai_research_taste_at_superhuman_coder']}")
            self.ai_research_taste_at_superhuman_coder = cfg.DEFAULT_PARAMETERS['ai_research_taste_at_superhuman_coder']
        else:
            self.ai_research_taste_at_superhuman_coder = np.clip(self.ai_research_taste_at_superhuman_coder, cfg.PARAM_CLIP_MIN, 1.0 - cfg.PARAM_CLIP_MIN)
        
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

        # Sanitize normalization parameters
        if not np.isfinite(self.progress_rate_normalization) or self.progress_rate_normalization <= 0:
            logger.warning(f"Invalid progress_rate_normalization: {self.progress_rate_normalization}, setting to 1.0")
            self.progress_rate_normalization = 1.0
        else:
            self.progress_rate_normalization = max(cfg.NORMALIZATION_MIN, self.progress_rate_normalization)
        
        if not np.isfinite(self.cognitive_output_normalization) or self.cognitive_output_normalization <= 0:
            logger.warning(f"Invalid cognitive_output_normalization: {self.cognitive_output_normalization}, setting to 1.0")
            self.cognitive_output_normalization = 1.0
        else:
            self.cognitive_output_normalization = max(cfg.NORMALIZATION_MIN, self.cognitive_output_normalization)


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


def compute_cognitive_output(automation_fraction: float, L_AI: float, L_HUMAN: float, rho: float, cognitive_normalization: float = 1.0) -> float:
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
        cognitive_normalization: Normalization constant for cognitive output
    
    Returns:
        Cognitive output
    """
    # Input validation
    if not all(np.isfinite([automation_fraction, L_AI, L_HUMAN, rho, cognitive_normalization])):
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
            
    return result * cognitive_normalization


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
                                 initial_research_stock: float, initial_research_stock_rate: float) -> float:
    """
    Compute software progress rate using research stock formulation:
    S(t) = RS(0) * RS'(t) / (RS'(0) * RS(t))
    
    Args:
        research_stock: Current research stock RS(t)
        research_stock_rate: Current research stock rate RS'(t)
        initial_research_stock: Initial research stock RS(0)
        initial_research_stock_rate: Initial research stock rate RS'(0)
    
    Returns:
        Software progress rate
    """
    # Input validation
    if not all(np.isfinite([research_stock, research_stock_rate, initial_research_stock, initial_research_stock_rate])):
        logger.warning("Non-finite inputs to compute_software_progress_rate")
        return 0.0
    
    if research_stock <= 0 or initial_research_stock <= 0:
        logger.warning("Non-positive research stock values")
        return 0.0
    
    if initial_research_stock_rate <= 0:
        logger.warning("Non-positive initial research stock rate")
        return 0.0
    
    # Compute software progress rate using research stock ratio formula
    try:
        numerator = initial_research_stock * research_stock_rate
        denominator = initial_research_stock_rate * research_stock
        
        if denominator == 0:
            logger.warning("Zero denominator in software progress rate calculation")
            return 0.0
        
        software_progress_rate = numerator / denominator
        
        if not np.isfinite(software_progress_rate) or software_progress_rate < 0:
            logger.warning(f"Invalid software progress rate: {software_progress_rate}")
            return 0.0
        
        return software_progress_rate
        
    except (ZeroDivisionError, OverflowError) as e:
        logger.warning(f"Error computing software progress rate: {e}")
        return 0.0


def compute_overall_progress_rate(software_progress_rate: float, training_compute: float, software_share: float) -> float:
    """
    Weighted average of software and training progress
    
    Args:
        software_progress_rate: Rate from software development
        training_compute: Training compute budget
        software_share: Weight on software progress [0,1]
    
    Returns:
        Overall progress rate
    """
    return software_share * software_progress_rate + (1 - software_share) * training_compute


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
                                   initial_progress: float = 1.0) -> float:
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
        initial_automation = 0
        initial_ai_research_taste = compute_ai_research_taste(initial_progress, params)
        initial_aggregate_research_taste = compute_aggregate_research_taste(initial_ai_research_taste)
        
        L_HUMAN_0 = _log_interp(start_time, time_series_data.time, time_series_data.L_HUMAN)
        L_AI_0 = _log_interp(start_time, time_series_data.time, time_series_data.L_AI)
        experiment_compute_0 = _log_interp(start_time, time_series_data.time, time_series_data.experiment_compute)
        
        cognitive_output_0 = compute_cognitive_output(
            initial_automation, L_AI_0, L_HUMAN_0, 
            params.rho_cognitive, params.cognitive_output_normalization
        )
        
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
        cognitive_output_dt = compute_cognitive_output(
            initial_automation, L_AI_dt, L_HUMAN_dt,
            params.rho_cognitive, params.cognitive_output_normalization
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
    training_compute: float
    cognitive_output: float
    research_stock_rate: float
    research_stock: float

def compute_initial_conditions(time_series_data: TimeSeriesData, params: Parameters, 
                             initial_progress: float = 1.0) -> InitialConditions:
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
    
    # Basic initial conditions
    initial_automation = compute_automation_fraction(initial_progress, params)
    initial_ai_research_taste = compute_ai_research_taste(initial_progress, params)
    initial_aggregate_research_taste = compute_aggregate_research_taste(initial_ai_research_taste)
    
    L_HUMAN = np.interp(start_time, time_series_data.time, time_series_data.L_HUMAN)
    L_AI = np.interp(start_time, time_series_data.time, time_series_data.L_AI)
    experiment_compute = np.interp(start_time, time_series_data.time, time_series_data.experiment_compute)
    training_compute = np.interp(start_time, time_series_data.time, time_series_data.training_compute)
    
    # Derived conditions
    cognitive_output = compute_cognitive_output(
        initial_automation, L_AI, L_HUMAN, 
        params.rho_cognitive, params.cognitive_output_normalization
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
        training_compute=training_compute,
        cognitive_output=cognitive_output,
        research_stock_rate=research_stock_rate,
        research_stock=research_stock
    )

def compute_progress_rate_normalization(initial_conditions: InitialConditions, 
                                      params: Parameters) -> float:
    """
    Calculate progress rate normalization so that initial progress rate equals 1.0.
    This eliminates duplication between estimate_parameters and compute_model.
    
    Args:
        initial_conditions: Pre-computed initial conditions
        params: Model parameters (with temporary progress_rate_normalization=1)
        
    Returns:
        Calculated normalization factor
    """
    # Compute initial software progress rate
    initial_software_progress_rate = compute_software_progress_rate(
        initial_conditions.research_stock, 
        initial_conditions.research_stock_rate,
        initial_conditions.research_stock,  # Use same value for initial reference
        initial_conditions.research_stock_rate  # Use same value for initial reference
    )
    
    # Compute unnormalized overall progress rate
    unnormalized_rate = compute_overall_progress_rate(
        initial_software_progress_rate, 
        initial_conditions.training_compute, 
        params.software_progress_share
    )
    
    # Return normalization factor that makes the rate equal to 1.0
    if unnormalized_rate > 0:
        return 1.0 / unnormalized_rate
    else:
        logger.warning("Unnormalized progress rate is zero or negative, using default normalization")
        return 1.0

def setup_model_with_normalization(time_series_data: TimeSeriesData, params: Parameters,
                                 initial_progress: float = 1.0) -> Tuple[Parameters, InitialConditions]:
    """
    Set up model parameters with proper progress rate normalization.
    This replaces the duplicated setup logic in estimate_parameters and compute_model.
    
    Args:
        time_series_data: Input time series
        params: Model parameters (will be modified to set progress_rate_normalization)
        initial_progress: Initial cumulative progress
        
    Returns:
        Tuple of (updated_params, initial_conditions)
    """
    # Compute initial conditions
    initial_conditions = compute_initial_conditions(time_series_data, params, initial_progress)
    
    # Calculate and set progress rate normalization
    params.progress_rate_normalization = compute_progress_rate_normalization(initial_conditions, params)
    
    # logger.info(f"Calculated progress_rate_normalization: {params.progress_rate_normalization}")
    
    return params, initial_conditions


def compute_automation_fraction(cumulative_progress: float, params: Parameters) -> float:
    """
    Sigmoid function for automation fraction based on cumulative progress.
    Uses a standard sigmoid: f(x) = L / (1 + e^(-k*(x-x0)))
    
    This replaces the previous confusing approach that used 2025-based anchor points.
    The new parameters are more intuitive:
    - automation_fraction_at_superhuman_coder: Maximum automation level (L)
    - progress_at_half_sc_automation: Progress where automation = L/2 (x0) 
    - automation_slope: Controls transition steepness (k)
    
    Args:
        cumulative_progress: Current cumulative progress.
        params: Model parameters containing sigmoid parameters.
    
    Returns:
        Automation fraction in [0, 1].
    """
    # Extract sigmoid parameters
    L = params.automation_fraction_at_superhuman_coder  # Upper asymptote
    x0 = params.progress_at_half_sc_automation  # Midpoint (where automation = L/2)
    k = params.automation_slope  # Slope parameter
    
    # Input validation
    if not np.isfinite(cumulative_progress):
        logger.warning(f"Non-finite cumulative_progress: {cumulative_progress}")
        cumulative_progress = 0.0
    
    # Calculate sigmoid: f(x) = L / (1 + e^(-k*(x-x0)))
    try:
        exponent = -k * (cumulative_progress - x0)
        
        # Handle extreme exponents to prevent overflow/underflow
        if exponent > cfg.SIGMOID_EXPONENT_CLAMP:  # e^100 is very large, sigmoid ≈ 0
            automation_fraction = 0.0
        elif exponent < -cfg.SIGMOID_EXPONENT_CLAMP:  # e^(-100) is very small, sigmoid ≈ L
            automation_fraction = L
        else:
            automation_fraction = L / (1 + np.exp(exponent))
            
    except (OverflowError, ValueError) as e:
        logger.warning(f"Numerical error in sigmoid calculation: {e}")
        # Fallback: linear interpolation between 0 and L
        if cumulative_progress <= x0:
            automation_fraction = L * 0.5 * (cumulative_progress / x0)
        else:
            automation_fraction = L * (0.5 + 0.5 * min(1.0, (cumulative_progress - x0) / x0))
    
    # Clamp to [0, 1] as a final safeguard
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
            ai_research_taste = L * (0.5 + 0.5 * min(1.0, (cumulative_progress - x0) / x0))
    
    # Clamp to [0, 1] as a final safeguard
    return np.clip(ai_research_taste, 0.0, 1.0)


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
    
    try:
        # Calculate exponential: taste(x) = taste_at_sc * exp(slope * (x - progress_at_sc))
        exponent = slope * (cumulative_progress - progress_at_sc)
        
        # Handle extreme exponents to prevent overflow/underflow
        if exponent > cfg.SIGMOID_EXPONENT_CLAMP:
            ai_research_taste = min(1.0, taste_at_sc * np.exp(cfg.SIGMOID_EXPONENT_CLAMP))
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
            ai_research_taste = min(1.0, taste_at_sc * 2.0)
    
    # Clamp to [0, 1] as a final safeguard
    return np.clip(ai_research_taste, 0.0, 1.0)


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
    
    try:
        # Compute offset so curve passes through (progress_at_sc, taste_at_sc)
        # We want: taste_at_sc = taste_distribution.get_taste_at_sd(slope * progress_at_sc + offset)
        # So: offset = taste_distribution.get_sd_of_taste(taste_at_sc) - slope * progress_at_sc
        
        # Clamp taste_at_sc to valid range to avoid log(0) issues
        taste_at_sc_clamped = max(1e-10, min(taste_at_sc, 1.0))
        
        target_sd = taste_distribution.get_sd_of_taste(taste_at_sc_clamped)
        offset = target_sd - slope * progress_at_sc
        
        # Compute AI research taste at current progress
        current_sd = slope * cumulative_progress + offset
        ai_research_taste = taste_distribution.get_taste_at_sd(current_sd)
        
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
            ai_research_taste = min(1.0, taste_at_sc * (1 + slope * 0.1 * delta))
    
    # Clamp to [0, 1] as a final safeguard
    return np.clip(ai_research_taste, 0.0, 1.0)


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
    
    if not np.isfinite(cumulative_progress) or cumulative_progress < 0:
        logger.warning(f"Invalid cumulative progress: {cumulative_progress}")
        cumulative_progress = max(0.0, cfg.PARAM_CLIP_MIN)  # Use small positive value
    
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
        
        training_compute = np.interp(t, time_series_data.time, time_series_data.training_compute)
        
        # Validate interpolated values
        if not all(np.isfinite([L_HUMAN, L_AI, experiment_compute, training_compute])):
            logger.warning(f"Non-finite interpolated values at t={t}")
            return [0.0, 0.0]
        
        # Ensure non-negative values
        L_HUMAN = max(0.0, L_HUMAN)
        L_AI = max(0.0, L_AI)
        experiment_compute = max(0.0, experiment_compute)
        training_compute = max(0.0, training_compute)
        
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
            automation_fraction, L_AI, L_HUMAN, params.rho_cognitive, params.cognitive_output_normalization
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
                initial_research_stock, initial_research_stock_rate
            )
        
        if not np.isfinite(software_progress_rate) or software_progress_rate < 0:
            logger.warning(f"Invalid software progress rate: {software_progress_rate}")
            return [0.0, 0.0]
        
        # Compute overall progress rate (dP/dt)
        overall_rate = compute_overall_progress_rate(
            software_progress_rate, training_compute, params.software_progress_share
        )
        
        if not np.isfinite(overall_rate) or overall_rate < 0:
            logger.warning(f"Invalid overall progress rate: {overall_rate}")
            return [0.0, 0.0]
        
        # Apply normalization with validation
        if not np.isfinite(params.progress_rate_normalization) or params.progress_rate_normalization <= 0:
            logger.warning(f"Invalid progress rate normalization: {params.progress_rate_normalization}")
            return [0.0, 0.0]
        
        normalized_progress_rate = overall_rate * params.progress_rate_normalization
        
        # Final validation
        if not np.isfinite(normalized_progress_rate) or normalized_progress_rate < 0:
            logger.warning(f"Invalid final normalized progress rate: {normalized_progress_rate}")
            return [0.0, 0.0]
        
        # Cap extremely large rates to prevent numerical issues
        if normalized_progress_rate > cfg.MAX_NORMALIZED_PROGRESS_RATE:
            logger.warning(f"Very large progress rate {normalized_progress_rate}, capping to {cfg.MAX_NORMALIZED_PROGRESS_RATE}")
            normalized_progress_rate = cfg.MAX_NORMALIZED_PROGRESS_RATE
        
        logger.debug(f"t={t:.2f}, progress={cumulative_progress:.3f}, research_stock={research_stock:.3f}, "
                    f"automation={automation_fraction:.3f}, dP/dt={normalized_progress_rate:.3f}, dRS/dt={research_stock_rate:.3f}")
        
        return [normalized_progress_rate, research_stock_rate]
        
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
            # Prevent progress from going negative or becoming extremely large
            if y[0] < 0:
                y[0] = 1e-6
            elif y[0] > cfg.PROGRESS_ODE_CLAMP_MAX:
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
    if not np.isfinite(initial_progress) or initial_progress <= 0:
        logger.warning(f"Invalid initial progress {initial_progress}, using fallback")
        initial_progress = 1.0
    
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
        logger.warning("All scipy integration methods failed, using simple Euler fallback")
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
                    
                    # Ensure values don't go negative or become too large
                    if progress_values[i] < 0:
                        progress_values[i] = progress_values[i-1]
                    elif progress_values[i] > cfg.PROGRESS_ODE_CLAMP_MAX:
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
        
        # Check for negative values
        if np.any(progress_values < 0):
            logger.warning("Negative progress values detected, clamping to zero")
            progress_values = np.maximum(progress_values, 0.0)
        
        if np.any(research_stock_values <= 0):
            logger.warning("Non-positive research stock values detected, clamping to minimum")
            research_stock_values = np.maximum(research_stock_values, 1e-6)
        
        return times, progress_values, research_stock_values
        
    except Exception as e:
        logger.error(f"Error creating dense output: {e}")
        raise RuntimeError(f"Integration succeeded but dense output failed: {e}")


def estimate_parameters(anchor_constraints: List[AnchorConstraint], time_series_data: TimeSeriesData, 
                       initial_params: Parameters, initial_progress: float = 1.0, fixed_params: Optional[List[str]] = None) -> Tuple[Parameters, List[Dict[str, Any]]]:
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
    
    # Always fix progress_rate_normalization - it should not be optimized
    if 'progress_rate_normalization' not in fixed_params:
        fixed_params = fixed_params.copy()
        fixed_params.append('progress_rate_normalization')
    
    # Use utility function to compute initial conditions and set progress rate normalization
    initial_params, initial_conditions = setup_model_with_normalization(time_series_data, initial_params, initial_progress)
    
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
            normalized_params, initial_conditions = setup_model_with_normalization(
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
                elif name in ['alpha', 'software_progress_share']:
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
        normalized_params, initial_conditions = setup_model_with_normalization(
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
        self.results = {}
        
        # Initialize taste distribution for working with research taste
        self.taste_distribution = TasteDistribution()
        
    def compute_progress_trajectory(self, time_range: List[float], initial_progress: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute progress over specified time range with comprehensive metrics
        
        Args:
            time_range: [start_time, end_time]
            initial_progress: Initial progress (defaults to 1.0)
        
        Returns:
            Tuple of (times, cumulative_progress_values, research_stock_values)
        """
        if initial_progress is None:
            initial_progress = 1.0  # Use a reasonable default value
        
        times, progress_values, research_stock_values = integrate_progress(time_range, initial_progress, self.data, self.params)
        
        # Use utility function to compute initial conditions
        initial_conditions = compute_initial_conditions(self.data, self.params, initial_progress)
        initial_research_stock_rate_val = initial_conditions.research_stock_rate
        initial_research_stock_val = initial_conditions.research_stock
        
        # Calculate all metrics in a single pass to avoid redundancy
        progress_rates = []
        research_stock_rates = []
        automation_fractions = []
        ai_research_tastes = []
        ai_research_taste_sds = []
        aggregate_research_tastes = []
        cognitive_outputs = []
        software_progress_rates = []
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
                aggregate_research_taste = compute_aggregate_research_taste(ai_research_taste)
                ai_research_tastes.append(ai_research_taste)
                ai_research_taste_sds.append(ai_research_taste_sd if np.isfinite(ai_research_taste_sd) else 0.0)
                aggregate_research_tastes.append(aggregate_research_taste)
                
                # Interpolate input time series to current time
                L_HUMAN = _log_interp(t, self.data.time, self.data.L_HUMAN)
                L_AI = _log_interp(t, self.data.time, self.data.L_AI)
                experiment_compute = _log_interp(t, self.data.time, self.data.experiment_compute)
                training_compute = _log_interp(t, self.data.time, self.data.training_compute)
                
                # Compute discounted experiment compute
                discounted_exp_compute_val = experiment_compute ** self.params.zeta
                discounted_exp_compute.append(discounted_exp_compute_val if np.isfinite(discounted_exp_compute_val) else 0.0)
                
                # Compute cognitive output
                cognitive_output = compute_cognitive_output(
                    automation_fraction, L_AI, L_HUMAN, 
                    self.params.rho_cognitive, self.params.cognitive_output_normalization
                )
                cognitive_outputs.append(cognitive_output if np.isfinite(cognitive_output) else 0.0)
                
                # Compute software progress rate
                current_research_stock_rate = research_stock_rates[i]
                software_rate = compute_software_progress_rate(
                    rs, current_research_stock_rate, 
                    initial_research_stock_val, 
                    initial_research_stock_rate_val
                )
                software_progress_rates.append(software_rate if np.isfinite(software_rate) else 0.0)
                
                # Calculate human-only progress rate (with automation fraction = 0)
                human_only_cognitive_output = L_HUMAN * self.params.cognitive_output_normalization
                human_only_aggregate_research_taste = compute_aggregate_research_taste(0) # No AI research taste
                human_only_research_stock_rate = compute_research_stock_rate(
                    experiment_compute, human_only_cognitive_output, 
                    self.params.alpha, self.params.rho_progress, self.params.zeta, human_only_aggregate_research_taste
                )
                human_only_research_stock_rates.append(human_only_research_stock_rate if np.isfinite(human_only_research_stock_rate) else 0.0)
                human_only_software_rate = compute_software_progress_rate(
                    rs, human_only_research_stock_rate,
                    initial_research_stock_val,
                    initial_research_stock_rate_val
                )
                human_only_software_progress_rates.append(human_only_software_rate if np.isfinite(human_only_software_rate) else 0.0)
                human_only_overall_rate = compute_overall_progress_rate(
                    human_only_software_rate, training_compute, self.params.software_progress_share
                ) * self.params.progress_rate_normalization
                
                human_only_progress_rates.append(
                    human_only_overall_rate if np.isfinite(human_only_overall_rate) else 0.0
                )
                
                # Calculate labor contributions to cognitive output
                human_contrib = L_HUMAN * self.params.cognitive_output_normalization
                ai_contrib = max(0.0, cognitive_output - human_contrib)  # Ensure non-negative
                
                human_labor_contributions.append(human_contrib)
                ai_labor_contributions.append(ai_contrib)

                # Calculate automation multipliers on various quantities
                ai_cognitive_output_multipliers.append(cognitive_output / human_contrib if ai_contrib > 0 else 0.0)
                ai_research_stock_multipliers.append(research_stock_rates[i] / human_only_research_stock_rate if human_only_research_stock_rate > 0 else 0.0)
                ai_software_progress_multipliers.append(software_rate / human_only_software_progress_rates[i] if human_only_software_progress_rates[i] > 0 else 0.0)
                ai_overall_progress_multipliers.append(progress_rates[i] / human_only_progress_rates[i] if human_only_progress_rates[i] > 0 else 0.0)

                
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
                aggregate_research_tastes.append(cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK)  # Default to no enhancement
                cognitive_outputs.append(0.0)
                software_progress_rates.append(0.0)
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
        
            
        # Store comprehensive results
        self.results = {
            'times': times,
            'progress': progress_values,
            'research_stock': research_stock_values,
            'automation_fraction': automation_fractions,
            'ai_research_taste': ai_research_tastes,
            'ai_research_taste_sd': ai_research_taste_sds,
            'aggregate_research_taste': aggregate_research_tastes,
            'progress_rates': progress_rates,
            'research_stock_rates': research_stock_rates,
            'cognitive_outputs': cognitive_outputs,
            'software_progress_rates': software_progress_rates,
            'human_only_progress_rates': human_only_progress_rates,
            'ai_labor_contributions': ai_labor_contributions,
            'human_labor_contributions': human_labor_contributions,
            'ai_cognitive_output_multipliers': ai_cognitive_output_multipliers,
            'ai_research_stock_multipliers': ai_research_stock_multipliers,
            'ai_software_progress_multipliers': ai_software_progress_multipliers,
            'ai_overall_progress_multipliers': ai_overall_progress_multipliers,
            'discounted_exp_compute': discounted_exp_compute,
            'input_time_series': {
                'time': self.data.time,
                'L_HUMAN': self.data.L_HUMAN,
                'L_AI': self.data.L_AI,
                'experiment_compute': self.data.experiment_compute,
                'training_compute': self.data.training_compute
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
    
    def plot_results(self, comprehensive: bool = False, save_path: Optional[str] = None):
        """Visualize progress, automation fraction, and component contributions"""
        try:
            from visualization import create_default_visualizer, quick_plot_results
            
            if 'times' not in self.results:
                raise ValueError("No results to plot. Run compute_progress_trajectory first.")
            
            if comprehensive:
                # Use comprehensive dashboard
                time_series_dict = {
                    'time': self.data.time,
                    'L_AI': self.data.L_AI,
                    'L_HUMAN': self.data.L_HUMAN,
                    'experiment_compute': self.data.experiment_compute,
                    'training_compute': self.data.training_compute
                }
                fig = quick_plot_results(self.results, time_series_dict, save_path)
            else:
                # Simple two-panel plot
                visualizer = create_default_visualizer()
                import matplotlib.pyplot as plt
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                visualizer.plot_progress_trajectory(
                    self.results['times'], self.results['progress'], 
                    'AI Progress Over Time', ax1
                )
                
                visualizer.plot_automation_fraction(
                    self.results['times'], self.results['automation_fraction'],
                    'Automation Fraction Over Time', ax2
                )
                
                plt.tight_layout()
                
                if save_path:
                    visualizer.save_plot(fig, save_path)
            
            return fig
            
        except ImportError:
            logger.warning("matplotlib or visualization module not available for plotting")
    
    def export_results(self, filename: str):
        """Export trajectories and parameters"""
        if 'times' not in self.results:
            raise ValueError("No results to export. Run compute_progress_trajectory first.")
        
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['time', 'cumulative_progress', 'automation_fraction', 'research_stock'])
            
            # Write data
            for i in range(len(self.results['times'])):
                writer.writerow([
                    self.results['times'][i],
                    self.results['progress'][i],
                    self.results['automation_fraction'][i],
                    self.results['research_stock'][i]
                ])
        
        logger.info(f"Results exported to {filename}")
    
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
    training_compute = np.array([float(row['training_compute']) for row in data])
    
    return TimeSeriesData(time, L_HUMAN, L_AI, experiment_compute, training_compute)



if __name__ == "__main__":
    # Example usage
    logger.info("AI Progress Modeling Script")
    
    # Create example time series data
    time = np.linspace(2019, 2030, 12)
    L_HUMAN = np.ones_like(time) * 1e6
    L_AI = np.logspace(3, 8, len(time))  # Exponential growth
    experiment_compute = np.logspace(6, 10, len(time))
    training_compute = np.logspace(6, 10, len(time))
    
    data = TimeSeriesData(time, L_HUMAN, L_AI, experiment_compute, training_compute)
    
    # Define anchor constraints
    anchors = [
        AnchorConstraint(
            conditions={"automation_fraction": 0.9, "L_AI": 1e9},
            target_variable="progress_rate",
            target_value=5.0,
            weight=1.0
        ),
        AnchorConstraint(
            conditions={"time": 2025, "cumulative_progress": 10.0},
            target_variable="automation_fraction",
            target_value=0.5,
            weight=1.0
        )
    ]
    
    # Initial parameter guess
    initial_params = Parameters()
    
    # Estimate parameters
    logger.info("Estimating parameters...")
    params, _ = estimate_parameters(anchors, data, initial_params)
    logger.info(f"Optimized parameters: {params}")
    
    # Run model
    model = ProgressModel(params, data)
    times, progress, research_stock = model.compute_progress_trajectory([2019, 2030], initial_progress=1.0)
    
    # Export results
    model.export_results("progress_results.csv")
    
    logger.info("Modeling complete")