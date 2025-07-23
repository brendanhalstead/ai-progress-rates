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
from typing import List, Tuple, Optional, Dict, Any
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


@dataclass
class Parameters:
    """Model parameters with validation"""
    # Production function parameters
    rho_cognitive: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['rho_cognitive'])
    rho_progress: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['rho_progress'])
    alpha: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['alpha'])
    software_progress_share: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['software_progress_share'])
    
    # Automation sigmoid parameters
    automation_fraction_at_superhuman_coder: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_fraction_at_superhuman_coder'])
    progress_at_half_sc_automation: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['progress_at_half_sc_automation'])
    automation_slope: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_slope'])
    
    # Research stock parameters
    research_stock_at_simulation_start: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['research_stock_at_simulation_start'])
    
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
        
        # Sanitize automation parameters
        if not np.isfinite(self.automation_fraction_at_superhuman_coder):
            logger.warning(f"Non-finite automation_fraction_at_superhuman_coder: {self.automation_fraction_at_superhuman_coder}, setting to 0.9")
            self.automation_fraction_at_superhuman_coder = 0.9
        else:
            self.automation_fraction_at_superhuman_coder = np.clip(self.automation_fraction_at_superhuman_coder, cfg.PARAM_CLIP_MIN, 1.0 - cfg.PARAM_CLIP_MIN)
        
        if not np.isfinite(self.progress_at_half_sc_automation) or self.progress_at_half_sc_automation <= 0:
            logger.warning(f"Invalid progress_at_half_sc_automation: {self.progress_at_half_sc_automation}, setting to 10.0")
            self.progress_at_half_sc_automation = 10.0
        else:
            self.progress_at_half_sc_automation = max(cfg.PARAM_CLIP_MIN, self.progress_at_half_sc_automation)
        
        if not np.isfinite(self.automation_slope):
            logger.warning(f"Non-finite automation_slope: {self.automation_slope}, setting to 1.0")
            self.automation_slope = 1.0
        else:
            # Clamp slope to reasonable range to prevent numerical instability
            self.automation_slope = np.clip(self.automation_slope, cfg.AUTOMATION_SLOPE_CLIP_MIN, cfg.AUTOMATION_SLOPE_CLIP_MAX)
        
        # Sanitize research stock parameters
        if not np.isfinite(self.research_stock_at_simulation_start) or self.research_stock_at_simulation_start <= 0:
            logger.warning(f"Invalid research_stock_at_simulation_start: {self.research_stock_at_simulation_start}, setting to 1.0")
            self.research_stock_at_simulation_start = 1.0
        else:
            self.research_stock_at_simulation_start = max(cfg.RESEARCH_STOCK_START_MIN, self.research_stock_at_simulation_start)
        
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


def compute_research_stock_rate(experiment_compute: float, cognitive_output: float, alpha: float, rho: float) -> float:
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
    
    Returns:
        Research stock growth rate RS'(t)
    """
    # Input validation
    if not all(np.isfinite([experiment_compute, cognitive_output, alpha, rho])):
        logger.warning("Non-finite inputs to compute_research_stock_rate")
        return 0.0
    
    if experiment_compute < 0 or cognitive_output < 0:
        logger.warning("Negative inputs to compute_research_stock_rate")
        return 0.0
    
    # Clamp alpha to valid range
    alpha = np.clip(alpha, 0.0, 1.0)
    
    # Use the generic CES function for computation
    rate = _ces_function(experiment_compute, cognitive_output, alpha, rho)
    
    # Cap extremely large rates to prevent numerical issues
    if rate > cfg.MAX_RESEARCH_STOCK_RATE:
        logger.warning(f"Very large research stock rate {rate}, capping to {cfg.MAX_RESEARCH_STOCK_RATE}")
        rate = cfg.MAX_RESEARCH_STOCK_RATE
        
    return rate


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


def progress_rate_at_time(t: float, state: List[float], time_series_data: TimeSeriesData, params: Parameters, 
                         initial_research_stock_rate: Optional[float] = None) -> List[float]:
    """
    Compute instantaneous rates for both progress and research stock.
    This is the RHS of the coupled differential equation system.
    
    Args:
        t: Current time
        state: [cumulative_progress, research_stock]
        time_series_data: Input time series
        params: Model parameters
        initial_research_stock_rate: RS'(0) needed for software progress calculation
    
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
        research_stock = max(cfg.PARAM_CLIP_MIN, params.research_stock_at_simulation_start)
    
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
        
        # Compute cognitive output with validation
        cognitive_output = compute_cognitive_output(
            automation_fraction, L_AI, L_HUMAN, params.rho_cognitive, params.cognitive_output_normalization
        )
        
        if not np.isfinite(cognitive_output) or cognitive_output < 0:
            logger.warning(f"Invalid cognitive output: {cognitive_output}")
            return [0.0, 0.0]
        
        # Compute research stock rate (dRS/dt) with validation
        research_stock_rate = compute_research_stock_rate(
            experiment_compute, cognitive_output, params.alpha, params.rho_progress
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
                params.research_stock_at_simulation_start, initial_research_stock_rate
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
    # Calculate initial research stock rate for consistent computation
    start_time = time_series_data.time[0]
    initial_automation = compute_automation_fraction(initial_progress, params)
    initial_L_HUMAN = np.interp(start_time, time_series_data.time, time_series_data.L_HUMAN)
    initial_L_AI = np.interp(start_time, time_series_data.time, time_series_data.L_AI)
    initial_experiment_compute = np.interp(start_time, time_series_data.time, time_series_data.experiment_compute)
    
    initial_cognitive_output = compute_cognitive_output(
        initial_automation, initial_L_AI, initial_L_HUMAN, 
        params.rho_cognitive, params.cognitive_output_normalization
    )
    initial_research_stock_rate = compute_research_stock_rate(
        initial_experiment_compute, initial_cognitive_output, 
        params.alpha, params.rho_progress
    )
    
    if initial_research_stock_rate <= 0:
        logger.warning(f"Invalid initial research stock rate: {initial_research_stock_rate}, using fallback")
        initial_research_stock_rate = 1.0
    
    def ode_func(t, y):
        try:
            rates = progress_rate_at_time(t, y, time_series_data, params, initial_research_stock_rate)
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
                y[1] = max(1e-6, params.research_stock_at_simulation_start)
            elif y[1] > cfg.RESEARCH_STOCK_ODE_CLAMP_MAX:
                logger.warning(f"Research stock {y[1]} too large at time {t}, clamping")
                y[1] = cfg.RESEARCH_STOCK_ODE_CLAMP_MAX
            
            rates = progress_rate_at_time(t, y, time_series_data, params, initial_research_stock_rate)
            
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
    
    initial_research_stock = params.research_stock_at_simulation_start
    initial_state = [initial_progress, initial_research_stock]
    
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
            
            progress_values = np.zeros(n_steps)
            research_stock_values = np.zeros(n_steps)
            progress_values[0] = initial_progress
            research_stock_values[0] = initial_research_stock
            
            for i in range(1, n_steps):
                try:
                    state = [progress_values[i-1], research_stock_values[i-1]]
                    rates = progress_rate_at_time(times[i-1], state, time_series_data, params, initial_research_stock_rate)
                    
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


def evaluate_anchor_constraint(constraint: AnchorConstraint, time_series_data: TimeSeriesData, params: Parameters, initial_progress: float) -> float:
    """
    Evaluate model at specified conditions and compare to target.
    Returns the error (difference from target).
    
    Args:
        constraint: Constraint specification
        time_series_data: Input time series
        params: Model parameters
    
    Returns:
        Error (model_value - target_value)
    """
    # Extract conditions
    conditions = constraint.conditions.copy()
    
    # Default values if not specified
    t = conditions.get('time', 2025.0)
    L_HUMAN = conditions.get('L_HUMAN', np.interp(t, time_series_data.time, time_series_data.L_HUMAN))
    L_AI = conditions.get('L_AI', np.interp(t, time_series_data.time, time_series_data.L_AI))
    experiment_compute = conditions.get('experiment_compute', np.interp(t, time_series_data.time, time_series_data.experiment_compute))
    training_compute = conditions.get('training_compute', np.interp(t, time_series_data.time, time_series_data.training_compute))
    
    # Get cumulative progress at the specified time
    if 'cumulative_progress' in conditions:
        cumulative_progress = conditions['cumulative_progress']
    else:
        # Run integration to get cumulative progress at time t
        try:
            start_time = time_series_data.time[0]
            if t <= start_time:
                cumulative_progress = initial_progress
            else:
                time_range = [start_time, t]
                times, progress_values, research_stock_values = integrate_progress(time_range, initial_progress, time_series_data, params)
                cumulative_progress = progress_values[-1]
        except Exception as e:
            logger.warning(f"Integration failed for time {t}: {e}")
            raise e  # Don't silently fail
    
    # Compute automation fraction unless explicitly provided
    if 'automation_fraction' in conditions:
        automation_fraction = conditions['automation_fraction']
    else:
        automation_fraction = compute_automation_fraction(cumulative_progress, params)
    
    # Compute target variable
    if constraint.target_variable == 'progress_rate':
        cognitive_output = compute_cognitive_output(automation_fraction, L_AI, L_HUMAN, params.rho_cognitive, params.cognitive_output_normalization)
        
        # For constraints, we need to compute the research stock rate and software progress rate
        research_stock_rate = compute_research_stock_rate(experiment_compute, cognitive_output, params.alpha, params.rho_progress)
        
        # Get initial research stock rate for the calculation
        start_time = time_series_data.time[0]
        initial_automation = compute_automation_fraction(initial_progress, params)
        initial_L_HUMAN = np.interp(start_time, time_series_data.time, time_series_data.L_HUMAN)
        initial_L_AI = np.interp(start_time, time_series_data.time, time_series_data.L_AI)
        initial_experiment_compute = np.interp(start_time, time_series_data.time, time_series_data.experiment_compute)
        
        initial_cognitive_output = compute_cognitive_output(
            initial_automation, initial_L_AI, initial_L_HUMAN, 
            params.rho_cognitive, params.cognitive_output_normalization
        )
        initial_research_stock_rate = compute_research_stock_rate(
            initial_experiment_compute, initial_cognitive_output, 
            params.alpha, params.rho_progress
        )
        
        # Determine research stock at evaluation time
        if 'research_stock' in conditions:
            research_stock = conditions['research_stock']
        else:
            # For anchor constraints, we might need to approximate or assume research stock
            # grows at a constant rate (simple approximation)
            time_elapsed = t - start_time
            research_stock = params.research_stock_at_simulation_start + initial_research_stock_rate * time_elapsed
            research_stock = max(1e-6, research_stock)  # Ensure positive
        
        software_progress_rate = compute_software_progress_rate(
            research_stock, research_stock_rate, 
            params.research_stock_at_simulation_start, initial_research_stock_rate
        )
        
        model_value = compute_overall_progress_rate(software_progress_rate, training_compute, params.software_progress_share)
        model_value *= params.progress_rate_normalization
    elif constraint.target_variable == 'automation_fraction':
        model_value = automation_fraction
    elif constraint.target_variable == 'cognitive_output':
        model_value = compute_cognitive_output(automation_fraction, L_AI, L_HUMAN, params.rho_cognitive, params.cognitive_output_normalization)
    else:
        raise ValueError(f"Unknown target variable: {constraint.target_variable}")
    
    # Use relative error for better scaling across different variable types
    if constraint.target_value != 0:
        # Relative error: (model - target) / target
        error = (model_value - constraint.target_value) / abs(constraint.target_value)
        # Cap the relative error to prevent numerical issues
        error = np.clip(error, -cfg.RELATIVE_ERROR_CLIP, cfg.RELATIVE_ERROR_CLIP)
    else:
        # Absolute error if target is zero to avoid division by zero
        error = model_value - constraint.target_value
    
    logger.debug(f"Constraint evaluation: target={constraint.target_variable}, model_value={model_value:.6f}, target_value={constraint.target_value:.6f}, error={error:.6f}")
    
    return error


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
        fixed_params = fixed_params.copy() if fixed_params else []
        fixed_params.append('progress_rate_normalization')
    
    # Calculate progress_rate_normalization to ensure initial progress rate = 1
    start_time = time_series_data.time[0]
    initial_automation = compute_automation_fraction(initial_progress, initial_params)
    initial_L_HUMAN = np.interp(start_time, time_series_data.time, time_series_data.L_HUMAN)
    initial_L_AI = np.interp(start_time, time_series_data.time, time_series_data.L_AI)
    initial_experiment_compute = np.interp(start_time, time_series_data.time, time_series_data.experiment_compute)
    initial_training_compute = np.interp(start_time, time_series_data.time, time_series_data.training_compute)
    
    initial_cognitive_output = compute_cognitive_output(
        initial_automation, initial_L_AI, initial_L_HUMAN, 
        initial_params.rho_cognitive, initial_params.cognitive_output_normalization
    )
    initial_software_progress_rate = compute_software_progress_rate(
        initial_experiment_compute, initial_cognitive_output, 
        initial_params.alpha, initial_params.rho_progress
    )
    initial_overall_rate = compute_overall_progress_rate(
        initial_software_progress_rate, initial_training_compute, 
        initial_params.software_progress_share
    )
    
    # Set normalization so that initial rate = 1.0
    if initial_overall_rate > 0:
        initial_params.progress_rate_normalization = 1.0 / initial_overall_rate
    else:
        initial_params.progress_rate_normalization = 1.0
    
    logger.info(f"Calculated progress_rate_normalization: {initial_params.progress_rate_normalization}")
    
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
            
            # Try to evaluate constraint with initial parameters to check for basic feasibility
            try:
                error = evaluate_anchor_constraint(constraint, time_series_data, initial_params, initial_progress)
                if not np.isfinite(error):
                    logger.warning(f"Constraint evaluation produces non-finite error with initial parameters")
                    return False
            except Exception as e:
                logger.warning(f"Constraint evaluation failed with initial parameters: {e}")
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
        
        # Evaluate constraints
        total_error = 0.0
        for i, constraint in enumerate(anchor_constraints):
            try:
                error = evaluate_anchor_constraint(constraint, time_series_data, params, initial_progress)
                weighted_error = constraint.weight * (error ** 2)
                total_error += weighted_error
                logger.debug(f"Constraint {i+1}: error={error:.6f}, weighted_error={weighted_error:.6f}")
            except Exception as e:
                logger.warning(f"Error evaluating constraint {i+1}: {e}")
                return cfg.OBJECTIVE_FUNCTION_CONFIG['high_penalty']  # High penalty for constraint evaluation failures
        
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
        
        return total_error + regularization
    
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
    
    # Evaluate final constraint satisfaction
    constraint_evaluations = []
    for i, constraint in enumerate(anchor_constraints):
        try:
            error = evaluate_anchor_constraint(constraint, time_series_data, optimized_params, initial_progress)
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
    
    optimized_params._constraint_evaluations = constraint_evaluations
    
    return optimized_params, constraint_evaluations


class ProgressModel:
    """Main class for AI progress modeling"""
    
    def __init__(self, params: Parameters, time_series_data: TimeSeriesData):
        self.params = params
        self.data = time_series_data
        self.results = {}
        
    def compute_progress_trajectory(self, time_range: List[float], initial_progress: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute progress over specified time range
        
        Args:
            time_range: [start_time, end_time]
            initial_progress: Initial progress (defaults to 1.0)
        
        Returns:
            Tuple of (times, cumulative_progress_values, research_stock_values)
        """
        if initial_progress is None:
            initial_progress = 1.0  # Use a reasonable default value
        
        times, progress_values, research_stock_values = integrate_progress(time_range, initial_progress, self.data, self.params)
        
        # Calculate rates at each time step
        progress_rates = []
        research_stock_rates = []
        
        # We need the initial research stock rate for consistent software progress calculations
        start_time = self.data.time[0]
        initial_automation = compute_automation_fraction(initial_progress, self.params)
        initial_L_HUMAN = np.interp(start_time, self.data.time, self.data.L_HUMAN)
        initial_L_AI = np.interp(start_time, self.data.time, self.data.L_AI)
        initial_experiment_compute = np.interp(start_time, self.data.time, self.data.experiment_compute)
        initial_cognitive_output = compute_cognitive_output(
            initial_automation, initial_L_AI, initial_L_HUMAN, 
            self.params.rho_cognitive, self.params.cognitive_output_normalization
        )
        initial_research_stock_rate_val = compute_research_stock_rate(
            initial_experiment_compute, initial_cognitive_output, 
            self.params.alpha, self.params.rho_progress
        )
        
        for i in range(len(times)):
            state = [progress_values[i], research_stock_values[i]]
            rates = progress_rate_at_time(times[i], state, self.data, self.params, initial_research_stock_rate_val)
            progress_rates.append(rates[0])
            research_stock_rates.append(rates[1])
            
        # Store results
        self.results['times'] = times
        self.results['progress'] = progress_values
        self.results['research_stock'] = research_stock_values
        self.results['automation_fraction'] = [compute_automation_fraction(p, self.params) for p in progress_values]
        self.results['progress_rates'] = progress_rates
        self.results['research_stock_rates'] = research_stock_rates
        
        logger.info(f"Computed trajectory from {time_range[0]} to {time_range[1]}")
        logger.info(f"Progress: {progress_values[0]:.3f} -> {progress_values[-1]:.3f}")
        logger.info(f"Research Stock: {research_stock_values[0]:.3f} -> {research_stock_values[-1]:.3f}")
        logger.info(f"Automation: {self.results['automation_fraction'][0]:.3f} -> {self.results['automation_fraction'][-1]:.3f}")
        
        return times, progress_values, research_stock_values
    
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