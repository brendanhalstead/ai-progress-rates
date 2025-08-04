"""
Configuration file for the progress model.
Contains hardcoded values for numerical stability, caps, and other model parameters.
"""

# =============================================================================
# NUMERICAL STABILITY & PRECISION
# =============================================================================
RHO_COBB_DOUGLAS_THRESHOLD = 1e-9
RHO_LEONTIEF_THRESHOLD = -50.0
SIGMOID_EXPONENT_CLAMP = 100.0
AUTOMATION_FRACTION_CLIP_MIN = 1e-9

# =============================================================================
# PARAMETER CLIPPING (in Parameters.__post_init__)
# =============================================================================
RHO_CLIP_MIN = -50.0
PARAM_CLIP_MIN = 1e-6
AUTOMATION_SLOPE_CLIP_MIN = 0.1
AUTOMATION_SLOPE_CLIP_MAX = 10.0
RESEARCH_STOCK_START_MIN = 1e-10
NORMALIZATION_MIN = 1e-10
ZETA_CLIP_MIN = 0.1
ZETA_CLIP_MAX = 1.0

# Aggregate Research Taste configuration
AGGREGATE_RESEARCH_TASTE_BASELINE = 1.0
AGGREGATE_RESEARCH_TASTE_FALLBACK = 1.0

# Research Taste Distribution Parameters (Log-Normal)
TOP_PERCENTILE = 0.01                    # fraction classed as "top" researchers
MEDIAN_TO_TOP_TASTE_GAP = 3.25           # threshold taste ÷ median taste

# Research Taste Schedule Configuration
TASTE_SCHEDULE_TYPES = ["exponential", "sigmoid", "sd_per_progress"]  # Available schedule types
DEFAULT_TASTE_SCHEDULE_TYPE = "exponential"

# =============================================================================
# MODEL RATE & VALUE CAPS
# =============================================================================
MAX_RESEARCH_STOCK_RATE = 10000000000.0
MAX_NORMALIZED_PROGRESS_RATE = 100.0
TIME_EXTRAPOLATION_WINDOW = 10.0

# =============================================================================
# ODE INTEGRATION
# =============================================================================
PROGRESS_ODE_CLAMP_MAX = 1e6
RESEARCH_STOCK_ODE_CLAMP_MAX = 1e10
ODE_MAX_STEP = 1.0
EULER_FALLBACK_MIN_STEPS = 100
EULER_FALLBACK_STEPS_PER_YEAR = 10
DENSE_OUTPUT_POINTS = 100

# ODE step size logging configuration
ODE_STEP_SIZE_LOGGING = False
ODE_SMALL_STEP_THRESHOLD = 1e-6  # Threshold for warning about very small steps
ODE_STEP_VARIATION_THRESHOLD = 100.0  # Threshold for warning about step size variation

# =============================================================================
# PARAMETER ESTIMATION
# =============================================================================
RELATIVE_ERROR_CLIP = 100.0

# Parameter bounds for optimization
PARAMETER_BOUNDS = {
    'rho_cognitive': (-10, 0),
    'rho_progress': (-1, 1),
    'alpha': (0.05, 0.95),
    'software_progress_share': (0.05, 0.95),
    'automation_fraction_at_superhuman_coder': (0.1, 0.99),
    'progress_at_half_sc_automation': (1.0, 500),
    'automation_slope': (0.1, 10.0),
    'cognitive_output_normalization': (0.00001, 10),
    'zeta': (ZETA_CLIP_MIN, ZETA_CLIP_MAX),
    # AI Research Taste parameter bounds
    'ai_research_taste_at_superhuman_coder': (0.1, 0.99),
    'progress_at_half_ai_research_taste': (1.0, 500),
    'ai_research_taste_slope': (0.1, 10.0),
    'progress_at_sc': (1.0, 500)
}

# Validation thresholds for parameter combinations
PARAM_VALIDATION_THRESHOLDS = {
    'automation_fraction_superhuman_coder_min': 0.05,
    'automation_fraction_superhuman_coder_max': 1,
    'progress_at_half_automation_min': 0.0, # should be > 0
    'automation_slope_min': 0.0, # should be > 0
    'automation_slope_max': 20.0,
    'rho_extreme_abs': 0.8,
    'rho_product_max': 0.5,
    'cognitive_output_normalization_max': 10
}

FEASIBILITY_CHECK_THRESHOLDS = {
    'progress_rate_target_max': 1000.0,
}

OBJECTIVE_FUNCTION_CONFIG = {
    'high_penalty': 1e6,
    'elasticity_regularization_weight': 0.001,
    'boundary_avoidance_regularization_weight': 0.001,
    'boundary_avoidance_threshold': 0.35,
}

OPTIMIZATION_CONFIG = {
    'early_termination_fun_threshold_excellent': 1e-6,
    'early_termination_fun_threshold_good': 1e-3,
}

STRATEGIC_STARTING_POINTS_CONFIG = {
    'extreme_factor_min': 0.1,
    'extreme_factor_max': 0.9,
    'lhs_points': 5,
    'high_progress_rate_threshold': 2.0,
    'rho_adjustment_factor': 0.8,
    'high_automation_threshold': 0.5,
    'progress_at_half_automation_adjustment_factor': 0.7,
    'perturbed_points': 3,
    'critical_param_perturbation_factor': 0.1,
    'other_param_perturbation_factor': 0.2
}

# =============================================================================
# DEFAULT MODEL PARAMETERS
# =============================================================================
DEFAULT_PARAMETERS = {
    'rho_cognitive': -2,
    'rho_progress': -0.1,
    'alpha': 0.5,
    'software_progress_share': 0.5,
    'automation_fraction_at_superhuman_coder': 0.99,
    'progress_at_half_sc_automation': 20.0,
    'automation_slope': 1.6,
    'progress_rate_normalization': 1.0,  # This is usually auto-calculated
    'cognitive_output_normalization': 1.26e-3,
    'zeta': 0.2,
    # AI Research Taste parameters
    'ai_research_taste_at_superhuman_coder': 0.95,
    'progress_at_half_ai_research_taste': 30.0,
    'ai_research_taste_slope': 1.2,
    'taste_schedule_type': DEFAULT_TASTE_SCHEDULE_TYPE,
    'progress_at_sc': 50.0,
} 