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
DEFAULT_TASTE_SCHEDULE_TYPE = "sd_per_progress"

# Horizon Extrapolation Configuration
HORIZON_EXTRAPOLATION_TYPES = ["exponential", "decaying doubling time"]  # Available extrapolation types
DEFAULT_HORIZON_EXTRAPOLATION_TYPE = "decaying doubling time"

# Manual Horizon Fitting Parameters
DEFAULT_ANCHOR_TIME = 2025.25
DEFAULT_ANCHOR_HORIZON = 15  # Will be optimized if None
DEFAULT_ANCHOR_DOUBLING_TIME = 0.550  # Will be optimized if None
DEFAULT_DOUBLING_DECAY_RATE = 0.050  # Will be optimized if None

# AI Research Taste clipping bounds
AI_RESEARCH_TASTE_MIN = 0.0
AI_RESEARCH_TASTE_MAX = 1e10  # Match the upper bound from PARAMETER_BOUNDS
AI_RESEARCH_TASTE_MAX_SD = 23

# Baseline Annual Compute Multiplier
BASELINE_ANNUAL_COMPUTE_MULTIPLIER_DEFAULT = 4.5

BASE_FOR_SOFTWARE_LOM = 10.0

# =============================================================================
# MODEL RATE & VALUE CAPS
# =============================================================================
MAX_RESEARCH_STOCK_RATE = 1000000000000000000000000.0
MAX_NORMALIZED_PROGRESS_RATE = 1000.0
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
    'software_scale': (0.1, 10),
    'automation_fraction_at_superhuman_coder': (0.1, 1.0),
    'progress_at_half_sc_automation': (1.0, 500),
    'automation_slope': (0.1, 10.0),
    'swe_multiplier_at_anchor_time': (1.0, 10.0),
    'cognitive_output_normalization': (0.00001, 10),
    'zeta': (ZETA_CLIP_MIN, ZETA_CLIP_MAX),
    # AI Research Taste parameter bounds
    'ai_research_taste_at_superhuman_coder': (0.1, 5),
    'progress_at_half_ai_research_taste': (1.0, 500),
    'ai_research_taste_slope': (0.1, 10.0),
    'progress_at_sc': (1.0, 500),
    'sc_time_horizon_minutes': (1000, 100000000000),
    # Manual horizon fitting parameter bounds
    'anchor_time': (2020.0, 2030.0),
    'anchor_horizon': (0.01, 100),  # minutes
    'anchor_doubling_time': (0.01, 2),  # doubling time in progress units
    'doubling_decay_rate': (0.001, 0.5),  # decay rate
    # Baseline Annual Compute Multiplier bounds
    'baseline_annual_compute_multiplier': (1.0, 20.0),
    # Lambda parameter bounds
    'lambda': (0.0, 2.0)
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
    'rho_progress': -0.15,
    'alpha': 0.7,
    'software_scale': 2.25,
    'automation_fraction_at_superhuman_coder': 1.0,
    'progress_at_half_sc_automation': 20.0,
    'automation_slope': 1.0,
    'swe_multiplier_at_anchor_time': 1.05,
    'cognitive_output_normalization': 0.3,
    'zeta': 0.55,
    # AI Research Taste parameters
    'ai_research_taste_at_superhuman_coder': 0.95,
    'progress_at_half_ai_research_taste': 30.0,
    'ai_research_taste_slope': 2.0,
    'taste_schedule_type': DEFAULT_TASTE_SCHEDULE_TYPE,
    'progress_at_sc': None,
    'sc_time_horizon_minutes': 124560.0, # 124560 minutes = 1 work year
    'horizon_extrapolation_type': DEFAULT_HORIZON_EXTRAPOLATION_TYPE,
    'automation_anchors': None,
    # Manual horizon fitting parameters
    'anchor_time': DEFAULT_ANCHOR_TIME,
    'anchor_horizon': DEFAULT_ANCHOR_HORIZON,
    'anchor_doubling_time': DEFAULT_ANCHOR_DOUBLING_TIME,
    'doubling_decay_rate': DEFAULT_DOUBLING_DECAY_RATE,
    # Baseline Annual Compute Multiplier
    'baseline_annual_compute_multiplier': BASELINE_ANNUAL_COMPUTE_MULTIPLIER_DEFAULT,
    # Lambda parameter
    'lambda': 0.65,
}

# =============================================================================
# PLOT CONFIGURATION - SINGLE SOURCE OF TRUTH
# =============================================================================

PLOT_METADATA = {
    # Time Horizons plots
    'plot_horizon_lengths': {
        'title': 'Time Horizon Lengths',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Horizon Length (log scale)', 'type': 'log', 'range': [-3, 11], 'custom_ticks': True},
        'data_keys': ['times', 'horizon_lengths'],
        'special_handling': ['metr_data', 'sc_time_horizon_minutes']
    },
    'plot_horizon_lengths_vs_progress': {
        'title': 'Horizon Length vs Progress',
        'x_axis': {'title': 'Cumulative Progress', 'type': 'linear'},
        'y_axis': {'title': 'Horizon Length (log scale)', 'type': 'log', 'range': [-3, 11], 'custom_ticks': True},
        'data_keys': ['progress_values', 'horizon_lengths'],
        'special_handling': ['metr_data', 'sc_time_horizon_minutes', 'progress_at_sc']
    },
    
    # Input plots
    'plot_human_labor': {
        'title': 'Human Labor',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Total AGI Researchers', 'type': 'log'},
        'data_keys': ['times', 'L_HUMAN']
    },
    'plot_ai_labor': {
        'title': 'AI Labor',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Human Equivalents', 'type': 'log'},
        'data_keys': ['times', 'L_AI']
    },
    'plot_experiment_compute': {
        'title': 'Experiment Compute',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': '(scaled) FLOPs', 'type': 'log'},
        'data_keys': ['times', 'experiment_compute']
    },
    'plot_training_compute_growth_rate': {
        'title': 'Training Compute Growth Rate',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'OOMs/year', 'type': 'linear'},
        'data_keys': ['times', 'training_compute_growth_rate']
    },
    'plot_software_efficiency': {
        'title': 'Software Efficiency',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'OOMs', 'type': 'linear'},
        'data_keys': ['times', 'software_efficiency']
    },
    'plot_labor_comparison': {
        'title': 'Available Labor Pools',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Automated-human equivalents', 'type': 'log'},
        'data_keys': ['time_series']
    },
    'plot_compute_comparison': {
        'title': 'Compute Comparison',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Compute (log scale)', 'type': 'log'},
        'data_keys': ['time_series']
    },
    
    # Automation plots
    'plot_automation_fraction': {
        'title': 'Coding Automation Fraction',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'SWE Tasks Automated, %', 'type': 'linear'},
        'data_keys': ['times', 'automation_fraction']
    },
    'plot_progress_vs_automation': {
        'title': 'Progress vs Automation',
        'x_axis': {'title': 'Cumulative Progress', 'type': 'linear'},
        'y_axis': {'title': 'SWE Tasks Automated, %', 'type': 'linear'},
        'data_keys': ['progress', 'automation_fraction']
    },
    'plot_ai_research_taste': {
        'title': 'AI Research Taste',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Research effort per unit experiment', 'type': 'log'},
        'data_keys': ['times', 'ai_research_taste']
    },
    'plot_ai_research_taste_sd': {
        'title': 'AI Research Taste (Standard Deviations)',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Std. Dev. from OB Median', 'type': 'linear'},
        'data_keys': ['times', 'ai_research_taste_sd']
    },
    'plot_ai_research_taste_quantile': {
        'title': 'AI Research Taste (Quantile)',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Quantile Among OB Researchers', 'type': 'linear'},
        'data_keys': ['times', 'ai_research_taste_quantile']
    },
    'plot_aggregate_research_taste': {
        'title': 'Aggregate Research Taste, Humans and AIs',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Research effort per unit experiment', 'type': 'log'},
        'data_keys': ['times', 'aggregate_research_taste']
    },
    'plot_ai_vs_aggregate_research_taste': {
        'title': 'AI vs Aggregate Research Taste',
        'x_axis': {'title': 'AI Research Taste', 'type': 'linear'},
        'y_axis': {'title': 'Aggregate Research Taste', 'type': 'linear'},
        'data_keys': ['ai_research_taste', 'aggregate_research_taste']
    },
    
    # Cognitive Output plots
    'plot_cognitive_output_with_compute': {
        'title': 'Inputs to Experiment Capacity',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Nonsense units', 'type': 'log'},
        'data_keys': ['times', 'cognitive_outputs']
    },
    'plot_cognitive_components': {
        'title': 'Labor Contributions',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Nonsense units', 'type': 'log'},
        'data_keys': ['times', 'cognitive_outputs', 'human_labor_contributions']
    },
    'plot_ai_cognitive_output_multiplier': {
        'title': 'AI Parallel Coding Labor Multiplier',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Coding labor per human', 'type': 'log'},
        'data_keys': ['times', 'ai_cognitive_output_multipliers']
    },
    
    # Software R&D plots
    'plot_research_stock': {
        'title': 'Cumulative Research Effort',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'mysterious unit (dollars??)', 'type': 'log'},
        'data_keys': ['times', 'research_stocks']
    },
    'plot_research_stock_rate': {
        'title': 'Research Effort',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'mysterious unit (dollars/year??)', 'type': 'log'},
        'data_keys': ['times', 'research_stock_rates']
    },
    'plot_experiment_capacity': {
        'title': 'Experiment Capacity',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Effort per unit taste', 'type': 'log'},
        'data_keys': ['times', 'experiment_capacity']
    },
    'plot_software_progress_rate': {
        'title': 'Software Progress Rate',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'OOMs/year', 'type': 'linear'},
        'data_keys': ['times', 'software_progress_rates']
    },
    
    # Combined Progress plots
    'plot_cumulative_progress': {
        'title': 'Components of Effective Compute',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'OOMs', 'type': 'linear'},
        'data_keys': ['times', 'progress', 'training_compute']
    },
    'plot_progress_rate': {
        'title': 'Effective Compute Growth Rate',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'OOMs/year', 'type': 'linear'},
        'data_keys': ['times', 'progress_rates']
    },
    'plot_rate_components': {
        'title': 'Rate Components',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'OOMs/year', 'type': 'linear', 'range': [0, 10]},
        'data_keys': ['times', 'progress_rates', 'training_compute_growth_rate']
    },
    
    # Other Metrics plots
    'plot_ai_research_stock_multiplier': {
        'title': 'AI Research Stock Multiplier',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': '???', 'type': 'log'},
        'data_keys': ['times', 'ai_research_stock_multipliers']
    },
    'plot_ai_software_progress_multiplier': {
        'title': 'AI Software Progress Multiplier',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': '???', 'type': 'log'},
        'data_keys': ['times', 'ai_software_progress_multipliers']
    },
    'plot_ai_overall_progress_multiplier': {
        'title': 'AI Overall Progress Multiplier',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': '???', 'type': 'log'},
        'data_keys': ['times', 'ai_overall_progress_multipliers']
    },
    'plot_all_ai_multipliers': {
        'title': 'All AI Multipliers',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Multiplier (log scale)', 'type': 'log'},
        'data_keys': ['times', 'ai_cognitive_output_multipliers', 'ai_research_stock_multipliers', 
                     'ai_software_progress_multipliers', 'ai_overall_progress_multipliers']
    },
    'plot_human_only_progress_rate': {
        'title': 'Human-Only Progress Rate',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Human-Only Rate', 'type': 'linear'},
        'data_keys': ['times', 'human_only_progress_rates']
    },
    'plot_automation_multiplier': {
        'title': 'Automation Multiplier',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Automation Multiplier', 'type': 'linear'},
        'data_keys': ['times', 'automation_multipliers']
    }
}

TAB_CONFIGURATIONS = {
    'time_horizons': {
        'id': 'time-horizons',
        'name': 'Time Horizons',
        'rows': 1, 'cols': 1,
        'subplot_titles': [''],  # Empty title to avoid duplication with tab title
        'plots': [
            {'function': 'plot_horizon_lengths', 'position': (1, 1)},
        ]
    },
    'inputs': {
        'id': 'inputs',
        'name': 'Inputs',
        'rows': 2, 'cols': 2,
        'plots': [
            {'function': 'plot_human_labor', 'position': (1, 1)},
            {'function': 'plot_ai_labor', 'position': (1, 2)},
            {'function': 'plot_experiment_compute', 'position': (2, 1)},
            {'function': 'plot_training_compute_growth_rate', 'position': (2, 2)}
        ]
    },
    'automation': {
        'id': 'automation',
        'name': 'Automation',
        'rows': 3, 'cols': 2,
        'plots': [
            {'function': 'plot_automation_fraction', 'position': (1, 1)},
            {'function': 'plot_ai_research_taste_sd', 'position': (1, 2)},
            {'function': 'plot_ai_research_taste', 'position': (2, 1)},
            {'function': 'plot_ai_research_taste_quantile', 'position': (2, 2)},
            {'function': 'plot_aggregate_research_taste', 'position': (3, 1)}
        ]
    },
    'cognitive_output': {
        'id': 'cognitive-output',
        'name': 'Coding Labor',
        'rows': 2, 'cols': 2,
        'plots': [
            {'function': 'plot_labor_comparison', 'position': (1, 1)},
            {'function': 'plot_cognitive_components', 'position': (1, 2)},
            {'function': 'plot_automation_fraction', 'position': (2, 1)},
            {'function': 'plot_ai_cognitive_output_multiplier', 'position': (2, 2)}
        ]
    },
    'research_effort': {
        'id': 'research-effort',
        'name': 'Research Effort',
        'rows': 2, 'cols': 2,
        'plots': [
            {'function': 'plot_cognitive_output_with_compute', 'position': (1, 1)},
            
            {'function': 'plot_experiment_capacity', 'position': (1, 2)},
            {'function': 'plot_aggregate_research_taste', 'position': (2, 1)},
            {'function': 'plot_research_stock_rate', 'position': (2, 2)}
        ]
    },
    'software_rd': {
        'id': 'software-rd',
        'name': 'Software R&D',
        'rows': 2, 'cols': 2,
        'plots': [
            {'function': 'plot_research_stock_rate', 'position': (1, 1)},
            {'function': 'plot_software_progress_rate', 'position': (1, 2)},
            {'function': 'plot_research_stock', 'position': (2,1)},
            {'function': 'plot_software_efficiency', 'position': (2, 2)}
            
        ]
    },
    'combined_progress': {
        'id': 'combined-progress',
        'name': 'Effective Compute',
        'rows': 2, 'cols': 1,
        'subplot_titles': ['', 'Components of effective compute growth'],
        'plots': [
            {'function': 'plot_cumulative_progress', 'position': (1, 1)},
            {'function': 'plot_rate_components', 'position': (2, 1)},
        ]
    },
    'other_metrics': {
        'id': 'other-metrics',
        'name': 'Other Metrics',
        'rows': 3, 'cols': 2,
        'plots': [
            {'function': 'plot_ai_cognitive_output_multiplier', 'position': (1, 1)},
            {'function': 'plot_ai_software_progress_multiplier', 'position': (1, 2)},
            {'function': 'plot_ai_overall_progress_multiplier', 'position': (2, 1)},
            {'function': 'plot_human_only_progress_rate', 'position': (2, 2)},
            {'function': 'plot_horizon_lengths_vs_progress', 'position': (3, 1)}
        ]
    }
} 