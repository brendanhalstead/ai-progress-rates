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
experiment_compute_exponent_CLIP_MIN = 0.001
experiment_compute_exponent_CLIP_MAX = 10.0
# Clipping for parallel penalty parameter
PARALLEL_PENALTY_MIN = 0.0
PARALLEL_PENALTY_MAX = 1.0

# =============================================================================
# OTHER RANDOM CONSTANTS
# =============================================================================
# Aggregate Research Taste configuration
AGGREGATE_RESEARCH_TASTE_BASELINE = 1.0
AGGREGATE_RESEARCH_TASTE_FALLBACK = 1.0

# Research Taste Distribution Parameters (Log-Normal)
TOP_PERCENTILE = 0.01                    # fraction classed as "top" researchers
MEDIAN_TO_TOP_TASTE_GAP = 3.25           # threshold taste ÷ median taste

# Research Taste Schedule Configuration (UI-level options)
# Internally, both SD-based options map to 'sd_per_progress' logic; units differ in UI only
TASTE_SCHEDULE_TYPES = ["SDs per effective OOM", "SDs per progress-year"]
DEFAULT_TASTE_SCHEDULE_TYPE = "SDs per effective OOM"

# Horizon Extrapolation Configuration
HORIZON_EXTRAPOLATION_TYPES = ["exponential", "decaying doubling time"]  # Available extrapolation types
DEFAULT_HORIZON_EXTRAPOLATION_TYPE = "decaying doubling time"

# Manual Horizon Fitting Parameters
DEFAULT_present_day = 2025.25
DEFAULT_present_horizon = 15  # Will be optimized if None
DEFAULT_present_doubling_time = 0.460  # Will be optimized if None
DEFAULT_DOUBLING_DECAY_RATE = 0.050  # Will be optimized if None

# AI Research Taste clipping bounds
AI_RESEARCH_TASTE_MIN = 0.0
AI_RESEARCH_TASTE_MAX = 1e10  # Match the upper bound from PARAMETER_BOUNDS
AI_RESEARCH_TASTE_MAX_SD = 23

# Baseline Annual Compute Multiplier
BASELINE_ANNUAL_COMPUTE_MULTIPLIER_DEFAULT = 4.5

BASE_FOR_SOFTWARE_LOM = 10.0

# labor and compute anchors are based on these
REFERENCE_YEAR = 2024.0
REFERENCE_LABOR_CHANGE = 30.0
REFERENCE_COMPUTE_CHANGE = 0.1

# benchmarks and gaps mode

# =============================================================================
# MODEL RATE & VALUE CAPS
# =============================================================================
MAX_RESEARCH_EFFORT = 1000000000000000000000000.0
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

# Parameter bounds for optimization, determines bounds for website sliders
PARAMETER_BOUNDS = {
    'rho_coding_labor': (-10, 0),
    'rho_experiment_capacity': (-1, 1),
    'alpha_experiment_capacity': (0.05, 0.95),
    'r_software': (0.1, 10),
    'automation_fraction_at_superhuman_coder': (0.1, 1.0),
    'swe_multiplier_at_present_day': (1.0, 10.0),
    'coding_labor_normalization': (0.00001, 10),
    'experiment_compute_exponent': (experiment_compute_exponent_CLIP_MIN, experiment_compute_exponent_CLIP_MAX),
    # AI Research Taste parameter bounds
    'ai_research_taste_at_superhuman_coder_sd': (-10, AI_RESEARCH_TASTE_MAX_SD),
    'ai_research_taste_slope': (0.1, 10.0),
    'progress_at_sc': (1.0, 500),
    'sc_time_horizon_minutes': (1000, 100000000000),
    'pre_gap_sc_time_horizon': (1000, 100000000000),
    # Manual horizon fitting parameter bounds
    'present_day': (2020.0, 2030.0),
    'present_horizon': (0.01, 100),  # minutes
    'present_doubling_time': (0.01, 2),  # doubling time in present years
    'doubling_decay_rate': (-0.5, 0.5),  # decay rate
    # Baseline Annual Compute Multiplier bounds
    'baseline_annual_compute_multiplier': (1.0, 20.0),
    # coding_labor_exponent deprecated in favor of parallel_penalty
    # exp capacity pseudoparameters
    'inf_labor_asymptote': (0, 100000),
    'inf_compute_asymptote': (0, 100000),
    'labor_anchor_exp_cap': (0, 1000),
    'compute_anchor_exp_cap': (0, 1000),
    'inv_compute_anchor_exp_cap': (0, 10),
    # parallel penalty for experiment capacity CES
    'parallel_penalty': (0.0, 1.0),
}

# Validation thresholds for parameter combinations
PARAM_VALIDATION_THRESHOLDS = {
    'automation_fraction_superhuman_coder_min': 0.05,
    'automation_fraction_superhuman_coder_max': 1,
    # removed legacy automation sigmoid thresholds
    'rho_extreme_abs': 0.8,
    'rho_product_max': 0.5,
    'coding_labor_normalization_max': 10
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
# DEFAULT MODEL PARAMETERS (FOR WEBSITE ONLY, MONTE CARLO USES CONFIG/SAMPLING_CONFIG.YAML)
# =============================================================================
DEFAULT_PARAMETERS = {
    'rho_coding_labor': -2,
    'direct_input_exp_cap_ces_params': False,
    'rho_experiment_capacity': -0.137,
    'alpha_experiment_capacity': 0.701,
    'r_software': 2.40,
    'automation_fraction_at_superhuman_coder': 1.0,
    'swe_multiplier_at_present_day': 1.35,
    'automation_interp_type': "linear",
    'coding_labor_normalization': 1,
    'experiment_compute_exponent': 0.562,
    # AI Research Taste parameters
    'ai_research_taste_at_superhuman_coder': 0.95,
    'ai_research_taste_at_superhuman_coder_sd': 0,  # Optional: specify SC taste in SD-within-human-range
    'ai_research_taste_slope': 2.0,
    'taste_schedule_type': DEFAULT_TASTE_SCHEDULE_TYPE,
    'progress_at_sc': None,
    'sc_time_horizon_minutes': 1223000000,
    'horizon_extrapolation_type': DEFAULT_HORIZON_EXTRAPOLATION_TYPE,
    'automation_anchors': None,
    # Manual horizon fitting parameters
    'present_day': DEFAULT_present_day,
    'present_horizon': DEFAULT_present_horizon,
    'present_doubling_time': DEFAULT_present_doubling_time,
    'doubling_decay_rate': DEFAULT_DOUBLING_DECAY_RATE,
    # Baseline Annual Compute Multiplier
    'baseline_annual_compute_multiplier': BASELINE_ANNUAL_COMPUTE_MULTIPLIER_DEFAULT,
    # coding_labor_exponent deprecated in favor of parallel_penalty
    # exp capacity pseudoparameters
    'inf_labor_asymptote': 15.0,
    'inf_compute_asymptote': 5000,
    'labor_anchor_exp_cap': 1.6,
    'compute_anchor_exp_cap': None,
    'inv_compute_anchor_exp_cap': 3.33,
    # benchmarks and gaps mode
    'include_gap': 'no gap',
    'gap_years': 1.5,
    'pre_gap_sc_time_horizon': 575500.0,
    # penalty on parallel coding labor contribution in exp capacity
    'parallel_penalty': 0.52,
}

# =============================================================================
# WEBSITE PLOT CONFIGURATION - SINGLE SOURCE OF TRUTH
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
    'plot_coding_labor_with_compute': {
        'title': 'Inputs to Experiment Capacity',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Nonsense units', 'type': 'log'},
        'data_keys': ['times', 'coding_labors']
    },
    'plot_cognitive_components': {
        'title': 'Normalized Coding Labor, Humans and AIs',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Nonsense units', 'type': 'log'},
        'data_keys': ['times', 'coding_labors', 'human_labor_contributions']
    },
    'plot_ai_coding_labor_multiplier': {
        'title': 'AI Parallel Coding Labor Multiplier',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'Coding labor per human', 'type': 'log'},
        'data_keys': ['times', 'ai_coding_labor_multipliers']
    },
    
    # Software R&D plots
    'plot_research_stock': {
        'title': 'Cumulative Research Effort',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'mysterious unit (dollars??)', 'type': 'log'},
        'data_keys': ['times', 'research_stocks']
    },
    'plot_research_effort': {
        'title': 'Research Effort',
        'x_axis': {'title': 'Time', 'type': 'linear'},
        'y_axis': {'title': 'mysterious unit (dollars/year??)', 'type': 'log'},
        'data_keys': ['times', 'research_efforts']
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
        'data_keys': ['times', 'ai_coding_labor_multipliers', 'ai_research_stock_multipliers', 
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
        ],
        'explanation': """
## Time Horizons

This tab shows how the 80% time horizon of frontier AI systems evolves over time. The red dashed line shows the required time horizon for the Superhuman Coder milestone. The SC time horizon can be adjusted in the sidebar.
        """
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
        ],
        'explanation': """
## Input Time Series

This tab displays the exogenous inputs that drive AI progress in the model:

### Human Labor (Top Left)
Human researchers and engineers working on AGI development, calculated using data from [this post](https://forum.effectivealtruism.org/posts/xoX936hEvpxToeuLw/estimating-the-substitutability-between-compute-and).

### AI Labor (Top Right) 
Inference compute budget for running agents to be spread across automated tasks. Measured in human-equivalents.

### Experiment Compute (Bottom Left)
Basically assumed to scale with training compute. Normalized arbitrarily to 1 in 2019.

### Training Compute Growth Rate (Bottom Right)
Factor by which frontier training run compute grows each year, measured in OOMs.
        """
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
        ],
        'explanation': """
## Automation (from SC time horizon and effective compute)

This tab shows how the current capability level feeds into the production function.

### Coding Automation Fraction (Top Left)
Abstract "percentage of coding tasks performed by AI". Exponentially interpolates between two anchors, using the current effective OOMs:
1. At (anchor time), it should equal (whatever percentage makes the coding labor multiplier equal a given value). set both of these in the sidebar.
2. At SC, it should equal 100%.

### Various plots of research taste
We assume that each OOM of effective compute increases AI research taste by a fixed number of standard deviations within the human range. 
The units of research taste are such that if you replace all researchers with researchers of a given taste level, holding experiment capacity constant, research effort scales by that taste level.
The scale is informed by our survey of frontier AGI researchers, where they gave a median 3.25x ratio in the speedups from replacing everyone with the top researcher vs replacing everyone with the median researcher.
We assume that research taste (in these units) is lognormally distributed among human researchers with a mean of 1. *Note: AI research taste gets capped at +23 standard deviations.*

### Aggregate Research Taste (Bottom Left)
Average of the research taste distribution, after replacing sub-AI human researchers with AI researchers.
        """
    },
    'coding_labor': {
        'id': 'cognitive-output',
        'name': 'Coding Labor',
        'rows': 2, 'cols': 2,
        'plots': [
            {'function': 'plot_labor_comparison', 'position': (1, 1)},
            {'function': 'plot_automation_fraction', 'position': (1, 2)},
            {'function': 'plot_ai_coding_labor_multiplier', 'position': (2, 1)},
            {'function': 'plot_cognitive_components', 'position': (2, 2)}
        ],
        'explanation': """
## Coding Labor (from coding automation fraction, human labor, and AI labor)

### Available labor pools
See Inputs tab.

### Coding Automation Fraction
See Automation tab.

### AI Coding Labor Multiplier (Bottom Left)
Obtained by taking the (pre-discounting) combined coding labor and dividing it by the (pre-discounting) human labor.

### Normalized Coding Labor (Bottom Right)
Depicts the actual combined coding labor that gets passed into the next tab. The combination is given by (insert equation).

        """
    },
    'research_effort': {
        'id': 'research-effort',
        'name': 'Research Effort',
        'rows': 2, 'cols': 2,
        'plots': [
            {'function': 'plot_coding_labor_with_compute', 'position': (1, 1)},
            
            {'function': 'plot_experiment_capacity', 'position': (1, 2)},
            {'function': 'plot_aggregate_research_taste', 'position': (2, 1)},
            {'function': 'plot_research_effort', 'position': (2, 2)}
        ],
        'explanation': """
## Research Effort (from coding labor, experiment compute, and research taste)

Research effort is aggregate research taste times experiment capacity. Experiment capacity is the CES combination of coding labor and experiment compute.
        """
    },
    'software_rd': {
        'id': 'software-rd',
        'name': 'Software R&D',
        'rows': 2, 'cols': 2,
        'plots': [
            {'function': 'plot_research_effort', 'position': (1, 1)},
            {'function': 'plot_software_progress_rate', 'position': (1, 2)},
            {'function': 'plot_research_stock', 'position': (2,1)},
            {'function': 'plot_software_efficiency', 'position': (2, 2)}
            
        ],
        'explanation': """
## Software Research & Development (from research effort)
This captures the diminishing returns to research effort. We use the same semi-endogenous "software law of motion" as Davidson's FTM. Except that we don't impose physical limits to software efficiency at the moment.
      """
    },
    'combined_progress': {
        'id': 'combined-progress',
        'name': 'Effective Compute',
        'rows': 2, 'cols': 1,
        'subplot_titles': ['', 'Components of effective compute growth'],
        'plots': [
            {'function': 'plot_cumulative_progress', 'position': (1, 1)},
            {'function': 'plot_rate_components', 'position': (2, 1)},
        ],
        'explanation': """
## Effective Compute (from software growth rate and training compute growth rate)
OOMs/year from hardware combine with OOMs/year from software efficiency to get OOMs/year of effective compute.
The cumulative OOMs of effective compute feeds back into the functions that determine 
- time horizon,
- automation fraction, and  
- AI research taste.
        """
    },
    'other_metrics': {
        'id': 'other-metrics',
        'name': 'Other Metrics',
        'rows': 3, 'cols': 2,
        'plots': [
            {'function': 'plot_ai_coding_labor_multiplier', 'position': (1, 1)},
            {'function': 'plot_ai_software_progress_multiplier', 'position': (1, 2)},
            {'function': 'plot_ai_overall_progress_multiplier', 'position': (2, 1)},
            {'function': 'plot_human_only_progress_rate', 'position': (2, 2)},
            {'function': 'plot_horizon_lengths_vs_progress', 'position': (3, 1)}
        ],
        'explanation': """
## Additional Metrics and Multipliers
Under construction, don't trust these.
        """
    }
} 