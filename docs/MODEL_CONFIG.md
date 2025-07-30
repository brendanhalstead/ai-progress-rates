# Model Configuration System

This document provides comprehensive documentation for the configuration system implemented in `model_config.py`. The configuration system allows fine-tuning of numerical stability parameters, optimization settings, and model behavior without modifying core code.

## Overview

The model configuration system provides centralized control over:
- **Numerical stability parameters**: Thresholds and limits for mathematical edge cases
- **Parameter bounds**: Valid ranges for optimization and validation
- **Integration settings**: ODE solver configuration and fallback strategies
- **Optimization control**: Regularization weights and termination criteria
- **Performance monitoring**: Logging and diagnostic settings

All configuration values are designed to ensure numerical stability while maintaining model flexibility across diverse scenarios.

## Configuration Sections

### 1. Numerical Stability & Precision

Critical thresholds for handling mathematical edge cases and ensuring numerical stability.

```python
# CES function edge case detection
RHO_COBB_DOUGLAS_THRESHOLD = 1e-9    # Threshold for Cobb-Douglas approximation
RHO_LEONTIEF_THRESHOLD = -50.0       # Minimum rho value before Leontief behavior

# Sigmoid function stability
SIGMOID_EXPONENT_CLAMP = 100.0        # Maximum exponent to prevent overflow
AUTOMATION_FRACTION_CLIP_MIN = 1e-9   # Minimum automation fraction
```

**Purpose**: These thresholds determine when the model switches between different mathematical formulations to maintain numerical stability:
- **Cobb-Douglas threshold**: When rho approaches 0, CES functions transition to logarithmic Cobb-Douglas form
- **Leontief threshold**: For very negative rho values, switches to minimum (Leontief) behavior
- **Sigmoid clamping**: Prevents exponential overflow in automation fraction calculations

### 2. Parameter Clipping & Validation

Bounds and limits applied during parameter initialization and validation.

```python
# Parameter validation bounds
RHO_CLIP_MIN = -50.0                  # Minimum elasticity parameter value
PARAM_CLIP_MIN = 1e-6                 # Minimum value for share parameters
AUTOMATION_SLOPE_CLIP_MIN = 0.1       # Minimum automation sigmoid slope
AUTOMATION_SLOPE_CLIP_MAX = 10.0      # Maximum automation sigmoid slope
RESEARCH_STOCK_START_MIN = 1e-10      # Minimum initial research stock
NORMALIZATION_MIN = 1e-10             # Minimum normalization constant
```

**Purpose**: These limits ensure parameters remain within mathematically and economically meaningful ranges:
- **Elasticity bounds**: Prevent extreme substitution elasticities that cause numerical instability
- **Share bounds**: Ensure production function shares are positive and meaningful
- **Automation bounds**: Keep sigmoid curves smooth and realistic
- **Stock bounds**: Prevent degenerate initial conditions

### 3. Model Rate & Value Caps

Upper limits on key model variables to prevent runaway growth and numerical overflow.

```python
# Maximum allowed values for key variables
MAX_RESEARCH_STOCK_RATE = 10000000000.0    # 10 billion - max research stock growth rate
MAX_NORMALIZED_PROGRESS_RATE = 100.0       # Maximum overall progress rate
TIME_EXTRAPOLATION_WINDOW = 10.0           # Years - time series extrapolation limit
```

**Purpose**: These caps prevent the model from producing unrealistic or numerically unstable results:
- **Research stock cap**: Limits knowledge accumulation rate to realistic bounds
- **Progress rate cap**: Prevents explosive progress growth that breaks model assumptions
- **Time extrapolation**: Limits how far beyond input data the model can extrapolate

### 4. ODE Integration Configuration

Settings controlling the numerical integration of differential equations.

```python
# Integration stability and performance
PROGRESS_ODE_CLAMP_MAX = 1e6          # Maximum cumulative progress during integration
RESEARCH_STOCK_ODE_CLAMP_MAX = 1e10   # Maximum research stock during integration
ODE_MAX_STEP = 1.0                    # Maximum integration step size (years)
EULER_FALLBACK_MIN_STEPS = 100        # Minimum steps for Euler fallback method
EULER_FALLBACK_STEPS_PER_YEAR = 10    # Euler steps per year (adaptive)
DENSE_OUTPUT_POINTS = 100             # Number of output points for trajectory

# Integration diagnostics
ODE_STEP_SIZE_LOGGING = False         # Enable detailed step size logging
ODE_SMALL_STEP_THRESHOLD = 1e-6       # Threshold for small step warnings
ODE_STEP_VARIATION_THRESHOLD = 100.0  # Threshold for step variation warnings
```

**Purpose**: Controls integration accuracy, stability, and performance:
- **Clamping limits**: Prevent state variables from reaching unrealistic values during integration
- **Step size control**: Balances accuracy with computational efficiency
- **Fallback settings**: Configure Euler method when advanced solvers fail
- **Diagnostic settings**: Enable detailed monitoring of integration performance

### 5. Parameter Estimation & Optimization

Configuration for the parameter optimization system.

```python
# Error handling and bounds
RELATIVE_ERROR_CLIP = 100.0           # Maximum relative error in constraint evaluation

# Parameter bounds for optimization
PARAMETER_BOUNDS = {
    'rho_cognitive': (-1, 0),                           # Cognitive substitution elasticity
    'rho_progress': (-1, 1),                            # Progress substitution elasticity  
    'alpha': (0.05, 0.95),                             # Experiment compute weight
    'software_progress_share': (0.05, 0.95),           # Software vs training weight
    'automation_fraction_at_superhuman_coder': (0.1, 0.99),  # Max automation level
    'progress_at_half_sc_automation': (1.0, 500),      # Automation midpoint
    'automation_slope': (0.1, 10.0),                   # Automation sigmoid steepness
    'cognitive_output_normalization': (0.00001, 10)    # Cognitive output scaling
}
```

**Economic Interpretation**:
- **Cognitive elasticity (-1, 0)**: Negative values indicate AI and humans are complements, not substitutes
- **Progress elasticity (-1, 1)**: Allows both complementary and substitutable relationships
- **Share parameters (0.05, 0.95)**: Avoids degenerate cases while allowing strong preferences
- **Automation parameters**: Realistic ranges based on AI development scenarios

### 6. Parameter Validation Thresholds

Advanced validation rules for parameter combinations and feasibility.

```python
# Parameter combination validation
PARAM_VALIDATION_THRESHOLDS = {
    'automation_fraction_superhuman_coder_min': 0.05,   # Minimum meaningful automation
    'automation_fraction_superhuman_coder_max': 1,      # Maximum possible automation
    'progress_at_half_automation_min': 0.0,             # Must be positive
    'automation_slope_min': 0.0,                        # Must be positive
    'automation_slope_max': 20.0,                       # Realistic steepness limit
    'rho_extreme_abs': 0.8,                            # Warning threshold for extreme rho
    'rho_product_max': 0.5,                            # Limit on rho parameter product
    'cognitive_output_normalization_max': 10            # Reasonable normalization bound
}

# Constraint feasibility checking
FEASIBILITY_CHECK_THRESHOLDS = {
    'progress_rate_target_max': 1000.0,                 # Maximum reasonable progress rate target
}
```

**Purpose**: Implements sophisticated validation beyond simple parameter bounds:
- **Cross-parameter rules**: Ensures parameter combinations make economic sense
- **Physical constraints**: Prevents combinations that violate model assumptions
- **Feasibility bounds**: Validates that optimization targets are achievable

### 7. Optimization Algorithm Configuration

Settings controlling the numerical optimization process.

```python
# Objective function weights and penalties
OBJECTIVE_FUNCTION_CONFIG = {
    'high_penalty': 1e6,                                # Penalty for invalid parameters
    'elasticity_regularization_weight': 0.001,         # Quartic penalty on extreme rho
    'boundary_avoidance_regularization_weight': 0.001, # Quadratic penalty near bounds
    'boundary_avoidance_threshold': 0.35,              # Distance from bounds for penalty
}

# Optimization termination criteria
OPTIMIZATION_CONFIG = {
    'early_termination_fun_threshold_excellent': 1e-6, # Excellent fit threshold
    'early_termination_fun_threshold_good': 1e-3,      # Good fit threshold
}
```

**Purpose**: Fine-tunes the optimization process for better convergence and stability:
- **Regularization**: Discourages numerically unstable parameter regions
- **Boundary avoidance**: Prevents parameters from clustering at bounds
- **Early termination**: Stops optimization when excellent fits are found

### 8. Strategic Starting Points Configuration

Settings for generating diverse initial parameter guesses during optimization.

```python
STRATEGIC_STARTING_POINTS_CONFIG = {
    'extreme_factor_min': 0.1,                          # Lower bound factor for extremes
    'extreme_factor_max': 0.9,                          # Upper bound factor for extremes
    'lhs_points': 5,                                    # Latin hypercube sampling points
    'high_progress_rate_threshold': 2.0,                # Threshold for high target rates
    'rho_adjustment_factor': 0.8,                       # Conservative rho for high targets
    'high_automation_threshold': 0.5,                   # Threshold for high automation
    'progress_at_half_automation_adjustment_factor': 0.7, # Adjustment for high automation
    'perturbed_points': 3,                              # Number of perturbed variations
    'critical_param_perturbation_factor': 0.1,         # Perturbation for critical params
    'other_param_perturbation_factor': 0.2,            # Perturbation for other params
}
```

**Purpose**: Configures the sophisticated initialization system that improves optimization robustness:
- **Boundary exploration**: Tests parameters near bounds to find global optima
- **Latin hypercube sampling**: Ensures diverse coverage of parameter space
- **Adaptive strategies**: Adjusts initialization based on constraint characteristics
- **Perturbation control**: Creates variations around promising parameter sets

### 9. Default Model Parameters

Baseline parameter values that provide stable, reasonable model behavior.

```python
DEFAULT_PARAMETERS = {
    'rho_cognitive': -0.2,                              # Mild complementarity
    'rho_progress': -0.1,                               # Slight complementarity
    'alpha': 0.5,                                       # Equal compute/cognitive weight
    'software_progress_share': 0.5,                     # Equal software/training weight
    'automation_fraction_at_superhuman_coder': 0.99,    # Near-complete automation
    'progress_at_half_sc_automation': 18.0,             # Moderate progress threshold
    'automation_slope': 1.6,                            # Smooth sigmoid transition
    'progress_rate_normalization': 1.0,                 # Normalized initial rate
    'cognitive_output_normalization': 1,                # Unit normalization
}
```

**Rationale**: These defaults are chosen to:
- **Ensure stability**: All combinations tested for numerical robustness
- **Reflect economic theory**: Parameters have meaningful economic interpretations
- **Enable exploration**: Provide good starting points for parameter estimation
- **Match intuition**: Results align with reasonable AI development scenarios

## Usage Guidelines

### For Model Users

**Using Default Settings**: The default configuration provides stable behavior for most use cases. Simply instantiate the model without custom parameters:

```python
from progress_model import Parameters, ProgressModel
params = Parameters()  # Uses defaults from model_config.py
```

**Customizing Parameters**: Override specific parameters while maintaining safe defaults:

```python
params = Parameters(
    rho_cognitive=-0.5,  # More complementarity
    automation_slope=2.0  # Steeper automation curve
)
```

### For Advanced Users

**Parameter Sensitivity Analysis**: Use the bounds in `PARAMETER_BOUNDS` to understand valid ranges:

```python
from model_config import PARAMETER_BOUNDS
print(f"Valid rho_cognitive range: {PARAMETER_BOUNDS['rho_cognitive']}")
```

**Optimization Constraints**: Configure constraint feasibility using validation thresholds:

```python
from model_config import FEASIBILITY_CHECK_THRESHOLDS
max_target = FEASIBILITY_CHECK_THRESHOLDS['progress_rate_target_max']
```

### For Developers

**Extending Configuration**: Add new parameters to the appropriate section:

```python
# Add to model_config.py
NEW_FEATURE_CONFIG = {
    'new_parameter_threshold': 0.5,
    'new_stability_limit': 1000.0,
}
```

**Modifying Defaults**: Update `DEFAULT_PARAMETERS` after thorough testing:

```python
# Test new defaults across parameter space
# Validate numerical stability
# Update documentation accordingly
```

## Configuration Best Practices

### Numerical Stability

1. **Conservative bounds**: Prefer slightly tighter bounds over numerical instability
2. **Gradual changes**: Make small adjustments and test thoroughly
3. **Cross-validation**: Check parameter combinations, not just individual values
4. **Edge case testing**: Validate behavior at parameter boundaries

### Performance Optimization

1. **Profile first**: Use diagnostic settings to identify bottlenecks
2. **Balance accuracy vs speed**: Looser tolerances for interactive use, strict for research
3. **Memory considerations**: Monitor caps on large values during long runs
4. **Early termination**: Use appropriate thresholds to avoid unnecessary computation

### Extensibility

1. **Modular design**: Add new sections rather than modifying existing ones
2. **Backward compatibility**: Maintain support for existing parameter sets
3. **Documentation**: Update this document when adding new configuration options
4. **Version control**: Track configuration changes for reproducibility

## Common Configuration Scenarios

### Research Applications
- **Tighter bounds**: Reduce parameter ranges for focused studies
- **Enhanced logging**: Enable ODE step size logging for diagnostics
- **Strict validation**: Lower feasibility thresholds for conservative estimates

### Interactive Web Applications
- **Looser tolerances**: Faster computation for real-time updates
- **Graceful degradation**: Higher penalty weights for stability
- **User-friendly bounds**: Narrower ranges to avoid problematic regions

### Production Deployments
- **Robust defaults**: Conservative parameters that rarely fail
- **Performance optimization**: Tuned integration settings for efficiency
- **Error handling**: Higher penalties and stricter validation

## Troubleshooting Configuration Issues

### Numerical Instability
- **Check parameter bounds**: Ensure values are within `PARAMETER_BOUNDS`
- **Review validation**: Use `PARAM_VALIDATION_THRESHOLDS` for guidance
- **Adjust caps**: Increase relevant limits in rate and value caps section

### Poor Optimization Performance
- **Tune regularization**: Adjust weights in `OBJECTIVE_FUNCTION_CONFIG`
- **Modify starting points**: Configure `STRATEGIC_STARTING_POINTS_CONFIG`
- **Check feasibility**: Review constraints against `FEASIBILITY_CHECK_THRESHOLDS`

### Integration Failures
- **Adjust ODE settings**: Modify integration configuration parameters
- **Check clamping**: Review state variable bounds
- **Enable logging**: Use diagnostic settings to identify issues

The configuration system provides fine-grained control over model behavior while maintaining stability and usability. When in doubt, start with default settings and make incremental adjustments based on specific requirements. 