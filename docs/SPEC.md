
# Specification for Progress Modeling Script

## 1. Overview
Create a Python script that models AI progress over time using nested CES production functions. The system has a feedback loop where automation fraction depends on cumulative progress, which affects the progress rate.

## 2. Core Data Structures

```python
@dataclass
class TimeSeriesData:
    """Input time series data"""
    time: np.array  # Decimal years
    L_HUMAN: np.array  # Human labor supply
    L_AI: np.array  # AI labor supply (human-equivalents)
    experiment_compute: np.array  # Experiment compute budget
    training_compute: np.array  # Training compute budget

@dataclass
class Parameters:
    """Model parameters"""
    # Production function parameters
    rho_cognitive: float  # Elasticity for cognitive output
    rho_progress: float   # Elasticity for software progress
    alpha: float          # Weight on experiment compute [0,1]
    software_progress_share: float  # Weight on software progress [0,1]
    
    # Automation mapping anchors
    progress_at_2025: float
    automation_fraction_at_2025: float
    progress_at_superhuman_coder: float
    automation_fraction_at_superhuman_coder: float
    
    # Normalization
    progress_rate_normalization: float = 1.0  # Auto-calculated to ensure initial progress rate = 1
    cognitive_output_normalization: float = 1.0

@dataclass
class AnchorConstraint:
    """Specifies a constraint for parameter estimation"""
    # Dict mapping variable names to values (can be partial)
    conditions: dict  # e.g., {"automation_fraction": 0.9, "L_AI": 1e6}
    # Expected outcome
    target_variable: str  # e.g., "progress_rate"
    target_value: float   # e.g., 5.0
    weight: float = 1.0   # Weight in optimization
```

## 3. Core Functions to Implement

### 3.1 Production Functions
```python
def compute_cognitive_output(automation_fraction, L_AI, L_HUMAN, rho, cognitive_normalization=1.0):
    """CES combination of AI and human labor"""
    # Handle edge cases (automation_fraction = 0 or 1)
    # Implement the nested CES formula
    # Apply cognitive_normalization constant to the output
    
def compute_software_progress_rate(experiment_compute, cognitive_output, alpha, rho):
    """CES combination of compute and cognitive work"""
    
def compute_overall_progress_rate(software_progress_rate, training_compute, software_share):
    """Weighted average of software and training progress"""
```

### 3.2 Automation Fraction Mapping
```python
def compute_automation_fraction(cumulative_progress, params: Parameters):
    """
    Log-space interpolation between anchor points.
    Returns value in [0, 1].
    """
    # Extract anchor points from params
    # Perform log-space interpolation
    # Clamp to [0, 1]
```

### 3.3 Progress Integration
```python
def progress_rate_at_time(t, cumulative_progress, time_series_data, params):
    """
    Compute instantaneous progress rate given cumulative progress.
    This is the RHS of the differential equation.
    """
    # Interpolate time series data to time t
    # Compute automation fraction from cumulative progress
    # Compute cognitive output, software progress rate, overall rate
    # Return normalized rate
    
def integrate_progress(time_range, initial_progress, time_series_data, params, 
                      direction='forward'):
    """
    Solve the differential equation:
    d(progress)/dt = progress_rate(t, progress)
    
    Returns: times, cumulative_progress_values
    """
    # Use scipy.integrate.solve_ivp or similar
    # Handle forward/backward integration
```

### 3.4 Anchor Constraint Evaluation
```python
def evaluate_anchor_constraint(constraint: AnchorConstraint, 
                             time_series_data: TimeSeriesData,
                             params: Parameters):
    """
    Evaluate model at specified conditions and compare to target.
    Returns the error (difference from target).
    """
    # Override relevant values with constraint.conditions
    # Compute the target variable
    # Return error
```

### 3.5 Parameter Estimation
```python
def estimate_parameters(anchor_constraints: List[AnchorConstraint],
                       time_series_data: TimeSeriesData,
                       initial_params: Parameters,
                       fixed_params: List[str] = None):
    """
    Find parameters that best satisfy anchor constraints.
    """
    # Set up optimization problem
    # Define objective function (sum of squared errors from anchors)
    # Use scipy.optimize with bounds
    # Return optimized parameters
```

## 4. Main Script Structure

```python
class ProgressModel:
    def __init__(self, params: Parameters, time_series_data: TimeSeriesData):
        self.params = params
        self.data = time_series_data
        
    def compute_progress_trajectory(self, time_range, initial_progress=None):
        """Compute progress over specified time range"""
        
    def plot_results(self):
        """Visualize progress, automation fraction, and component contributions"""
        
    def export_results(self, filename):
        """Export trajectories and parameters"""

# Example usage:
if __name__ == "__main__":
    # Load time series data
    data = load_time_series_data("input_data.csv")
    
    # Define anchor constraints
    anchors = [
        AnchorConstraint(
            conditions={"automation_fraction": 0.9, "L_AI": 1e9},
            target_variable="progress_rate",
            target_value=5.0
        ),
        # More anchors...
    ]
    
    # Initial parameter guess
    initial_params = Parameters(
        rho_cognitive=0.5,
        rho_progress=0.5,
        alpha=0.5,
        cognitive_output_normalization=1.0,
        # etc...
    )
    
    # Estimate parameters
    params = estimate_parameters(anchors, data, initial_params)
    
    # Run model
    model = ProgressModel(params, data)
    times, progress = model.compute_progress_trajectory([2019, 2030])
    model.plot_results()
```

## 5. Implementation Notes

1. **Numerical Stability**: CES functions can be numerically unstable for extreme rho values. Implement careful handling of edge cases.

2. **Interpolation**: Use `scipy.interpolate.interp1d` for time series data interpolation.

3. **Differential Equation**: Use `scipy.integrate.solve_ivp` with an adaptive method like 'RK45'.

4. **Optimization**: Start with `scipy.optimize.minimize` with bounds. Consider using `differential_evolution` for global optimization if local minima are an issue.

5. **Logging**: Include verbose logging to track progress rate components and automation fraction evolution.

6. **Cognitive Output Normalization**: The `cognitive_output_normalization` parameter allows scaling the cognitive output to match empirical observations or calibrate the model to specific scenarios. This normalization is applied after the CES combination of AI and human labor.

