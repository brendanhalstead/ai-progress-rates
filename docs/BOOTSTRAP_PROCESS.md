# Superhuman Coder Bootstrap Process

## Overview

The AI progress model uses a sophisticated two-step bootstrap process to dynamically calculate the progress level at which AI reaches superhuman coding ability. This process replaces hardcoded configuration values with empirically-derived estimates based on METR benchmark data.

## The Problem

Originally, the model used a hardcoded `progress_at_sc` parameter from `model_config.py` (default: 20.0). However, this approach had limitations:

1. **Arbitrary Values**: The hardcoded value was not grounded in empirical data
2. **Inconsistency**: The value didn't correspond to actual benchmark performance data
3. **Parameter Coupling**: Different `sc_time_horizon_minutes` values should correspond to different progress levels

## The Solution: Two-Step Bootstrap Process

The bootstrap process dynamically calculates the appropriate `progress_at_sc` value based on:
- METR benchmark data (horizon lengths vs model performance)
- The specified `sc_time_horizon_minutes` parameter
- A fitted trajectory between progress and task horizon lengths

### Step 1: Horizon Trajectory Estimation (`estimate_horizon_trajectory`)

This step creates a "backcast" model to establish the relationship between progress and task completion horizons.

#### Process:
1. **Create Backcast Parameters**: Clone the model parameters but zero out AI contributions
   ```python
   backcast_params = copy.deepcopy(self.params)
   backcast_params.progress_at_half_automation_fraction = 100  # Effectively no automation
   backcast_params.ai_research_taste_at_superhuman_coder = 0   # No AI research taste
   backcast_params.progress_at_sc = 100.0                      # Irrelevant for backcast
   ```

2. **Run Backcast Model**: Compute progress trajectory with zero AI contributions
   ```python
   backcast_model = ProgressModel(backcast_params, self.data)
   backcast_times, backcast_progress, _ = backcast_model.compute_progress_trajectory(time_range, initial_progress)
   ```

3. **Load METR Data**: Read benchmark results from `benchmark_results.yaml`

4. **Match Progress to Benchmarks**: For each model in METR data:
   - Convert release date to decimal year
   - Interpolate progress value at release date using backcast results
   - Extract p80_horizon_length estimates

5. **Fit Regression**: Fit linear regression: `log(horizon) = slope * progress + intercept`
   ```python
   popt, pcov = optimize.curve_fit(linear_func, progress_values, log_horizon_values)
   slope, intercept = popt
   ```

6. **Calculate SC Progress**: Solve for progress level where horizon equals `sc_time_horizon_minutes`
   ```python
   self.sc_progress = (np.log(self.params.sc_time_horizon_minutes) - intercept) / slope
   ```

#### Key Output:
- `self.sc_progress`: The calculated progress level for superhuman coding
- `self.horizon_trajectory`: Function mapping progress → horizon length

### Step 2: Full Model Computation (`compute_progress_trajectory`)

This step runs the complete model using the calculated `sc_progress` value.

#### Process:
1. **Parameter Substitution**: Replace hardcoded `progress_at_sc` with calculated value
   ```python
   params_to_use = self.params
   if hasattr(self, 'sc_progress') and self.sc_progress is not None:
       params_to_use = copy.deepcopy(self.params)
       params_to_use.progress_at_sc = self.sc_progress
   ```

2. **Run Full Model**: Compute trajectory with AI contributions using calculated SC progress

3. **Calculate SC Timing**: Determine when progress reaches the SC level
   ```python
   if progress_values[-1] >= sc_progress_target:
       sc_time = np.interp(sc_progress_target, progress_values, times)
   ```

## Implementation Details

### Key Functions

#### `estimate_horizon_trajectory(self, time_range, initial_progress=None)`
**Location**: `progress_model.py` (lines ~2025-2145)
**Purpose**: Fits horizon trajectory and calculates `sc_progress`
**Key Operations**:
- Creates backcast model with zero AI contributions
- Loads and processes METR benchmark data
- Fits linear regression between progress and log(horizon)
- Calculates and stores `self.sc_progress`

#### `compute_progress_trajectory(self, time_range, initial_progress=None)`
**Location**: `progress_model.py` (lines ~2145-2370)
**Purpose**: Computes full model trajectory using calculated SC progress
**Key Operations**:
- Substitutes calculated `sc_progress` for hardcoded `progress_at_sc`
- Runs complete model with AI contributions
- Calculates timing when SC level is reached
- Stores comprehensive results including SC timing

### Parameter Flow

```
model_config.py (hardcoded values)
        ↓
Parameters.__init__() 
        ↓
estimate_horizon_trajectory()
   ├── Creates backcast with no AI
   ├── Fits horizon regression
   └── Calculates self.sc_progress
        ↓
compute_progress_trajectory()
   ├── Uses self.sc_progress if available
   ├── Otherwise falls back to config value
   └── Computes full trajectory
```

### Data Dependencies

#### Required Files:
- `benchmark_results.yaml`: METR benchmark data with model performance metrics
- `input_data.csv`: Time series data for model inputs

#### METR Data Structure:
```yaml
results:
  model_name:
    release_date: "YYYY-MM-DD"
    agents:
      agent_config:
        p80_horizon_length:
          estimate: float
        is_sota: boolean
```

## Usage Examples

### Basic Usage
```python
from progress_model import ProgressModel, Parameters, load_time_series_data

# Create model
params = Parameters()
params.sc_time_horizon_minutes = 50000.0  # 50k minutes for SC
data = load_time_series_data('input_data.csv')
model = ProgressModel(params, data)

# Step 1: Estimate horizon trajectory
time_range = [2019, 2035]
horizon_func = model.estimate_horizon_trajectory(time_range, 1.0)
print(f"Calculated SC progress: {model.sc_progress:.3f}")

# Step 2: Compute full trajectory  
times, progress, research_stock = model.compute_progress_trajectory(time_range, 1.0)

# Access SC timing
if model.results.get('sc_time'):
    print(f"SC reached at time: {model.results['sc_time']:.3f}")
```

### Parameter Comparison
```python
# Original hardcoded value
print(f"Config progress_at_sc: {params.progress_at_sc}")  # e.g., 20.0

# Calculated value (after bootstrap)
print(f"Calculated sc_progress: {model.sc_progress}")     # e.g., 23.64
```

## API Integration

The bootstrap process is integrated into the web API:

### Endpoints Affected:
- `/api/compute`: Automatically runs bootstrap process
- `/api/estimate-parameters`: Runs bootstrap during parameter estimation

### Response Enhancement:
```json
{
  "summary": {
    "sc_time": 2031.005,           // When SC level is reached
    "sc_progress_level": 23.643    // Calculated progress level for SC
  }
}
```

## Error Handling

### Common Scenarios:
1. **Missing METR Data**: Falls back to config values with warning
2. **Insufficient Data Points**: Returns None, uses config values
3. **SC Not Reached**: `sc_time` set to None, logged appropriately
4. **Regression Failure**: Returns None, falls back gracefully

### Logging:
```python
logger.info(f"Fitted horizon trajectory: log(horizon) = {slope:.6f} * progress + {intercept:.6f}")
logger.info(f"Progress level at sc_time_horizon_minutes ({minutes} min): {sc_progress:.4f}")
logger.info(f"Superhuman Coder level ({sc_progress:.3f}) reached at time {sc_time:.3f}")
```

## Technical Considerations

### Why Two Steps?
1. **Chicken-and-Egg Problem**: Need progress values to fit horizon trajectory, but need SC progress to compute trajectory
2. **Empirical Grounding**: Backcast provides progress values corresponding to actual model release dates
3. **Parameter Consistency**: Ensures SC progress aligns with specified time horizon

### Numerical Stability:
- Uses log-linear regression to handle wide range of horizon values
- Includes bounds checking and fallback mechanisms
- Handles edge cases (SC not reached, insufficient data)

### Performance:
- Backcast model runs quickly (no AI complexity)
- Regression fitting is computationally lightweight
- Full model runs once with correct parameters

## Debugging Tips

### Check Bootstrap Success:
```python
# Verify horizon trajectory was fitted
assert hasattr(model, 'horizon_trajectory')
assert model.horizon_trajectory is not None

# Verify SC progress was calculated  
assert hasattr(model, 'sc_progress')
assert model.sc_progress is not None

# Check if SC is reached in trajectory
if 'sc_time' in model.results:
    print(f"SC reached: {model.results['sc_time']}")
else:
    print("SC not reached in trajectory")
```

### Common Issues:
1. **METR File Missing**: Check `benchmark_results.yaml` exists
2. **No Valid Benchmarks**: Verify METR data has p80_horizon_length values
3. **Time Range Too Short**: SC might not be reached within trajectory
4. **Parameter Conflicts**: Check that time horizon makes sense for progress scale

## Future Enhancements

### Potential Improvements:
1. **Multiple Horizons**: Support different task types (coding, reasoning, etc.)
2. **Uncertainty Quantification**: Include confidence intervals from regression
3. **Dynamic Updates**: Refresh calculations with new benchmark data
4. **Validation Metrics**: Cross-validation of horizon predictions

### Extension Points:
1. **Custom Horizon Functions**: Replace linear regression with more sophisticated models
2. **Multi-Modal Fitting**: Separate trajectories for different capability domains
3. **Temporal Dynamics**: Account for changing relationships over time