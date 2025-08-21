# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a Python script for modeling AI progress over time using nested CES (Constant Elasticity of Substitution) production functions. The system models a feedback loop where automation fraction depends on cumulative progress, which affects the progress rate.

## Core Architecture

The model consists of several key components:

1. **Data Structures**:
   - `TimeSeriesData`: Input time series (time, L_HUMAN, L_AI, experiment_compute, training_compute)
   - `Parameters`: Model parameters (elasticities, weights, automation anchors)
   - `AnchorConstraint`: Constraints for parameter estimation

2. **Production Functions**:
   - `compute_coding_labor()`: CES combination of AI and human labor
   - `compute_software_progress_rate()`: CES combination of compute and cognitive work
   - `compute_overall_progress_rate()`: Weighted average of software and training progress

3. **Core Model Logic**:
   - `compute_automation_fraction()`: Log-space interpolation between anchor points
   - `progress_rate_at_time()`: Computes instantaneous progress rate
   - `integrate_progress()`: Solves the differential equation for progress over time
   - `ProgressModel`: Main class orchestrating the modeling

4. **Visualization System** (Added):
   - `visualization.py`: Modular visualization module with `ProgressVisualizer` class
   - `PlotConfig`: Configuration for consistent styling and layout
   - Comprehensive dashboard with multiple plot types
   - Parameter sensitivity analysis capabilities
   - Clean integration with existing `ProgressModel.plot_results()` method

## Development Setup

This is a Python project requiring:
- `numpy` for numerical computations
- `scipy` for optimization and differential equation solving
- `matplotlib` for visualization
- `dataclasses` for structured data (Python 3.7+)

## Key Implementation Notes

1. **Numerical Stability**: CES functions require careful handling of edge cases for extreme rho values
2. **Interpolation**: Uses `scipy.interpolate.interp1d` for time series data
3. **Differential Equations**: Uses `scipy.integrate.solve_ivp` with adaptive methods
4. **Optimization**: Parameter estimation via `scipy.optimize.minimize` with bounds
5. **Visualization**: Modular system with configurable styling, supports both simple and comprehensive dashboards

## Main Workflow

The typical workflow follows SPEC.md:
1. Load time series data from CSV
2. Define anchor constraints for parameter estimation
3. Estimate model parameters using optimization
4. Run the ProgressModel to compute trajectories
5. Visualize and export results using built-in visualization system

## Visualization Usage

The visualization system provides several ways to create plots:

### Simple Usage:
```python
# Basic two-panel plot
model.plot_results()

# Comprehensive dashboard
model.plot_results(comprehensive=True, save_path="dashboard.png")
```

### Advanced Usage:
```python
from visualization import create_default_visualizer, quick_plot_results

# Custom visualization
visualizer = create_default_visualizer()
fig = visualizer.plot_comprehensive_dashboard(results, time_series_data)

# Parameter sensitivity analysis
fig = visualizer.plot_parameter_sensitivity(param_name, param_values, trajectories, times)
```

### Available Plot Types:
- Progress trajectory over time
- Automation fraction with percentage formatting
- Progress rate analysis
- Component contributions (software vs training)
- Labor contributions (AI vs human with automation overlay)
- Parameter sensitivity analysis
- Comprehensive dashboard combining all plots

The visualization module is located in `visualization.py` and integrates cleanly with the existing `ProgressModel` class in `progress_model.py`.

## Environment Guidelines

- Always use the virtual environment.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.