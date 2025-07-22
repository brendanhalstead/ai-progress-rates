#!/usr/bin/env python3
"""
Test script for visualization functionality
"""

import numpy as np
import sys
import logging
from progress_model import (
    ProgressModel, Parameters, TimeSeriesData, 
    load_time_series_data, AnchorConstraint, estimate_parameters
)
from visualization import create_default_visualizer, quick_plot_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_visualization():
    """Test basic visualization functionality"""
    logger.info("Testing basic visualization...")
    
    # Create sample data
    time = np.linspace(2019, 2030, 12)
    L_HUMAN = np.ones_like(time) * 1e6
    L_AI = np.logspace(3, 8, len(time))
    experiment_compute = np.logspace(6, 10, len(time))
    training_compute = np.logspace(6, 10, len(time))
    
    data = TimeSeriesData(time, L_HUMAN, L_AI, experiment_compute, training_compute)
    
    # Simple parameters
    params = Parameters(
        rho_cognitive=0.5,
        rho_progress=0.5,
        alpha=0.5,
        software_progress_share=0.7,
        automation_fraction_at_superhuman_coder=0.9,
        progress_at_half_sc_automation=50.0,  # Progress level where automation = 45% (half of 90%)
        automation_slope=2.0,  # Moderate slope for smooth transition
        progress_rate_normalization=1.0,
        cognitive_output_normalization=1e-3
    )
    
    # Run model
    model = ProgressModel(params, data)
    times, progress = model.compute_progress_trajectory([2019, 2030], initial_progress=1.0)
    
    # Test simple plot
    logger.info("Testing simple plot...")
    fig1 = model.plot_results(comprehensive=False)
    
    # Test comprehensive dashboard
    logger.info("Testing comprehensive dashboard...")
    fig2 = model.plot_results(comprehensive=True)
    
    # Test individual visualization components
    logger.info("Testing individual components...")
    visualizer = create_default_visualizer()
    
    import matplotlib.pyplot as plt
    
    # Test individual plots
    fig3, ax = plt.subplots()
    visualizer.plot_progress_trajectory(times, progress, ax=ax)
    
    fig4, ax = plt.subplots()
    visualizer.plot_automation_fraction(times, model.results['automation_fraction'], ax=ax)
    
    logger.info("All visualization tests completed successfully!")
    
    return [fig1, fig2, fig3, fig4]


def test_parameter_sensitivity():
    """Test parameter sensitivity visualization"""
    logger.info("Testing parameter sensitivity analysis...")
    
    # Create sample data
    time = np.linspace(2019, 2030, 12)
    L_HUMAN = np.ones_like(time) * 1e6
    L_AI = np.logspace(3, 8, len(time))
    experiment_compute = np.logspace(6, 10, len(time))
    training_compute = np.logspace(6, 10, len(time))
    
    data = TimeSeriesData(time, L_HUMAN, L_AI, experiment_compute, training_compute)
    
    # Test sensitivity to rho_cognitive
    rho_values = np.linspace(-2, 2, 5)
    trajectories = []
    
    for rho in rho_values:
        params = Parameters(
            rho_cognitive=rho,
            rho_progress=0.5,
            alpha=0.5,
            software_progress_share=0.7,
            automation_fraction_at_superhuman_coder=0.9,
            progress_at_half_sc_automation=50.0,  # Progress level where automation = 45% (half of 90%)
            automation_slope=2.0,  # Moderate slope for smooth transition
            progress_rate_normalization=1.0,
            cognitive_output_normalization=1e-3
        )
        
        model = ProgressModel(params, data)
        times, progress = model.compute_progress_trajectory([2019, 2030], initial_progress=1.0)
        trajectories.append(progress)
    
    # Create sensitivity plot
    visualizer = create_default_visualizer()
    fig = visualizer.plot_parameter_sensitivity(
        'rho_cognitive', rho_values, trajectories, times
    )
    
    logger.info("Parameter sensitivity test completed!")
    return fig


def main():
    """Run all visualization tests"""
    try:
        # Test basic functionality
        basic_figs = test_basic_visualization()
        
        # Test sensitivity analysis
        sensitivity_fig = test_parameter_sensitivity()
        
        logger.info("All tests passed! Visualization module is working correctly.")
        
        # Show plots (comment out if running headless)
        import matplotlib.pyplot as plt
        plt.show()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()