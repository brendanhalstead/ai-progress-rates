#!/usr/bin/env python3
"""
Auxiliary Metrics Calculator

Calculates all auxiliary metrics needed for visualization from core model results.
This separates metric computation logic from webapp and plotting code.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from progress_model import (
    compute_automation_fraction, compute_cognitive_output,
    compute_software_progress_rate, compute_overall_progress_rate,
    compute_research_stock_rate, Parameters, TimeSeriesData
)

logger = logging.getLogger(__name__)


def calculate_auxiliary_metrics(model_results: Dict[str, Any], params: Parameters, 
                              time_series_data: TimeSeriesData, 
                              initial_research_stock_rate: float) -> Dict[str, Any]:
    """
    Calculate all auxiliary metrics needed for visualization from core model results.
    
    This function takes the basic model outputs and computes additional metrics
    that are useful for analysis and visualization but not core to the model itself.
    
    Args:
        model_results: Dict containing core model results with keys:
            - 'times': Array of time points
            - 'progress': Array of cumulative progress values  
            - 'research_stock': Array of research stock values
            - 'automation_fraction': Array of automation fraction values
            - 'progress_rates': Array of overall progress rates
            - 'research_stock_rates': Array of research stock rates
        params: Model parameters object
        time_series_data: Input time series data
        initial_research_stock_rate: Initial research stock rate for calculations
        
    Returns:
        Dict containing all auxiliary metrics:
            - 'cognitive_outputs': Cognitive output at each time
            - 'software_progress_rates': Software-specific progress rates
            - 'human_only_progress_rates': Progress rates with zero automation
            - 'ai_labor_contributions': AI contribution to cognitive output
            - 'human_labor_contributions': Human contribution to cognitive output
            - 'automation_multipliers': Ratio of overall to human-only progress rates
    """
    
    # Extract core results
    times = model_results['times']
    progress_values = model_results['progress'] 
    research_stock_values = model_results['research_stock']
    progress_rates = model_results['progress_rates']
    research_stock_rates = model_results['research_stock_rates']
    
    # Initialize auxiliary metric arrays
    cognitive_outputs = []
    software_progress_rates = []
    human_only_progress_rates = []
    ai_labor_contributions = []
    human_labor_contributions = []
    
    logger.info(f"Calculating auxiliary metrics for {len(times)} time points")
    
    # Calculate metrics at each time point
    for i, (t, p, rs) in enumerate(zip(times, progress_values, research_stock_values)):
        try:
            # Interpolate input time series to current time
            L_HUMAN = np.interp(t, time_series_data.time, time_series_data.L_HUMAN)
            L_AI = np.interp(t, time_series_data.time, time_series_data.L_AI)
            experiment_compute = np.interp(t, time_series_data.time, time_series_data.experiment_compute)
            training_compute = np.interp(t, time_series_data.time, time_series_data.training_compute)

            # Calculate automation fraction and cognitive output
            automation_fraction = compute_automation_fraction(p, params)
            cognitive_output = compute_cognitive_output(
                automation_fraction, L_AI, L_HUMAN, 
                params.rho_cognitive, params.cognitive_output_normalization
            )
            cognitive_outputs.append(cognitive_output if np.isfinite(cognitive_output) else 0.0)

            # Calculate software progress rate
            current_research_stock_rate = research_stock_rates[i]
            software_rate = compute_software_progress_rate(
                rs, current_research_stock_rate, 
                params.research_stock_at_simulation_start, 
                initial_research_stock_rate
            )
            software_progress_rates.append(software_rate if np.isfinite(software_rate) else 0.0)

            # Calculate human-only progress rate (with automation fraction = 0)
            human_only_cognitive_output = L_HUMAN * params.cognitive_output_normalization
            human_only_research_stock_rate = compute_research_stock_rate(
                experiment_compute, human_only_cognitive_output, 
                params.alpha, params.rho_progress
            )
            human_only_software_rate = compute_software_progress_rate(
                rs, human_only_research_stock_rate,
                params.research_stock_at_simulation_start,
                initial_research_stock_rate
            )
            human_only_overall_rate = compute_overall_progress_rate(
                human_only_software_rate, training_compute, params.software_progress_share
            ) * params.progress_rate_normalization
            
            human_only_progress_rates.append(
                human_only_overall_rate if np.isfinite(human_only_overall_rate) else 0.0
            )
            
            # Calculate labor contributions to cognitive output
            # This is an approximation based on the cognitive output normalization
            human_contrib = L_HUMAN * params.cognitive_output_normalization
            ai_contrib = max(0.0, cognitive_output - human_contrib)  # Ensure non-negative
            
            human_labor_contributions.append(human_contrib)
            ai_labor_contributions.append(ai_contrib)

        except Exception as e:
            logger.warning(f"Error calculating auxiliary metrics at t={t}: {e}")
            # Use safe fallback values
            cognitive_outputs.append(0.0)
            software_progress_rates.append(0.0)
            human_only_progress_rates.append(0.0)
            human_labor_contributions.append(0.0)
            ai_labor_contributions.append(0.0)
    
    # Calculate automation multipliers (overall rate / human-only rate)
    automation_multipliers = []
    for i in range(len(progress_rates)):
        if human_only_progress_rates[i] > 0:
            multiplier = progress_rates[i] / human_only_progress_rates[i]
            automation_multipliers.append(multiplier if np.isfinite(multiplier) else 1.0)
        else:
            automation_multipliers.append(1.0)  # No multiplier if human-only rate is zero

    # Package all auxiliary metrics
    auxiliary_metrics = {
        'cognitive_outputs': cognitive_outputs,
        'software_progress_rates': software_progress_rates,  
        'human_only_progress_rates': human_only_progress_rates,
        'ai_labor_contributions': ai_labor_contributions,
        'human_labor_contributions': human_labor_contributions,
        'automation_multipliers': automation_multipliers
    }
    
    logger.info("Auxiliary metrics calculation completed successfully")
    return auxiliary_metrics


def calculate_all_metrics(model_results: Dict[str, Any], params: Parameters,
                         time_series_data: TimeSeriesData, 
                         initial_research_stock_rate: float) -> Dict[str, Any]:
    """
    Calculate both core and auxiliary metrics in a single comprehensive structure.
    
    This is a convenience function that combines core model results with 
    auxiliary metrics for easy consumption by visualization code.
    
    Args:
        model_results: Core model results
        params: Model parameters
        time_series_data: Input time series
        initial_research_stock_rate: Initial research stock rate
    
    Returns:
        Dict containing both core and auxiliary metrics
    """
    
    # Start with core model results
    all_metrics = model_results.copy()
    
    # Add auxiliary metrics
    auxiliary_metrics = calculate_auxiliary_metrics(
        model_results, params, time_series_data, initial_research_stock_rate
    )
    all_metrics.update(auxiliary_metrics)
    
    # Add input time series for reference
    all_metrics['input_time_series'] = {
        'time': time_series_data.time,
        'L_HUMAN': time_series_data.L_HUMAN,
        'L_AI': time_series_data.L_AI,
        'experiment_compute': time_series_data.experiment_compute,
        'training_compute': time_series_data.training_compute
    }
    
    return all_metrics


 