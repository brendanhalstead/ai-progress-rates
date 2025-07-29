#!/usr/bin/env python3
"""
Interactive Web App for AI Progress Modeling

Flask-based web application providing interactive interface for exploring
AI progress trajectories with real-time parameter adjustment and visualization.
"""

# Set matplotlib backend before any other imports to prevent GUI errors on headless servers
import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, render_template, send_file
import json
import numpy as np
import plotly.utils
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import io
import csv
from datetime import datetime
import logging
from typing import Dict, Any

from progress_model import (
    ProgressModel, Parameters, TimeSeriesData, 
    AnchorConstraint, estimate_parameters,
    progress_rate_at_time, compute_cognitive_output,
    compute_software_progress_rate, compute_automation_fraction,
    compute_research_stock_rate, compute_overall_progress_rate
)
from metrics_calculator import calculate_all_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global storage for session data (in production, use Redis or database)
session_data = {
    'time_series': None,
    'current_params': None,
    'results': None
}

def create_default_time_series():
    """Load default time series data from input_data.csv"""
    try:
        # Try to load from input_data.csv
        import os
        csv_path = os.path.join(os.path.dirname(__file__), 'input_data.csv')
        
        if os.path.exists(csv_path):
            logger.info("Loading default time series data from input_data.csv")
            with open(csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                data = list(reader)
            
            time = np.array([float(row['time']) for row in data])
            L_HUMAN = np.array([float(row['L_HUMAN']) for row in data])
            L_AI = np.array([float(row['L_AI']) for row in data])
            experiment_compute = np.array([float(row['experiment_compute']) for row in data])
            training_compute = np.array([float(row['training_compute']) for row in data])
            
            logger.info(f"Loaded time series data: {len(data)} points from {time[0]} to {time[-1]}")
            return TimeSeriesData(time, L_HUMAN, L_AI, experiment_compute, training_compute)
        else:
            logger.warning("input_data.csv not found, falling back to synthetic data")
            raise FileNotFoundError("input_data.csv not found")
            
    except Exception as e:
        logger.warning(f"Error loading input_data.csv: {e}, falling back to synthetic data")
        # Fallback to synthetic data if CSV loading fails
        time = np.linspace(2019, 2030, 12)
        L_HUMAN = np.ones_like(time) * 1e6
        L_AI = np.logspace(3, 8, len(time))
        experiment_compute = np.logspace(6, 10, len(time))  # Use exponential growth as fallback
        training_compute = np.logspace(6, 10, len(time))
        
        logger.info("Using synthetic fallback data")
        return TimeSeriesData(time, L_HUMAN, L_AI, experiment_compute, training_compute)

def create_default_parameters():
    """Create default model parameters"""
    return Parameters()

def time_series_to_dict(data: TimeSeriesData):
    """Convert TimeSeriesData to dictionary for JSON serialization"""
    return {
        'time': data.time.tolist(),
        'L_HUMAN': data.L_HUMAN.tolist(),
        'L_AI': data.L_AI.tolist(),
        'experiment_compute': data.experiment_compute.tolist(),
        'training_compute': data.training_compute.tolist()
    }

def params_to_dict(params: Parameters):
    """Convert parameters to dictionary for JSON serialization"""
    return {
        'rho_cognitive': params.rho_cognitive,
        'rho_progress': params.rho_progress,
        'alpha': params.alpha,
        'software_progress_share': params.software_progress_share,
        'automation_fraction_at_superhuman_coder': params.automation_fraction_at_superhuman_coder,
        'progress_at_half_sc_automation': params.progress_at_half_sc_automation,
        'automation_slope': params.automation_slope,
        'research_stock_at_simulation_start': params.research_stock_at_simulation_start,
        'progress_rate_normalization': params.progress_rate_normalization,
        'cognitive_output_normalization': params.cognitive_output_normalization
    }

def calculate_progress_rate_normalization(params: Parameters, time_series_data: TimeSeriesData, 
                                         start_time: float, initial_progress: float, initial_research_stock: float) -> float:
    """
    Calculate progress rate normalization so that initial progress rate equals 1.
    
    Args:
        params: Model parameters (with temporary progress_rate_normalization=1)
        time_series_data: Input time series
        start_time: Simulation start time
        initial_progress: Initial cumulative progress
        initial_research_stock: Initial research stock
        
    Returns:
        Calculated normalization factor
    """
    # We need the initial research stock rate to compute the software progress rate
    initial_automation_at_start = compute_automation_fraction(initial_progress, params)
    initial_L_HUMAN_at_start = np.interp(start_time, time_series_data.time, time_series_data.L_HUMAN)
    initial_L_AI_at_start = np.interp(start_time, time_series_data.time, time_series_data.L_AI)
    initial_experiment_compute_at_start = np.interp(start_time, time_series_data.time, time_series_data.experiment_compute)
    initial_training_compute_at_start = np.interp(start_time, time_series_data.time, time_series_data.training_compute)

    initial_cognitive_output_at_start = compute_cognitive_output(
        initial_automation_at_start, initial_L_AI_at_start, initial_L_HUMAN_at_start,
        params.rho_cognitive, params.cognitive_output_normalization
    )
    
    initial_research_stock_rate_at_start = compute_research_stock_rate(
        initial_experiment_compute_at_start, initial_cognitive_output_at_start,
        params.alpha, params.rho_progress
    )

    if initial_research_stock_rate_at_start <= 0:
        initial_research_stock_rate_at_start = 1.0

    initial_software_progress_rate = compute_software_progress_rate(
        initial_research_stock, 
        initial_research_stock_rate_at_start,
        params.research_stock_at_simulation_start,
        initial_research_stock_rate_at_start # RS'(0) is the same as RS'(t) at start_time
    )

    unnormalized_rate = compute_overall_progress_rate(
        initial_software_progress_rate, initial_training_compute_at_start, params.software_progress_share
    )
    
    # Return the normalization factor that makes the rate equal to 1
    if unnormalized_rate > 0:
        return 1.0 / unnormalized_rate
    else:
        logger.warning("Unnormalized progress rate is zero or negative, using default normalization")
        return 1.0

def create_plotly_dashboard(metrics: Dict[str, Any]):
    """Create interactive Plotly dashboard"""
    
    # Extract required data
    times = np.array(metrics['times'], dtype=float)
    progress = np.array(metrics['progress'], dtype=float)
    automation_fraction = np.array(metrics['automation_fraction'], dtype=float)
    
    # Validate input data
    if len(times) == 0 or len(progress) == 0 or len(automation_fraction) == 0:
        raise ValueError("Empty input data")
    
    # Handle invalid values
    valid_mask = np.isfinite(times) & np.isfinite(progress) & np.isfinite(automation_fraction)
    times = times[valid_mask]
    progress = progress[valid_mask]
    automation_fraction = automation_fraction[valid_mask]
    
    # Extract optional metrics and apply validation
    progress_rates = None
    if 'progress_rates' in metrics and metrics['progress_rates'] is not None and len(metrics['progress_rates']) > 0:
        progress_rates = np.array(metrics['progress_rates'], dtype=float)[valid_mask]
        progress_rates = np.where(np.isfinite(progress_rates), progress_rates, 0)
    
    software_progress_rates = None
    if 'software_progress_rates' in metrics and metrics['software_progress_rates'] is not None and len(metrics['software_progress_rates']) > 0:
        software_progress_rates = np.array(metrics['software_progress_rates'], dtype=float)[valid_mask]
        software_progress_rates = np.where(np.isfinite(software_progress_rates), software_progress_rates, 0)
    
    cognitive_outputs = None
    if 'cognitive_outputs' in metrics and metrics['cognitive_outputs'] is not None and len(metrics['cognitive_outputs']) > 0:
        cognitive_outputs = np.array(metrics['cognitive_outputs'], dtype=float)[valid_mask]
        cognitive_outputs = np.where(np.isfinite(cognitive_outputs), cognitive_outputs, 0)

    research_stocks = None
    if 'research_stock' in metrics and metrics['research_stock'] is not None and len(metrics['research_stock']) > 0:
        research_stocks = np.array(metrics['research_stock'], dtype=float)[valid_mask]
        research_stocks = np.where(np.isfinite(research_stocks), research_stocks, 0)

    research_stock_rates = None
    if 'research_stock_rates' in metrics and metrics['research_stock_rates'] is not None and len(metrics['research_stock_rates']) > 0:
        research_stock_rates = np.array(metrics['research_stock_rates'], dtype=float)[valid_mask]
        research_stock_rates = np.where(np.isfinite(research_stock_rates), research_stock_rates, 0)

    human_only_progress_rates = None
    if 'human_only_progress_rates' in metrics and metrics['human_only_progress_rates'] is not None and len(metrics['human_only_progress_rates']) > 0:
        human_only_progress_rates = np.array(metrics['human_only_progress_rates'], dtype=float)[valid_mask]
        human_only_progress_rates = np.where(np.isfinite(human_only_progress_rates), human_only_progress_rates, 0)

    ai_labor_contributions = None
    if 'ai_labor_contributions' in metrics and metrics['ai_labor_contributions'] is not None and len(metrics['ai_labor_contributions']) > 0:
        ai_labor_contributions = np.array(metrics['ai_labor_contributions'], dtype=float)[valid_mask]
        ai_labor_contributions = np.where(np.isfinite(ai_labor_contributions), ai_labor_contributions, 0)

    human_labor_contributions = None
    if 'human_labor_contributions' in metrics and metrics['human_labor_contributions'] is not None and len(metrics['human_labor_contributions']) > 0:
        human_labor_contributions = np.array(metrics['human_labor_contributions'], dtype=float)[valid_mask]
        human_labor_contributions = np.where(np.isfinite(human_labor_contributions), human_labor_contributions, 0)
    
    # Create subplots - expand to 7x2 layout for input time series and human-only rates
    fig = make_subplots(
        rows=7, cols=2,
        subplot_titles=('Cumulative Progress', 'Automation Fraction', 
                       'Overall Progress Rate', 'Software Progress Rate',
                       'Cognitive Output & Experiment Compute', 'Progress vs Automation',
                       'Rate Components', 'Cognitive Output Components',
                       'Human vs AI Labor', 'Training Compute',
                       'Research Stock', 'Research Stock Rate',
                       'Human-Only Progress Rate', 'Automation Progress Multiplier'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Cumulative Progress
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=progress.tolist(), 
                  name='Cumulative Progress',
                  line=dict(color='#1f77b4', width=3),
                  mode='lines+markers', marker=dict(size=4)),
        row=1, col=1
    )
    
    # Plot 2: Automation Fraction
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=(automation_fraction*100).tolist(), 
                  name='Automation %', line=dict(color='#ff7f0e', width=3),
                  mode='lines+markers', marker=dict(size=4)),
        row=1, col=2
    )
    
    # Plot 3: Overall Progress Rate (if available)
    if progress_rates is not None and len(progress_rates) > 0:
        fig.add_trace(
            go.Scatter(x=times.tolist(), y=progress_rates.tolist(), 
                      name='Overall Progress Rate',
                      line=dict(color='#2ca02c', width=3),
                      mode='lines+markers', marker=dict(size=4)),
            row=2, col=1
        )
    
    # Plot 4: Software Progress Rate (if available)
    if software_progress_rates is not None and len(software_progress_rates) > 0:
        fig.add_trace(
            go.Scatter(x=times.tolist(), y=software_progress_rates.tolist(),
                      name='Software Progress Rate',
                      line=dict(color='#ff7f0e', width=3),
                      mode='lines+markers', marker=dict(size=4)),
            row=2, col=2
        )
    
    # Plot 5: Cognitive Output and Experiment Compute (if available)
    if cognitive_outputs is not None and len(cognitive_outputs) > 0:
        fig.add_trace(
            go.Scatter(x=times.tolist(), y=cognitive_outputs.tolist(),
                      name='Cognitive Output',
                      line=dict(color='#9467bd', width=3),
                      mode='lines+markers', marker=dict(size=4)),
            row=3, col=1, secondary_y=False
        )
    
    # Add experiment compute on the same plot with secondary y-axis
    if session_data['time_series'] is not None:
        time_series = session_data['time_series']
        # Interpolate experiment compute to match cognitive output time points
        experiment_compute_interp = np.interp(times, time_series.time, time_series.experiment_compute)
        fig.add_trace(
            go.Scatter(x=times.tolist(), y=experiment_compute_interp.tolist(),
                      name='Experiment Compute',
                      line=dict(color='#2ca02c', width=3, dash='dash'),
                      mode='lines+markers', marker=dict(size=4)),
            row=3, col=1, secondary_y=True
        )
    
    # Plot 6: Progress vs Automation scatter
    fig.add_trace(
        go.Scatter(x=progress.tolist(), y=(automation_fraction*100).tolist(),
                  mode='markers+lines', name='Progress vs Automation',
                  line=dict(color='#d62728', width=2),
                  marker=dict(size=6)),
        row=3, col=2
    )
    
    # Plot 7: Rate Components Comparison (if both rates available)
    if progress_rates is not None and software_progress_rates is not None and len(progress_rates) > 0 and len(software_progress_rates) > 0:
        fig.add_trace(
            go.Scatter(x=times.tolist(), y=progress_rates.tolist(),
                      name='Overall Rate',
                      line=dict(color='#2ca02c', width=2, dash='solid'),
                      mode='lines'),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=times.tolist(), y=software_progress_rates.tolist(),
                      name='Software Rate',
                      line=dict(color='#ff7f0e', width=2, dash='dash'),
                      mode='lines'),
            row=4, col=1
        )
    
    # Plot 8: Cognitive Output Components (AI vs Human contribution)
    if ai_labor_contributions is not None and human_labor_contributions is not None and len(ai_labor_contributions) > 0 and len(human_labor_contributions) > 0:
        fig.add_trace(
            go.Scatter(x=times.tolist(), y=ai_labor_contributions.tolist(),
                      name='AI contribution',
                      line=dict(color='#1f77b4', width=2),
                      mode='lines'),
            row=4, col=2
        )
        fig.add_trace(
            go.Scatter(x=times.tolist(), y=human_labor_contributions.tolist(),
                      name='Humans only',
                      line=dict(color='#ff7f0e', width=2),
                      mode='lines'),
            row=4, col=2
        )
    
    # Plot 9: Human vs AI Labor
    if session_data['time_series'] is not None:
        time_series = session_data['time_series']
        fig.add_trace(
            go.Scatter(x=time_series.time.tolist(), y=time_series.L_HUMAN.tolist(),
                      name='Human Labor',
                      line=dict(color='#ff7f0e', width=2),
                      mode='lines+markers', marker=dict(size=4)),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_series.time.tolist(), y=time_series.L_AI.tolist(),
                      name='AI Labor',
                      line=dict(color='#1f77b4', width=2),
                      mode='lines+markers', marker=dict(size=4)),
            row=5, col=1
        )
        
        # Plot 10: Training Compute
        fig.add_trace(
            go.Scatter(x=time_series.time.tolist(), y=time_series.training_compute.tolist(),
                      name='Training Compute',
                      line=dict(color='#d62728', width=2),
                      mode='lines+markers', marker=dict(size=4)),
            row=5, col=2
        )

    # Plot 11: Research Stock
    if research_stocks is not None and len(research_stocks) > 0:
        fig.add_trace(
            go.Scatter(x=times.tolist(), y=research_stocks.tolist(),
                      name='Research Stock',
                      line=dict(color='#8c564b', width=3),
                      mode='lines+markers', marker=dict(size=4)),
            row=6, col=1
        )

    # Plot 12: Research Stock Rate
    if research_stock_rates is not None and len(research_stock_rates) > 0:
        fig.add_trace(
            go.Scatter(x=times.tolist(), y=research_stock_rates.tolist(),
                      name='Research Stock Rate',
                      line=dict(color='#e377c2', width=3),
                      mode='lines+markers', marker=dict(size=4)),
            row=6, col=2
        )

    # Plot 13: Human-Only Progress Rate
    if human_only_progress_rates is not None and len(human_only_progress_rates) > 0:
        fig.add_trace(
            go.Scatter(x=times.tolist(), y=human_only_progress_rates.tolist(),
                      name='Human-Only Progress Rate',
                      line=dict(color='#ff7f0e', width=3),
                      mode='lines+markers', marker=dict(size=4)),
            row=7, col=1
        )

    # Plot 14: Automation Progress Multiplier
    if progress_rates is not None and human_only_progress_rates is not None and len(progress_rates) > 0 and len(human_only_progress_rates) > 0:
        # Calculate automation multiplier (overall rate / human-only rate)
        automation_multiplier = []
        for i in range(len(progress_rates)):
            if human_only_progress_rates[i] > 0:
                multiplier = progress_rates[i] / human_only_progress_rates[i]
                automation_multiplier.append(multiplier if np.isfinite(multiplier) else 1.0)
            else:
                automation_multiplier.append(1.0)  # No multiplier if human-only rate is zero
        
        fig.add_trace(
            go.Scatter(x=times.tolist(), y=automation_multiplier,
                      name='Automation Multiplier',
                      line=dict(color='#d62728', width=3),
                      mode='lines+markers', marker=dict(size=4)),
            row=7, col=2
        )
        
        # Add horizontal line at y=1 for reference (no automation benefit)
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, row=7, col=2)
    
    # Update layout
    fig.update_layout(
        height=2100,  # Increased height for 7x2 layout
        showlegend=False,
        title_text="AI Progress Metrics",
        title_x=0.5,
        plot_bgcolor='white'
    )
    
    # Update axis labels and scaling
    fig.update_xaxes(title_text="Time", row=1, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text="Time", row=1, col=2, gridcolor='lightgray')
    fig.update_xaxes(title_text="Time", row=2, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text="Time", row=2, col=2, gridcolor='lightgray')
    fig.update_xaxes(title_text="Time", row=3, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text="Cumulative Progress", row=3, col=2, gridcolor='lightgray')
    fig.update_xaxes(title_text="Time", row=4, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text="Time", row=4, col=2, gridcolor='lightgray')
    
    # Use linear scale for progress plots.
    fig.update_yaxes(title_text="Progress", row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Automation (%)", row=1, col=2, gridcolor='lightgray')
    
    # Use log scale for rates if they span multiple orders of magnitude
    if progress_rates is not None and len(progress_rates) > 0 and np.max(progress_rates) > 0:
        fig.update_yaxes(title_text="Overall Rate (log scale)", type="log", row=2, col=1, gridcolor='lightgray')
    else:
        fig.update_yaxes(title_text="Overall Rate", row=2, col=1, gridcolor='lightgray')
    
    if software_progress_rates is not None and len(software_progress_rates) > 0 and np.max(software_progress_rates) > 0:
        fig.update_yaxes(title_text="Software Rate (log scale)", type="log", row=2, col=2, gridcolor='lightgray')
    else:
        fig.update_yaxes(title_text="Software Rate", row=2, col=2, gridcolor='lightgray')
    
    # Cognitive output and experiment compute scale - dual y-axes
    if cognitive_outputs is not None and len(cognitive_outputs) > 0 and np.max(cognitive_outputs) > 0:
        fig.update_yaxes(title_text="Cognitive Output (log scale)", type="log", row=3, col=1, gridcolor='lightgray', secondary_y=False)
        fig.update_yaxes(title_text="Experiment Compute (log scale)", type="log", row=3, col=1, gridcolor='lightgray', secondary_y=True)
    else:
        fig.update_yaxes(title_text="Cognitive Output", row=3, col=1, gridcolor='lightgray', secondary_y=False)
        fig.update_yaxes(title_text="Experiment Compute (log scale)", type="log", row=3, col=1, gridcolor='lightgray', secondary_y=True)
    
    fig.update_yaxes(title_text="Automation (%)", row=3, col=2, gridcolor='lightgray')
    fig.update_yaxes(title_text="Rate (log scale)", type="log", row=4, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Labor Contribution (log scale)", type="log", row=4, col=2, gridcolor='lightgray')
    
    # Add axis labels for new input time series plots
    fig.update_xaxes(title_text="Time", row=5, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text="Time", row=5, col=2, gridcolor='lightgray')
    fig.update_yaxes(title_text="Labor (log scale)", type="log", row=5, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Training Compute (log scale)", type="log", row=5, col=2, gridcolor='lightgray')

    # Add axis labels for new research stock plots
    fig.update_xaxes(title_text="Time", row=6, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text="Time", row=6, col=2, gridcolor='lightgray')
    fig.update_yaxes(title_text="Research Stock (log scale)", type="log", row=6, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Research Stock Rate (log scale)", type="log", row=6, col=2, gridcolor='lightgray')

    # Add axis labels for human-only progress rate plots
    fig.update_xaxes(title_text="Time", row=7, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text="Time", row=7, col=2, gridcolor='lightgray')
    
    # Use log scale for human-only progress rate if it spans multiple orders of magnitude
    if human_only_progress_rates is not None and len(human_only_progress_rates) > 0 and np.max(human_only_progress_rates) > 0:
        fig.update_yaxes(title_text="Human-Only Rate (log scale)", type="log", row=7, col=1, gridcolor='lightgray')
    else:
        fig.update_yaxes(title_text="Human-Only Rate", row=7, col=1, gridcolor='lightgray')
    
    # Automation multiplier axis (always linear scale)
    fig.update_yaxes(title_text="Automation Multiplier", row=7, col=2, gridcolor='lightgray')
    
    return fig

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/compute', methods=['POST'])
def compute_model():
    """Compute model with given parameters"""
    try:
        data = request.json
        
        # Parse parameters
        params_dict = data.get('parameters', {})
        params = Parameters(**params_dict)
        
        # Use stored time series or default
        time_series = session_data['time_series']
        if time_series is None:
            time_series = create_default_time_series()
            session_data['time_series'] = time_series
        
        # Get time range
        time_range = data.get('time_range', [2019, 2030])
        initial_progress = data.get('initial_progress', 1.0)
        
        # Calculate initial research stock rate, RS'(0), for software progress calculations
        try:
            start_time = time_series.time[0]
            initial_automation_at_start = compute_automation_fraction(initial_progress, params)
            initial_L_HUMAN_at_start = np.interp(start_time, time_series.time, time_series.L_HUMAN)
            initial_L_AI_at_start = np.interp(start_time, time_series.time, time_series.L_AI)
            initial_experiment_compute_at_start = np.interp(start_time, time_series.time, time_series.experiment_compute)
            
            initial_cognitive_output_at_start = compute_cognitive_output(
                initial_automation_at_start, initial_L_AI_at_start, initial_L_HUMAN_at_start,
                params.rho_cognitive, params.cognitive_output_normalization
            )
            
            initial_research_stock_rate = compute_research_stock_rate(
                initial_experiment_compute_at_start, initial_cognitive_output_at_start,
                params.alpha, params.rho_progress
            )
            
            if not np.isfinite(initial_research_stock_rate) or initial_research_stock_rate <= 0:
                logger.warning(f"Invalid initial research stock rate ({initial_research_stock_rate}), using fallback 1.0")
                initial_research_stock_rate = 1.0

        except Exception as e:
            logger.error(f"Error calculating initial research stock rate: {e}")
            return jsonify({
                'success': False,
                'error': f'Failed to calculate initial conditions: {e}',
                'error_type': 'initialization_failure'
            }), 500

        # Calculate and set the progress rate normalization to ensure initial rate = 1
        params.progress_rate_normalization = calculate_progress_rate_normalization(
            params, time_series, time_range[0], initial_progress, params.research_stock_at_simulation_start
        )
        logger.info(f"Calculated progress rate normalization: {params.progress_rate_normalization}")
        
        # Compute model with comprehensive validation and error handling
        try:
            model = ProgressModel(params, time_series)
            times, progress_values, research_stock_values = model.compute_progress_trajectory(
                time_range, initial_progress
            )
            
            # Validate results
            if len(times) == 0 or len(progress_values) == 0:
                logger.error("Model computation produced no results")
                return jsonify({
                    'success': False, 
                    'error': 'Model computation failed to produce results. Check parameter values and constraints.',
                    'suggestions': [
                        'Try more conservative parameter values',
                        'Check if time range is valid',
                        'Verify initial progress value is reasonable'
                    ]
                }), 500
            
            if not all(np.isfinite(progress_values)):
                logger.error("Model computation produced non-finite values")
                # Try to salvage what we can
                finite_mask = np.isfinite(progress_values)
                if np.any(finite_mask):
                    logger.warning("Attempting to interpolate over non-finite values")
                    times = times[finite_mask]
                    progress_values = progress_values[finite_mask]
                    if len(times) < 10:  # Too few valid points
                        return jsonify({
                            'success': False,
                            'error': 'Model produced mostly non-finite values. Parameters may be causing numerical instability.',
                            'suggestions': [
                                'Reduce elasticity parameter magnitudes',
                                'Check normalization parameters',
                                'Verify anchor constraints are reasonable'
                            ]
                        }), 500
                else:
                    return jsonify({
                        'success': False,
                        'error': 'All computed values are non-finite. Severe numerical instability detected.',
                        'suggestions': [
                            'Reset to default parameters',
                            'Use more conservative parameter bounds',
                            'Check input time series data for anomalies'
                        ]
                    }), 500
                
        except RuntimeError as e:
            if "integration" in str(e).lower():
                logger.error(f"Integration failed: {e}")
                return jsonify({
                    'success': False,
                    'error': f'Differential equation integration failed: {str(e)}',
                    'error_type': 'integration_failure',
                    'suggestions': [
                        'Try smaller time steps or shorter time range',
                        'Check if parameters cause mathematical instability',
                        'Verify initial conditions are reasonable'
                    ]
                }), 500
            else:
                logger.error(f"Runtime error in model computation: {e}")
                return jsonify({
                    'success': False,
                    'error': f'Model computation error: {str(e)}',
                    'error_type': 'runtime_error'
                }), 500
        except ValueError as e:
            logger.error(f"Value error in model computation: {e}")
            return jsonify({
                'success': False,
                'error': f'Invalid parameter or data values: {str(e)}',
                'error_type': 'value_error',
                'suggestions': [
                    'Check parameter ranges and constraints',
                    'Verify input data is valid',
                    'Try different initial conditions'
                ]
            }), 500
        except Exception as e:
            logger.error(f"Unexpected error in model computation: {e}")
            return jsonify({
                'success': False,
                'error': f'Unexpected error occurred: {str(e)}',
                'error_type': 'unexpected_error',
                'suggestions': [
                    'Try resetting to default parameters',
                    'Check system logs for more details',
                    'Contact support if problem persists'
                ]
            }), 500
        
        # Calculate all metrics using the dedicated metrics calculator
        all_metrics = calculate_all_metrics(
            model.results, params, time_series, initial_research_stock_rate
        )
        
        # Store results (including auxiliary metrics for potential export)
        session_data['current_params'] = params
        session_data['results'] = all_metrics
        
        # Create Plotly figure using the calculated metrics
        fig = create_plotly_dashboard(all_metrics)
        
        return jsonify({
            'success': True,
            'plot': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)),
            'summary': {
                'final_progress': float(progress_values[-1]),
                'final_automation': float(model.results['automation_fraction'][-1]),
                'avg_progress_rate': float(np.mean(model.results['progress_rates'])),
                'time_range': time_range
            }
        })
        
    except Exception as e:
        logger.error(f"Error computing model: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    """Upload custom time series data"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read CSV data
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.DictReader(stream)
        data = list(csv_input)
        
        # Parse data
        time = np.array([float(row['time']) for row in data])
        L_HUMAN = np.array([float(row['L_HUMAN']) for row in data])
        L_AI = np.array([float(row['L_AI']) for row in data])
        experiment_compute = np.array([float(row['experiment_compute']) for row in data])
        training_compute = np.array([float(row['training_compute']) for row in data])
        
        time_series = TimeSeriesData(time, L_HUMAN, L_AI, experiment_compute, training_compute)
        session_data['time_series'] = time_series
        
        return jsonify({
            'success': True,
            'data_summary': {
                'time_range': [float(time.min()), float(time.max())],
                'data_points': len(time),
                'preview': time_series_to_dict(time_series)
            }
        })
        
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/estimate-parameters', methods=['POST'])
def estimate_params():
    """Estimate parameters based on anchor constraints"""
    try:
        data = request.json
        
        # Parse anchor constraints
        anchor_data = data.get('anchors', [])
        anchors = []
        for a in anchor_data:
            anchors.append(AnchorConstraint(
                conditions=a['conditions'],
                target_variable=a['target_variable'],
                target_value=a['target_value'],
                weight=a.get('weight', 1.0)
            ))
        
        # Get initial parameters and initial progress
        initial_params_dict = data.get('initial_parameters', {})
        initial_params = Parameters(**initial_params_dict)
        initial_progress = data.get('initial_progress', 1.0)
        
        # Use stored time series or default
        time_series = session_data['time_series']
        if time_series is None:
            time_series = create_default_time_series()
        
        # Estimate parameters with comprehensive error handling
        try:
            estimated_params, constraint_evals = estimate_parameters(anchors, time_series, initial_params, initial_progress)
            
            # Validate the optimization results
            initial_obj = getattr(estimated_params, '_initial_objective', None)
            final_obj = getattr(estimated_params, '_final_objective', None)
            
            # Check if optimization actually improved anything
            if initial_obj is not None and final_obj is not None:
                improvement = initial_obj - final_obj
                if improvement < 1e-8:
                    logger.warning("Parameter estimation made minimal improvement")
                    # Still return success but with warning
                    return jsonify({
                        'success': True,
                        'estimated_parameters': params_to_dict(estimated_params),
                        'optimization_info': {
                            'initial_objective': initial_obj,
                            'final_objective': final_obj,
                            'improvement': improvement,
                            'warning': 'Optimization made minimal improvement. Current parameters may already be near-optimal, or constraints may be conflicting.'
                        },
                        'constraint_evaluations': constraint_evals
                    })
            
            # Check constraint satisfaction
            avg_satisfaction = np.mean([eval.get('satisfaction', 0) for eval in constraint_evals]) if constraint_evals else 0
            if avg_satisfaction < 0.5:
                logger.warning(f"Poor constraint satisfaction: {avg_satisfaction:.2f}")
                return jsonify({
                    'success': True,  # Still successful but with warning
                    'estimated_parameters': params_to_dict(estimated_params),
                    'optimization_info': {
                        'initial_objective': initial_obj,
                        'final_objective': final_obj,
                        'improvement': initial_obj - final_obj if (initial_obj and final_obj) else None,
                        'warning': f'Low average constraint satisfaction ({avg_satisfaction:.1%}). Consider adjusting constraints or accepting current parameters.'
                    },
                    'constraint_evaluations': constraint_evals
                })
            
            return jsonify({
                'success': True,
                'estimated_parameters': params_to_dict(estimated_params),
                'optimization_info': {
                    'initial_objective': initial_obj,
                    'final_objective': final_obj,
                    'improvement': initial_obj - final_obj if (initial_obj and final_obj) else None
                },
                'constraint_evaluations': constraint_evals
            })
            
        except ValueError as e:
            if "infeasible" in str(e).lower():
                logger.error(f"Constraint feasibility error: {e}")
                return jsonify({
                    'success': False,
                    'error': f'Constraint feasibility issue: {str(e)}',
                    'error_type': 'constraint_infeasible',
                    'suggestions': [
                        'Check that constraint values are physically reasonable',
                        'Ensure constraints are not contradictory',
                        'Try reducing constraint weights or target values',
                        'Verify time series data covers the constraint conditions'
                    ]
                }), 400
            else:
                logger.error(f"Value error in parameter estimation: {e}")
                return jsonify({
                    'success': False,
                    'error': f'Invalid constraint or parameter values: {str(e)}',
                    'error_type': 'value_error',
                    'suggestions': [
                        'Check constraint definitions for valid ranges',
                        'Verify target values are reasonable',
                        'Ensure initial parameters are within bounds'
                    ]
                }), 400
        except RuntimeError as e:
            logger.error(f"Runtime error in parameter estimation: {e}")
            return jsonify({
                'success': False,
                'error': f'Optimization failed: {str(e)}',
                'error_type': 'optimization_failure',
                'suggestions': [
                    'Try simpler constraints with fewer conditions',
                    'Check if constraints are mathematically feasible',
                    'Use more conservative initial parameter values',
                    'Reduce the number of constraints'
                ]
            }), 500
        except Exception as e:
            logger.error(f"Unexpected error in parameter estimation: {e}")
            return jsonify({
                'success': False,
                'error': f'Parameter estimation failed: {str(e)}',
                'error_type': 'unexpected_error',
                'suggestions': [
                    'Try resetting constraints and parameters',
                    'Check system logs for detailed error information',
                    'Verify input data integrity'
                ]
            }), 500
        
    except Exception as e:
        logger.error(f"Error estimating parameters: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export-csv')
def export_csv():
    """Export current results as CSV"""
    try:
        if session_data['results'] is None:
            return jsonify({'success': False, 'error': 'No results to export'}), 400
        
        results = session_data['results']
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['time', 'cumulative_progress', 'automation_fraction', 'progress_rate', 'software_progress_rate', 'cognitive_output', 'research_stock', 'research_stock_rate', 'human_only_progress_rate', 'ai_labor_contribution', 'human_labor_contribution'])
        
        # Write data
        for i in range(len(results['times'])):
            row = [
                results['times'][i],
                results['progress'][i],
                results['automation_fraction'][i],
                results['progress_rates'][i]
            ]
            # Add software progress rate if available
            if 'software_progress_rates' in results and i < len(results['software_progress_rates']):
                row.append(results['software_progress_rates'][i])
            else:
                row.append(0.0)
            
            # Add cognitive output if available
            if 'cognitive_outputs' in results and i < len(results['cognitive_outputs']):
                row.append(results['cognitive_outputs'][i])
            else:
                row.append(0.0)

            if 'research_stock' in results and i < len(results['research_stock']):
                row.append(results['research_stock'][i])
            else:
                row.append(0.0)

            if 'research_stock_rates' in results and i < len(results['research_stock_rates']):
                row.append(results['research_stock_rates'][i])
            else:
                row.append(0.0)

            # Add human-only progress rate if available
            if 'human_only_progress_rates' in results and i < len(results['human_only_progress_rates']):
                row.append(results['human_only_progress_rates'][i])
            else:
                row.append(0.0)

            # Add AI labor contribution if available
            if 'ai_labor_contributions' in results and i < len(results['ai_labor_contributions']):
                row.append(results['ai_labor_contributions'][i])
            else:
                row.append(0.0)

            # Add human labor contribution if available
            if 'human_labor_contributions' in results and i < len(results['human_labor_contributions']):
                row.append(results['human_labor_contributions'][i])
            else:
                row.append(0.0)
            
            writer.writerow(row)
        
        # Prepare file for download
        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        output.close()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_progress_results_{timestamp}.csv"
        
        return send_file(
            mem,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/default-data')
def get_default_data():
    """Get default time series data and parameters"""
    time_series = create_default_time_series()
    params = create_default_parameters()
    
    session_data['time_series'] = time_series
    session_data['current_params'] = params
    
    return jsonify({
        'time_series': time_series_to_dict(time_series),
        'parameters': {
            'rho_cognitive': params.rho_cognitive,
            'rho_progress': params.rho_progress,
            'alpha': params.alpha,
            'software_progress_share': params.software_progress_share,
            'automation_fraction_at_superhuman_coder': params.automation_fraction_at_superhuman_coder,
            'progress_at_half_sc_automation': params.progress_at_half_sc_automation,
            'automation_slope': params.automation_slope,
            'research_stock_at_simulation_start': params.research_stock_at_simulation_start,
            'progress_rate_normalization': params.progress_rate_normalization,
            'cognitive_output_normalization': params.cognitive_output_normalization
        }
    })

# Initialize app data when module loads (for both direct run and gunicorn)
import os
try:
    session_data['time_series'] = create_default_time_series()
    session_data['current_params'] = create_default_parameters()
    logger.info("Application initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize application data: {e}")
    # Set minimal defaults to prevent complete failure
    session_data['time_series'] = None
    session_data['current_params'] = None

if __name__ == '__main__':
    # Get port from environment variable for deployment platforms
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(debug=debug, host='0.0.0.0', port=port)