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
from typing import Dict, Any, List, Callable

from progress_model import (
    ProgressModel, Parameters, TimeSeriesData, 
    AnchorConstraint, estimate_parameters,
    progress_rate_at_time, compute_cognitive_output,
    compute_software_progress_rate, compute_automation_fraction,
    compute_research_stock_rate, compute_overall_progress_rate,
    calculate_initial_research_stock, setup_model_with_normalization, compute_initial_conditions
)

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

# Plot Configuration System
class PlotConfig:
    """Configuration for a single plot"""
    def __init__(self, title: str, plot_func: Callable, row: int, col: int, 
                 secondary_y: bool = False, **kwargs):
        self.title = title
        self.plot_func = plot_func
        self.row = row
        self.col = col
        self.secondary_y = secondary_y
        self.kwargs = kwargs

class TabConfig:
    """Configuration for a tab containing multiple plots"""
    def __init__(self, tab_id: str, tab_name: str, plots: List[PlotConfig], 
                 rows: int, cols: int, subplot_titles: List[str] = None,
                 specs: List[List[Dict]] = None):
        self.tab_id = tab_id
        self.tab_name = tab_name
        self.plots = plots
        self.rows = rows
        self.cols = cols
        self.subplot_titles = subplot_titles or [plot.title for plot in plots]
        self.specs = specs

# Plot Functions for Input Time Series
def plot_human_labor(fig, times, values, row, col):
    """Plot human labor over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=values.tolist(),
                  name='Human Labor',
                  line=dict(color='#ff7f0e', width=3),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col
    )

def plot_ai_labor(fig, times, values, row, col):
    """Plot AI labor over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=values.tolist(),
                  name='AI Labor',
                  line=dict(color='#1f77b4', width=3),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col
    )

def plot_experiment_compute(fig, times, values, row, col):
    """Plot experiment compute over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=values.tolist(),
                  name='Experiment Compute',
                  line=dict(color='#2ca02c', width=3),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col
    )

def plot_training_compute(fig, times, values, row, col):
    """Plot training compute over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=values.tolist(),
                  name='Training Compute',
                  line=dict(color='#d62728', width=3),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col
    )

def plot_labor_comparison(fig, time_series, row, col):
    """Plot human vs AI labor comparison"""
    fig.add_trace(
        go.Scatter(x=time_series.time.tolist(), y=time_series.L_HUMAN.tolist(),
                  name='Human Labor',
                  line=dict(color='#ff7f0e', width=2),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=time_series.time.tolist(), y=time_series.L_AI.tolist(),
                  name='AI Labor',
                  line=dict(color='#1f77b4', width=2),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col
    )

def plot_compute_comparison(fig, time_series, row, col):
    """Plot experiment vs training compute comparison"""
    fig.add_trace(
        go.Scatter(x=time_series.time.tolist(), y=time_series.experiment_compute.tolist(),
                  name='Experiment Compute',
                  line=dict(color='#2ca02c', width=2),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=time_series.time.tolist(), y=time_series.training_compute.tolist(),
                  name='Training Compute',
                  line=dict(color='#d62728', width=2),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col
    )

# Plot Functions for Output Metrics
def plot_cumulative_progress(fig, times, progress, row, col):
    """Plot cumulative progress over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=progress.tolist(), 
                  name='Cumulative Progress',
                  line=dict(color='#1f77b4', width=3),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col
    )

def plot_automation_fraction(fig, times, automation_fraction, row, col):
    """Plot automation fraction over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=(automation_fraction*100).tolist(), 
                  name='Automation %', line=dict(color='#ff7f0e', width=3),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col
    )

def plot_progress_rate(fig, times, progress_rates, row, col):
    """Plot overall progress rate over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=progress_rates.tolist(), 
                  name='Overall Progress Rate',
                  line=dict(color='#2ca02c', width=3),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col
    )

def plot_software_progress_rate(fig, times, software_progress_rates, row, col):
    """Plot software progress rate over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=software_progress_rates.tolist(),
                  name='Software Progress Rate',
                  line=dict(color='#ff7f0e', width=3),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col
    )

def plot_cognitive_output_with_compute(fig, times, cognitive_outputs, row, col, secondary_y=False):
    """Plot cognitive output with experiment compute on secondary axis"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=cognitive_outputs.tolist(),
                  name='Cognitive Output',
                  line=dict(color='#9467bd', width=3),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col, secondary_y=False
    )
    
    # Add experiment compute on secondary y-axis
    time_series = session_data['time_series']
    experiment_compute_interp = np.interp(times, time_series.time, time_series.experiment_compute)
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=experiment_compute_interp.tolist(),
                  name='Experiment Compute',
                  line=dict(color='#2ca02c', width=3, dash='dash'),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col, secondary_y=True
    )

def plot_progress_vs_automation(fig, progress, automation_fraction, row, col):
    """Plot progress vs automation scatter"""
    fig.add_trace(
        go.Scatter(x=progress.tolist(), y=(automation_fraction*100).tolist(),
                  mode='markers+lines', name='Progress vs Automation',
                  line=dict(color='#d62728', width=2),
                  marker=dict(size=6)),
        row=row, col=col
    )

def plot_rate_components(fig, times, progress_rates, software_progress_rates, row, col):
    """Plot rate components comparison"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=progress_rates.tolist(),
                  name='Overall Rate',
                  line=dict(color='#2ca02c', width=2, dash='solid'),
                  mode='lines'),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=software_progress_rates.tolist(),
                  name='Software Rate',
                  line=dict(color='#ff7f0e', width=2, dash='dash'),
                  mode='lines'),
        row=row, col=col
    )

def plot_cognitive_components(fig, times, ai_labor_contributions, human_labor_contributions, row, col):
    """Plot cognitive output components (AI vs Human)"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=ai_labor_contributions.tolist(),
                  name='AI contribution',
                  line=dict(color='#1f77b4', width=2),
                  mode='lines'),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=human_labor_contributions.tolist(),
                  name='Humans only',
                  line=dict(color='#ff7f0e', width=2),
                  mode='lines'),
        row=row, col=col
    )

def plot_research_stock(fig, times, research_stocks, row, col):
    """Plot research stock over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=research_stocks.tolist(),
                  name='Research Stock',
                  line=dict(color='#8c564b', width=3),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col
    )

def plot_research_stock_rate(fig, times, research_stock_rates, row, col):
    """Plot research stock rate over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=research_stock_rates.tolist(),
                  name='Research Stock Rate',
                  line=dict(color='#e377c2', width=3),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col
    )

def plot_human_only_progress_rate(fig, times, human_only_progress_rates, row, col):
    """Plot human-only progress rate over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=human_only_progress_rates.tolist(),
                  name='Human-Only Progress Rate',
                  line=dict(color='#ff7f0e', width=3),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col
    )

def plot_automation_multiplier(fig, times, automation_multipliers, row, col):
    """Plot automation progress multiplier over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=automation_multipliers.tolist(),
                  name='Automation Multiplier',
                  line=dict(color='#d62728', width=3),
                  mode='lines+markers', marker=dict(size=4)),
        row=row, col=col
    )
    # Add horizontal reference line at y=1
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)

# Tab Configuration
def get_tab_configurations():
    """
    Get the configuration for all tabs and their plots.
    
    This function provides a clean, configuration-based approach to organizing plots.
    To add a new tab, create a new TabConfig with its plots.
    To move plots between tabs, simply move the PlotConfig objects.
    To add new plots, create new plot functions and PlotConfig objects.
    
    Example of adding a new tab:
    
    new_plots = [
        PlotConfig("My Plot", lambda fig, data, r, c: my_plot_function(fig, data, r, c), 1, 1),
    ]
    new_tab = TabConfig(
        tab_id="my_tab",
        tab_name="My Custom Tab", 
        plots=new_plots,
        rows=1, cols=1
    )
    
    Then add new_tab to the return list.
    """
    
    # Input Time Series Tab
    input_plots = [
        PlotConfig("Human Labor", lambda fig, data, r, c: plot_human_labor(fig, data['time_series'].time, data['time_series'].L_HUMAN, r, c), 1, 1),
        PlotConfig("AI Labor", lambda fig, data, r, c: plot_ai_labor(fig, data['time_series'].time, data['time_series'].L_AI, r, c), 1, 2),
        PlotConfig("Experiment Compute", lambda fig, data, r, c: plot_experiment_compute(fig, data['time_series'].time, data['time_series'].experiment_compute, r, c), 2, 1),
        PlotConfig("Training Compute", lambda fig, data, r, c: plot_training_compute(fig, data['time_series'].time, data['time_series'].training_compute, r, c), 2, 2),
        PlotConfig("Labor Comparison", lambda fig, data, r, c: plot_labor_comparison(fig, data['time_series'], r, c), 3, 1),
        PlotConfig("Compute Comparison", lambda fig, data, r, c: plot_compute_comparison(fig, data['time_series'], r, c), 3, 2),
    ]
    
    input_tab = TabConfig(
        tab_id="input_data",
        tab_name="Input Time Series",
        plots=input_plots,
        rows=3,
        cols=2,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Output Metrics Tab
    output_plots = [
        PlotConfig("Cumulative Progress", lambda fig, data, r, c: plot_cumulative_progress(fig, data['metrics']['times'], data['metrics']['progress'], r, c), 1, 1),
        PlotConfig("Automation Fraction", lambda fig, data, r, c: plot_automation_fraction(fig, data['metrics']['times'], data['metrics']['automation_fraction'], r, c), 1, 2),
        PlotConfig("Overall Progress Rate", lambda fig, data, r, c: plot_progress_rate(fig, data['metrics']['times'], data['metrics']['progress_rates'], r, c), 2, 1),
        PlotConfig("Software Progress Rate", lambda fig, data, r, c: plot_software_progress_rate(fig, data['metrics']['times'], data['metrics']['software_progress_rates'], r, c), 2, 2),
        PlotConfig("Cognitive Output & Compute", lambda fig, data, r, c: plot_cognitive_output_with_compute(fig, data['metrics']['times'], data['metrics']['cognitive_outputs'], r, c), 3, 1, secondary_y=True),
        PlotConfig("Progress vs Automation", lambda fig, data, r, c: plot_progress_vs_automation(fig, data['metrics']['progress'], data['metrics']['automation_fraction'], r, c), 3, 2),
        PlotConfig("Rate Components", lambda fig, data, r, c: plot_rate_components(fig, data['metrics']['times'], data['metrics']['progress_rates'], data['metrics']['software_progress_rates'], r, c), 4, 1),
        PlotConfig("Cognitive Components", lambda fig, data, r, c: plot_cognitive_components(fig, data['metrics']['times'], data['metrics']['ai_labor_contributions'], data['metrics']['human_labor_contributions'], r, c), 4, 2),
        PlotConfig("Research Stock", lambda fig, data, r, c: plot_research_stock(fig, data['metrics']['times'], data['metrics']['research_stock'], r, c), 5, 1),
        PlotConfig("Research Stock Rate", lambda fig, data, r, c: plot_research_stock_rate(fig, data['metrics']['times'], data['metrics']['research_stock_rates'], r, c), 5, 2),
        PlotConfig("Human-Only Progress Rate", lambda fig, data, r, c: plot_human_only_progress_rate(fig, data['metrics']['times'], data['metrics']['human_only_progress_rates'], r, c), 6, 1),
        PlotConfig("Automation Multiplier", lambda fig, data, r, c: plot_automation_multiplier(fig, data['metrics']['times'], data['metrics']['automation_multipliers'], r, c), 6, 2),
    ]
    
    output_tab = TabConfig(
        tab_id="output_metrics",
        tab_name="Output Metrics",
        plots=output_plots,
        rows=6,
        cols=2,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    return [input_tab, output_tab]

def create_tab_figure(tab_config: TabConfig, data: Dict[str, Any]) -> go.Figure:
    """Create a plotly figure for a specific tab"""
    
    # Create subplots
    fig = make_subplots(
        rows=tab_config.rows, 
        cols=tab_config.cols,
        subplot_titles=tab_config.subplot_titles,
        specs=tab_config.specs
    )
    
    # Add plots
    for plot_config in tab_config.plots:
        try:
            plot_config.plot_func(fig, data, plot_config.row, plot_config.col)
        except Exception as e:
            logger.warning(f"Failed to create plot '{plot_config.title}': {e}")
            continue
    
    # Update layout
    fig.update_layout(
        autosize=True,
        showlegend=False,
        title_text=tab_config.tab_name,
        title_x=0.5,
        plot_bgcolor='white'
    )
    
    # Update axes
    update_axes_for_tab(fig, tab_config, data)
    
    return fig

def update_axes_for_tab(fig: go.Figure, tab_config: TabConfig, data: Dict[str, Any]):
    """Update axis labels and scaling for a specific tab"""
    
    if tab_config.tab_id == "input_data":
        # Input data axes
        for i in range(1, tab_config.rows + 1):
            for j in range(1, tab_config.cols + 1):
                fig.update_xaxes(title_text="Time", row=i, col=j, gridcolor='lightgray')
        
        fig.update_yaxes(title_text="Human Labor (log scale)", type="log", row=1, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="AI Labor (log scale)", type="log", row=1, col=2, gridcolor='lightgray')
        fig.update_yaxes(title_text="Experiment Compute (log scale)", type="log", row=2, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Training Compute (log scale)", type="log", row=2, col=2, gridcolor='lightgray')
        fig.update_yaxes(title_text="Labor (log scale)", type="log", row=3, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Compute (log scale)", type="log", row=3, col=2, gridcolor='lightgray')
        
    elif tab_config.tab_id == "output_metrics":
        # Output metrics axes - similar to original dashboard
        metrics = data.get('metrics', {})
        times = np.array(metrics.get('times', []))
        progress_rates = np.array(metrics.get('progress_rates', []))
        software_progress_rates = np.array(metrics.get('software_progress_rates', []))
        cognitive_outputs = np.array(metrics.get('cognitive_outputs', []))
        human_only_progress_rates = np.array(metrics.get('human_only_progress_rates', []))
        
        # Set x-axis labels
        for i in range(1, tab_config.rows + 1):
            for j in range(1, tab_config.cols + 1):
                if i == 3 and j == 2:  # Progress vs Automation scatter
                    fig.update_xaxes(title_text="Cumulative Progress", row=i, col=j, gridcolor='lightgray')
                else:
                    fig.update_xaxes(title_text="Time", row=i, col=j, gridcolor='lightgray')
        
        # Set y-axis labels and scaling
        fig.update_yaxes(title_text="Progress", row=1, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Automation (%)", row=1, col=2, gridcolor='lightgray')
        
        # Progress rates with log scale if needed
        if len(progress_rates) > 0 and np.max(progress_rates) > 0:
            fig.update_yaxes(title_text="Overall Rate (log scale)", type="log", row=2, col=1, gridcolor='lightgray')
        else:
            fig.update_yaxes(title_text="Overall Rate", row=2, col=1, gridcolor='lightgray')
        
        if len(software_progress_rates) > 0 and np.max(software_progress_rates) > 0:
            fig.update_yaxes(title_text="Software Rate (log scale)", type="log", row=2, col=2, gridcolor='lightgray')
        else:
            fig.update_yaxes(title_text="Software Rate", row=2, col=2, gridcolor='lightgray')
        
        # Cognitive output with dual y-axes
        if len(cognitive_outputs) > 0 and np.max(cognitive_outputs) > 0:
            fig.update_yaxes(title_text="Cognitive Output (log scale)", type="log", row=3, col=1, gridcolor='lightgray', secondary_y=False)
            fig.update_yaxes(title_text="Experiment Compute (log scale)", type="log", row=3, col=1, gridcolor='lightgray', secondary_y=True)
        else:
            fig.update_yaxes(title_text="Cognitive Output", row=3, col=1, gridcolor='lightgray', secondary_y=False)
            fig.update_yaxes(title_text="Experiment Compute (log scale)", type="log", row=3, col=1, gridcolor='lightgray', secondary_y=True)
        
        fig.update_yaxes(title_text="Automation (%)", row=3, col=2, gridcolor='lightgray')
        fig.update_yaxes(title_text="Rate (log scale)", type="log", row=4, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Labor Contribution (log scale)", type="log", row=4, col=2, gridcolor='lightgray')
        fig.update_yaxes(title_text="Research Stock (log scale)", type="log", row=5, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Research Stock Rate (log scale)", type="log", row=5, col=2, gridcolor='lightgray')
        
        # Human-only progress rate scaling
        if len(human_only_progress_rates) > 0 and np.max(human_only_progress_rates) > 0:
            fig.update_yaxes(title_text="Human-Only Rate (log scale)", type="log", row=6, col=1, gridcolor='lightgray')
        else:
            fig.update_yaxes(title_text="Human-Only Rate", row=6, col=1, gridcolor='lightgray')
        
        fig.update_yaxes(title_text="Automation Multiplier", row=6, col=2, gridcolor='lightgray')

def create_multi_tab_dashboard(metrics: Dict[str, Any]) -> Dict[str, go.Figure]:
    """Create dashboard with multiple tabs"""
    
    # Validate and clean metrics data
    times = np.array(metrics['times'], dtype=float)
    progress = np.array(metrics['progress'], dtype=float)
    automation_fraction = np.array(metrics['automation_fraction'], dtype=float)
    
    # Validate input data
    if len(times) == 0:
        raise ValueError("Empty input data")
    
    # Handle invalid values
    valid_mask = np.isfinite(times) & np.isfinite(progress) & np.isfinite(automation_fraction)
    if not np.any(valid_mask):
        raise ValueError("No valid data points found")
    
    # Clean all metrics arrays
    cleaned_metrics = {}
    for key, values in metrics.items():
        if isinstance(values, (list, np.ndarray)):
            values_array = np.array(values, dtype=float)
            cleaned_values = values_array[valid_mask] if len(values_array) == len(valid_mask) else values_array
            cleaned_metrics[key] = np.where(np.isfinite(cleaned_values), cleaned_values, 0)
        else:
            cleaned_metrics[key] = values
    
    # Special handling for automation_multipliers - default to 1.0 instead of 0
    if 'automation_multipliers' in cleaned_metrics:
        cleaned_metrics['automation_multipliers'] = np.where(
            np.isfinite(cleaned_metrics['automation_multipliers']), 
            cleaned_metrics['automation_multipliers'], 
            1.0
        )
    
    # Prepare data for plotting
    plot_data = {
        'time_series': session_data['time_series'],
        'metrics': cleaned_metrics
    }
    
    # Get tab configurations
    tab_configs = get_tab_configurations()
    
    # Create figures for each tab
    figures = {}
    for tab_config in tab_configs:
        try:
            figures[tab_config.tab_id] = create_tab_figure(tab_config, plot_data)
        except Exception as e:
            logger.error(f"Failed to create tab '{tab_config.tab_name}': {e}")
            continue
    
    return figures

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
        time = np.linspace(2029, 2030, 12)
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
        'progress_rate_normalization': params.progress_rate_normalization,
        'cognitive_output_normalization': params.cognitive_output_normalization
    }



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
        time_range = data.get('time_range', [2029, 2030])
        initial_progress = data.get('initial_progress', 1.0)
        
        # Use utility function to set up model with proper normalization
        try:
            params, initial_conditions = setup_model_with_normalization(time_series, params, initial_progress)
            # Extract needed values for metrics calculation
            initial_research_stock_rate = initial_conditions.research_stock_rate
            initial_research_stock_calc = initial_conditions.research_stock
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            return jsonify({
                'success': False,
                'error': f'Failed to set up model: {e}',
                'error_type': 'initialization_failure'
            }), 500
        
        # Compute model with comprehensive validation and error handling
        try:
            model = ProgressModel(params, time_series)
            times, progress_values, research_stock_values = model.compute_progress_trajectory(
                time_range, initial_progress
            )
            
            # All metrics are now available in model.results - no need for separate calculation
            
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
        
        # All metrics are now computed by ProgressModel - use them directly
        all_metrics = model.results
        
        # Store results (including auxiliary metrics for potential export)
        session_data['current_params'] = params
        session_data['results'] = all_metrics
        
        # Create multi-tab dashboard using the calculated metrics
        figures = create_multi_tab_dashboard(all_metrics)
        
        # Convert figures to JSON format
        plots = {}
        for tab_id, fig in figures.items():
            plots[tab_id] = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        
        # Get tab metadata
        tab_configs = get_tab_configurations()
        tabs_info = [{'id': tab.tab_id, 'name': tab.tab_name} for tab in tab_configs]
        
        return jsonify({
            'success': True,
            'plots': plots,
            'tabs': tabs_info,
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

@app.route('/api/parameter-config', methods=['GET'])
def get_parameter_config():
    """Get parameter configuration including bounds, defaults, and metadata"""
    try:
        import model_config as cfg
        
        parameter_config = {
            'bounds': cfg.PARAMETER_BOUNDS,
            'defaults': cfg.DEFAULT_PARAMETERS,
            'validation_thresholds': cfg.PARAM_VALIDATION_THRESHOLDS,
            'descriptions': {
                'rho_cognitive': {
                    'name': 'Cognitive Elasticity (ρ_cognitive)',
                    'description': 'Elasticity of substitution between AI and human cognitive labor',
                    'units': 'dimensionless'
                },
                'rho_progress': {
                    'name': 'Progress Elasticity (ρ_progress)', 
                    'description': 'Elasticity of substitution in progress production function',
                    'units': 'dimensionless'
                },
                'alpha': {
                    'name': 'Compute Weight (α)',
                    'description': 'Weight of compute vs cognitive output in progress production',
                    'units': 'dimensionless'
                },
                'software_progress_share': {
                    'name': 'Software Share',
                    'description': 'Share of progress attributable to software vs hardware',
                    'units': 'dimensionless'
                },
                'automation_fraction_at_superhuman_coder': {
                    'name': 'Max Automation',
                    'description': 'Automation fraction when AI reaches superhuman coding ability',
                    'units': 'fraction'
                },
                'progress_at_half_sc_automation': {
                    'name': 'Half-Max Progress',
                    'description': 'Progress level at 50% of max automation',
                    'units': 'dimensionless'
                },
                'automation_slope': {
                    'name': 'Automation Slope',  
                    'description': 'Steepness of automation curve',
                    'units': 'dimensionless'
                },
                'cognitive_output_normalization': {
                    'name': 'Cognitive Output Normalization',
                    'description': 'Normalization factor for cognitive output',
                    'units': 'dimensionless'
                },
                'progress_rate_normalization': {
                    'name': 'Progress Rate Normalization',
                    'description': 'Normalization factor for progress rates (auto-calculated)',
                    'units': 'dimensionless'
                }
            }
        }
        
        return jsonify({
            'success': True,
            'config': parameter_config
        })
        
    except Exception as e:
        logger.error(f"Error getting parameter config: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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
        fixed_params = data.get('fixed_params', [])
        
        # Use stored time series or default
        time_series = session_data['time_series']
        if time_series is None:
            time_series = create_default_time_series()
        
        # Estimate parameters with comprehensive error handling
        try:
            estimated_params, constraint_evals = estimate_parameters(
                anchors, time_series, initial_params, initial_progress, fixed_params=fixed_params
            )
            
            # Set up estimated parameters with proper normalization
            time_range = data.get('time_range', [time_series.time[0], time_series.time[-1]])
            estimated_params, initial_conditions = setup_model_with_normalization(time_series, estimated_params, initial_progress)
            initial_research_stock_calc = initial_conditions.research_stock
            
            # Update session state with the optimized parameters
            session_data['current_params'] = estimated_params
            
            # Automatically run the model computation with the estimated parameters
            try:
                model = ProgressModel(estimated_params, time_series)
                times, progress_values, research_stock_values = model.compute_progress_trajectory(
                    time_range, initial_progress
                )
                
                # All metrics are now computed by ProgressModel - use them directly
                all_metrics = model.results
                
                # Store results
                session_data['results'] = all_metrics
                
                # Create multi-tab dashboard
                figures = create_multi_tab_dashboard(all_metrics)
                
                # Convert figures to JSON format
                plots = {}
                for tab_id, fig in figures.items():
                    plots[tab_id] = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
                
                # Get tab metadata
                tab_configs = get_tab_configurations()
                tabs_info = [{'id': tab.tab_id, 'name': tab.tab_name} for tab in tab_configs]
                
                # Validate the optimization results
                initial_obj = getattr(estimated_params, '_initial_objective', None)
                final_obj = getattr(estimated_params, '_final_objective', None)
                
                return jsonify({
                    'success': True,
                    'estimated_parameters': params_to_dict(estimated_params),
                    'optimization_info': {
                        'initial_objective': initial_obj,
                        'final_objective': final_obj,
                        'improvement': initial_obj - final_obj if (initial_obj and final_obj) else None
                    },
                    'constraint_evaluations': constraint_evals,
                    'plots': plots,
                    'tabs': tabs_info,
                    'summary': {
                        'final_progress': float(progress_values[-1]),
                        'final_automation': float(model.results['automation_fraction'][-1]),
                        'avg_progress_rate': float(np.mean(model.results['progress_rates'])),
                        'time_range': time_range
                    }
                })
                
            except Exception as compute_error:
                logger.error(f"Error computing model with estimated parameters: {compute_error}")
                # Still return the estimated parameters even if computation fails
                return jsonify({
                    'success': True,
                    'estimated_parameters': params_to_dict(estimated_params),
                    'optimization_info': {
                        'initial_objective': getattr(estimated_params, '_initial_objective', None),
                        'final_objective': getattr(estimated_params, '_final_objective', None),
                        'warning': f'Parameter estimation succeeded but model computation failed: {str(compute_error)}'
                    },
                    'constraint_evaluations': constraint_evals,
                    'computation_error': str(compute_error)
                })
            
            # This code is now unreachable, but keeping for reference in case of refactoring
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
            'progress_rate_normalization': params.progress_rate_normalization,
            'cognitive_output_normalization': params.cognitive_output_normalization
        }
    })

@app.route('/api/tab-config')
def get_tab_config():
    """Get tab configuration for frontend"""
    try:
        tab_configs = get_tab_configurations()
        
        config_data = []
        for tab_config in tab_configs:
            plot_info = []
            for plot_config in tab_config.plots:
                plot_info.append({
                    'title': plot_config.title,
                    'row': plot_config.row,
                    'col': plot_config.col,
                    'secondary_y': plot_config.secondary_y
                })
            
            config_data.append({
                'tab_id': tab_config.tab_id,
                'tab_name': tab_config.tab_name,
                'rows': tab_config.rows,
                'cols': tab_config.cols,
                'plots': plot_info
            })
        
        return jsonify({
            'success': True,
            'tabs': config_data
        })
        
    except Exception as e:
        logger.error(f"Error getting tab config: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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