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
import yaml

from progress_model import (
    ProgressModel, Parameters, TimeSeriesData, 
)
from model_config import PLOT_METADATA, TAB_CONFIGURATIONS, PARAMETER_BOUNDS

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

# Register blueprints
try:
    from monte_carlo import mc_bp
    app.register_blueprint(mc_bp)
except Exception as _e:
    # Defer blueprint errors to runtime logs; core app should still function
    logger.warning(f"Monte Carlo blueprint not registered: {_e}")

# Plot Configuration System
class PlotConfig:
    """Configuration for a single plot using centralized metadata"""
    def __init__(self, function_name: str, plot_func: Callable, row: int, col: int):
        self.function_name = function_name
        self.plot_func = plot_func
        self.row = row
        self.col = col
        
        # Load configuration from centralized metadata
        if function_name not in PLOT_METADATA:
            raise ValueError(f"Unknown plot function: {function_name}")
        
        metadata = PLOT_METADATA[function_name]
        self.title = metadata['title']
        self.x_axis_title = metadata['x_axis']['title']
        self.x_axis_type = metadata['x_axis'].get('type', 'linear')
        self.y_axis_title = metadata['y_axis']['title']
        self.y_axis_type = metadata['y_axis'].get('type', 'linear')
        self.y_axis_range = metadata['y_axis'].get('range')
        self.y_axis_custom_ticks = metadata['y_axis'].get('custom_ticks', False)
        
        # Secondary y-axis support (for future extension)
        self.secondary_y = False
        self.y_axis_secondary_title = None
        self.y_axis_secondary_type = "linear"
        
        # Store full metadata for advanced use cases
        self.metadata = metadata

class TabConfig:
    """Configuration for a tab containing multiple plots"""
    def __init__(self, tab_id: str, tab_name: str, plots: List[PlotConfig], 
                 rows: int, cols: int, subplot_titles: List[str] = None,
                 specs: List[List[Dict]] = None, explanation: str = None):
        self.tab_id = tab_id
        self.tab_name = tab_name
        self.plots = plots
        self.rows = rows
        self.cols = cols
        self.subplot_titles = subplot_titles or [plot.title for plot in plots]
        self.specs = specs
        self.explanation = explanation

# Professional Plotly Theme
def get_professional_plotly_template() -> go.layout.Template:
    """Return a cohesive, professional Plotly template used across all figures."""
    return go.layout.Template(
        layout=dict(
            font=dict(
                family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
                size=14,
                color="#2b2d42",
            ),
            title=dict(
                font=dict(size=18, color="#2b2d42"),
                x=0.5,
                xanchor="center",
            ),
            colorway=[
                "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
                "#EDC949", "#AF7AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
            ],
            paper_bgcolor="white",
            plot_bgcolor="white",
            hovermode="x unified",
            hoverlabel=dict(bgcolor="white", bordercolor="#e0e0e0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=""),
            margin=dict(t=60, b=40, l=60, r=40),
            xaxis=dict(
                showline=True, linewidth=1, linecolor="#d9d9d9",
                showgrid=True, gridcolor="#ebedf0", gridwidth=1,
                zeroline=False, ticks="outside", tickcolor="#bdbdbd", ticklen=6, mirror=True,
                showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, spikedash="dot", spikecolor="#999999",
            ),
            yaxis=dict(
                showline=True, linewidth=1, linecolor="#d9d9d9",
                showgrid=True, gridcolor="#ebedf0", gridwidth=1,
                zeroline=False, ticks="outside", tickcolor="#bdbdbd", ticklen=6, mirror=True,
            ),
        )
    )

# Plot Functions for Input Time Series
def plot_human_labor(fig, times, values, row, col):
    """Plot human labor over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=values.tolist(),
                  name='Human Labor',
                  line=dict(color='#ff7f0e', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[(format_decimal_year_to_month_year(t), format_number_mbt(v)) for t, v in zip(times, values)],
                  hovertemplate='Year: %{customdata[0]}<br>Human Labor: %{customdata[1]}<extra></extra>'),
        row=row, col=col
    )

def plot_ai_labor(fig, times, values, row, col):
    """Plot AI labor over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=values.tolist(),
                  name='Inference Compute for R&D Automation',
                  line=dict(color='#1f77b4', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[(format_decimal_year_to_month_year(t), format_number_mbt(v)) for t, v in zip(times, values)],
                  hovertemplate='Year: %{customdata[0]}<br>Inference Compute: %{customdata[1]}<extra></extra>'),
        row=row, col=col
    )

def plot_experiment_compute(fig, times, values, row, col):
    """Plot experiment compute over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=values.tolist(),
                  name='Experiment Compute',
                  line=dict(color='#2ca02c', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[(format_decimal_year_to_month_year(t), format_number_mbt(v)) for t, v in zip(times, values)],
                  hovertemplate='Year: %{customdata[0]}<br>Experiment Compute: %{customdata[1]}<extra></extra>'),
        row=row, col=col
    )

def plot_training_compute_growth_rate(fig, times, values, row, col):
    """Plot training compute over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=values.tolist(),
                  name='Training Compute',
                  line=dict(color='#d62728', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[(format_decimal_year_to_month_year(t), format_number_mbt(v)) for t, v in zip(times, values)],
                  hovertemplate='Year: %{customdata[0]}<br>Training Compute: %{customdata[1]}<extra></extra>'),
        row=row, col=col
    )

def plot_software_efficiency(fig, times, values, row, col):
    """Plot software efficiency over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=values.tolist(),
                  name='Software Efficiency',
                  line=dict(color='#ff7f0e', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[(format_decimal_year_to_month_year(t), f"{v:.3f}") for t, v in zip(times, values)],
                  hovertemplate='Year: %{customdata[0]}<br>Software Efficiency: %{customdata[1]}<extra></extra>'),
        row=row, col=col
    )

def plot_labor_comparison(fig, time_series, row, col):
    """Plot human vs AI labor comparison"""
    fig.add_trace(
        go.Scatter(x=time_series.time.tolist(), y=time_series.L_HUMAN.tolist(),
                  name='Human Labor',
                  line=dict(color='#ff7f0e', width=2),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in time_series.time],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=time_series.time.tolist(), y=time_series.inference_compute.tolist(),
                  name='Inference Compute',
                  line=dict(color='#1f77b4', width=2),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in time_series.time],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )

def plot_compute_comparison(fig, time_series, row, col):
    """Plot experiment vs training compute comparison"""
    fig.add_trace(
        go.Scatter(x=time_series.time.tolist(), y=time_series.experiment_compute.tolist(),
                  name='Experiment Compute',
                  line=dict(color='#2ca02c', width=2),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in time_series.time],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=time_series.time.tolist(), y=time_series.training_compute_growth_rate.tolist(),
                  name='Training Compute',
                  line=dict(color='#d62728', width=2),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in time_series.time],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )

# Plot Functions for Output Metrics
def plot_cumulative_progress(fig, times, progress, row, col):
    """Plot cumulative progress over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=progress.tolist(), 
                  name='Effective Compute (OOMs)',
                  line=dict(color='#1f77b4', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    
    # Add training compute and vertical line for SC time if available
    results = session_data.get('results')
    
    # Add training compute if available
    if results and 'training_compute' in results:
        training_compute = results['training_compute']
        fig.add_trace(
            go.Scatter(x=times.tolist(), y=training_compute.tolist(), 
                      name='Training Compute',
                      line=dict(color='#9467bd', width=3),
                      mode='lines+markers', marker=dict(size=4),
                      customdata=[(format_decimal_year_to_month_year(t), format_number_mbt(v)) for t, v in zip(times, training_compute)],
                      hovertemplate='Year: %{customdata[0]}<br>Training Compute: %{customdata[1]}<extra></extra>'),
            row=row, col=col
        )
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[progress.min(), progress.max()],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_automation_fraction(fig, times, automation_fraction, row, col):
    """Plot automation fraction over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=(automation_fraction*100).tolist(), 
                  name='Automation %', line=dict(color='#ff7f0e', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    
    # Add vertical line for SC time if available
    results = session_data.get('results')
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[(automation_fraction*100).min(), (automation_fraction*100).max()],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_progress_rate(fig, times, progress_rates, row, col):
    """Plot overall progress rate over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=progress_rates.tolist(), 
                  name='Overall Progress Rate',
                  line=dict(color='#2ca02c', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )

def plot_software_progress_rate(fig, times, software_progress_rates, row, col):
    """Plot software progress rate over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=software_progress_rates.tolist(),
                  name='Software Progress Rate',
                  line=dict(color='#ff7f0e', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    
    # Add vertical line for SC time if available
    results = session_data.get('results')
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[software_progress_rates.min(), software_progress_rates.max()],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_coding_labor_with_compute(fig, times, coding_labors, row, col, secondary_y=False):
    """Plot cognitive output with discounted experiment compute"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=coding_labors.tolist(),
                  name='Normalized Coding Labor',
                  line=dict(color='#9467bd', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col, secondary_y=False
    )
    
    # Add discounted experiment compute from model results
    results = session_data['results']
    if results and 'discounted_exp_compute' in results:
        fig.add_trace(
            go.Scatter(x=times.tolist(), y=results['discounted_exp_compute'],
                      name='Discounted Experiment Compute',
                      line=dict(color='#2ca02c', width=3),
                      mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
            row=row, col=col, secondary_y=secondary_y
        )
    
    # Add vertical line for SC time if available
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            y_min = min(coding_labors.min(), min(results.get('discounted_exp_compute', [coding_labors.min()])))
            y_max = max(coding_labors.max(), max(results.get('discounted_exp_compute', [coding_labors.max()])))
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[y_min, y_max],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_coding_labor_with_present_resources(fig, times, coding_labors_with_present_resources, row, col):
    """Plot coding labor with present-day human labor baseline and current inference compute"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=coding_labors_with_present_resources.tolist(),
                  name='Coding Labor (present resources)',
                  line=dict(color='#2ca02c', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    # Add vertical line for SC time if available
    results = session_data.get('results')
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[coding_labors_with_present_resources.min(), coding_labors_with_present_resources.max()],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
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

def plot_rate_components(fig, times, progress_rates, training_compute_growth_rates, row, col):
    """Plot rate components comparison"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=progress_rates.tolist(),
                  name='Effective Compute Growth Rate',
                  line=dict(color='#2ca02c', width=2, dash='solid'),
                  mode='lines',
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=training_compute_growth_rates.tolist(),
                  name='Training Compute Growth Rate',
                  line=dict(color='#ff7f0e', width=2, dash='dash'),
                  mode='lines',
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )

def plot_cognitive_components(fig, times, coding_labors, human_labor_contributions, row, col):
    """Plot cognitive output components (AI vs Human)"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=coding_labors.tolist(),
                  name='Humans and AIs',
                  line=dict(color='#1f77b4', width=2),
                  mode='lines',
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=human_labor_contributions.tolist(),
                  name='Humans only',
                  line=dict(color='#ff7f0e', width=2),
                  mode='lines',
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    
    # Add vertical line for SC time if available
    results = session_data.get('results')
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            y_min = min(coding_labors.min(), human_labor_contributions.min())
            y_max = max(coding_labors.max(), human_labor_contributions.max())
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[y_min, y_max],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_research_stock(fig, times, research_stocks, row, col):
    """Plot research stock over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=research_stocks.tolist(),
                  name='Research Stock',
                  line=dict(color='#8c564b', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    
    # Add vertical line for SC time if available
    results = session_data.get('results')
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[research_stocks.min(), research_stocks.max()],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_research_effort(fig, times, research_efforts, row, col):
    """Plot research stock rate over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=research_efforts.tolist(),
                  name='Research Stock Rate',
                  line=dict(color='#e377c2', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    
    # Add vertical line for SC time if available
    results = session_data.get('results')
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[research_efforts.min(), research_efforts.max()],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_experiment_capacity(fig, times, experiment_capacity, row, col):
    """Plot experiment capacity over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=experiment_capacity.tolist(),
                  name='Experiment Capacity',
                  line=dict(color='#2ca02c', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    
    # Add vertical line for SC time if available
    results = session_data.get('results')
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[experiment_capacity.min(), experiment_capacity.max()],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_human_only_progress_rate(fig, times, human_only_progress_rates, row, col):
    """Plot human-only progress rate over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=human_only_progress_rates.tolist(),
                  name='Human-Only Progress Rate',
                  line=dict(color='#ff7f0e', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )

def plot_automation_multiplier(fig, times, automation_multipliers, row, col):
    """Plot automation progress multiplier over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=automation_multipliers.tolist(),
                  name='Automation Multiplier',
                  line=dict(color='#d62728', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    # Add horizontal reference line at y=1
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)

def plot_ai_coding_labor_multiplier(fig, times, ai_coding_labor_multipliers, row, col):
    """Plot AI cognitive output multiplier over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=ai_coding_labor_multipliers.tolist(),
                  name='AI Cognitive Output Multiplier',
                  line=dict(color='#9467bd', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    # Add horizontal reference line at y=1
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)
    
    # Add vertical line for SC time if available
    results = session_data.get('results')
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[ai_coding_labor_multipliers.min(), ai_coding_labor_multipliers.max()],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_ai_coding_labor_mult_ref_present_day(fig, times, ai_coding_labor_mult_ref_present_day, row, col):
    """Plot AI coding labor multiplier referenced to present-day human labor"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=ai_coding_labor_mult_ref_present_day.tolist(),
                  name='AI Coding Labor Multiplier (ref present day)',
                  line=dict(color='#17becf', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    # Add horizontal reference line at y=1
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)

    # Add vertical line for SC time if available
    results = session_data.get('results')
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[ai_coding_labor_mult_ref_present_day.min(), ai_coding_labor_mult_ref_present_day.max()],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_ai_research_stock_multiplier(fig, times, ai_research_stock_multipliers, row, col):
    """Plot AI research stock multiplier over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=ai_research_stock_multipliers.tolist(),
                  name='AI Research Stock Multiplier',
                  line=dict(color='#8c564b', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    # Add horizontal reference line at y=1
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)
    
    # Add vertical line for SC time if available
    results = session_data.get('results')
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[ai_research_stock_multipliers.min(), ai_research_stock_multipliers.max()],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_ai_software_progress_multiplier(fig, times, ai_software_progress_multipliers, row, col):
    """Plot AI software progress multiplier over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=ai_software_progress_multipliers.tolist(),
                  name='AI Software Progress Multiplier',
                  line=dict(color='#ff7f0e', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    # Add horizontal reference line at y=1
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)
    
    # Add vertical line for SC time if available
    results = session_data.get('results')
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[ai_software_progress_multipliers.min(), ai_software_progress_multipliers.max()],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_ai_overall_progress_multiplier(fig, times, ai_overall_progress_multipliers, row, col):
    """Plot AI overall progress multiplier over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=ai_overall_progress_multipliers.tolist(),
                  name='AI Overall Progress Multiplier',
                  line=dict(color='#2ca02c', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    # Add horizontal reference line at y=1
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)
    
    # Add vertical line for SC time if available
    results = session_data.get('results')
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[ai_overall_progress_multipliers.min(), ai_overall_progress_multipliers.max()],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_all_ai_multipliers(fig, times, ai_coding_labor_multipliers, ai_research_stock_multipliers, 
                           ai_software_progress_multipliers, ai_overall_progress_multipliers, row, col):
    """Plot all AI multipliers on the same chart for comparison"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=ai_coding_labor_multipliers.tolist(),
                  name='Cognitive Output',
                  line=dict(color='#9467bd', width=2),
                  mode='lines',
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=ai_research_stock_multipliers.tolist(),
                  name='Research Stock',
                  line=dict(color='#8c564b', width=2),
                  mode='lines',
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=ai_software_progress_multipliers.tolist(),
                  name='Software Progress',
                  line=dict(color='#ff7f0e', width=2),
                  mode='lines',
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=ai_overall_progress_multipliers.tolist(),
                  name='Overall Progress',
                  line=dict(color='#2ca02c', width=2),
                  mode='lines',
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    # Add horizontal reference line at y=1
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)

def plot_ai_sw_progress_mult_ref_present_day(fig, times, ai_sw_progress_mult_ref_present_day, row, col):
    """Plot AI software progress multiplier referenced to present-day baseline"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=ai_sw_progress_mult_ref_present_day.tolist(),
                  name='AI Software Progress Multiplier (ref present day)',
                  line=dict(color='#bcbd22', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    # Add horizontal reference line at y=1
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)
    
    # Add vertical line for SC time if available
    results = session_data.get('results')
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[ai_sw_progress_mult_ref_present_day.min(), ai_sw_progress_mult_ref_present_day.max()],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_ai_research_taste(fig, times, ai_research_taste, row, col):
    """Plot AI research taste over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=ai_research_taste.tolist(),
                  name='AI Research Taste',
                  line=dict(color='#1f77b4', width=3),
                  mode='lines+markers', marker=dict(size=4, color='#1f77b4')),
        row=row, col=col
    )
    
    # Add vertical line for SC time if available
    results = session_data.get('results')
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[ai_research_taste.min(), ai_research_taste.max()],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_ai_research_taste_sd(fig, times, ai_research_taste_sd, row, col):
    """Plot AI research taste level in human-range standard deviations over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=ai_research_taste_sd.tolist(),
                  name='AI Research Taste (SD)',
                  line=dict(color='#ff7f0e', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    # Add horizontal reference line at y=0 (mean of human distribution)
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)
    
    # Add vertical line for SC time if available
    results = session_data.get('results')
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[ai_research_taste_sd.min(), ai_research_taste_sd.max()],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_ai_research_taste_quantile(fig, times, ai_research_taste_quantile, row, col):
    """Plot AI research taste quantile in human distribution over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=ai_research_taste_quantile.tolist(),
                  name='AI Research Taste (Quantile)',
                  line=dict(color='#e377c2', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    # Add horizontal reference lines at key quantiles
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)  # Median
    fig.add_hline(y=0.9, line_dash="dot", line_color="red", opacity=0.3, row=row, col=col)   # 90th percentile
    fig.add_hline(y=0.95, line_dash="dot", line_color="red", opacity=0.5, row=row, col=col)  # 95th percentile
    
    # Add vertical line for SC time if available
    results = session_data.get('results')
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[ai_research_taste_quantile.min(), ai_research_taste_quantile.max()],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_aggregate_research_taste(fig, times, aggregate_research_taste, row, col):
    """Plot aggregate research taste over time"""
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=aggregate_research_taste.tolist(),
                  name='Aggregate Research Taste',
                  line=dict(color='#bcbd22', width=3),
                  mode='lines+markers', marker=dict(size=4),
                  customdata=[format_decimal_year_to_month_year(t) for t in times],
                  hovertemplate='Year: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>'),
        row=row, col=col
    )
    # Add horizontal reference line at y=1 (no research taste enhancement)
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)
    
    # Add vertical line for SC time if available
    results = session_data.get('results')
    if results and results.get('sc_time') is not None:
        sc_time = results['sc_time']
        sc_progress = results.get('sc_progress_level')
        if sc_time >= times.min() and sc_time <= times.max():
            fig.add_trace(
                go.Scatter(x=[sc_time, sc_time], 
                          y=[aggregate_research_taste.min(), aggregate_research_taste.max()],
                          name='ACD-AI Time',
                          line=dict(color='#d62728', width=2, dash='dash'),
                          mode='lines',
                          hovertemplate=f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<br>ACD-AI Progress: {sc_progress:.3f}<extra></extra>' if sc_progress else f'ACD-AI Time: {format_decimal_year_to_month_year(sc_time)}<extra></extra>'),
                row=row, col=col
            )

def plot_ai_vs_aggregate_research_taste(fig, ai_research_taste, aggregate_research_taste, row, col):
    """Plot aggregate research taste vs AI research taste as a scatter plot"""
    fig.add_trace(
        go.Scatter(x=ai_research_taste.tolist(), y=aggregate_research_taste.tolist(),
                  name='Aggregate vs AI Research Taste',
                  mode='markers',
                  marker=dict(color='#d62728', size=6, opacity=0.7)),
        row=row, col=col
    )

def format_time_duration(minutes):
    """Convert minutes to appropriate time unit string"""
    if minutes < 1:
        # For values under 1 minute, show in seconds
        seconds = minutes * 60
        return f"{seconds:.0f} sec"
    elif minutes < 60:
        return f"{minutes:.0f} min"
    elif minutes < 480:  # Less than 8 hours (1 work day)
        hours = minutes / 60
        return f"{hours:.0f} hrs"
    elif minutes < 2400:  # Less than 5 work days (1 work week)
        days = minutes / 480
        return f"{days:.0f} days"
    elif minutes < 10380:  # Less than ~21.6 work days (1 work month)
        weeks = minutes / 2400
        return f"{weeks:.0f} weeks"
    elif minutes < 124560:  # Less than 12 work months (1 work year)
        months = minutes / 10380
        return f"{months:.0f} months"
    else:
        years = minutes / 124560
        return f"{years:.0f} years"

def format_decimal_year_to_month_year(decimal_year):
    """Convert decimal year (e.g., 2025.333) to month-year format (e.g., 'Apr 2025')"""
    if decimal_year is None or not np.isfinite(decimal_year):
        return '—'
    
    year = int(decimal_year)
    year_fraction = decimal_year - year
    
    # Convert fraction of year to month (0-11), but handle edge case where rounding gives 12
    month_index = round(year_fraction * 12)
    
    # Handle edge case: if month_index is 12, it should be January of next year
    if month_index >= 12:
        month_index = 0
        year += 1
    
    # Month abbreviations
    month_abbreviations = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]
    
    return f"{month_abbreviations[month_index]} {year}"

def format_number_mbt(value):
    """Convert numbers to M (million), B (billion), T (trillion) format
    
    Args:
        value: Numeric value to format
        
    Returns:
        String with M/B/T notation (e.g., "1.5M", "2.3B", "4.7T")
    """
    if value == 0:
        return "0"
    
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    
    # Handle very large numbers (quadrillions and beyond)
    if abs_value >= 1e15:
        # Quadrillion and beyond - use scientific notation for extremely large values
        exp = int(np.floor(np.log10(abs_value)))
        mantissa = abs_value / (10 ** exp)
        formatted = f"{sign}{mantissa:.1f}e{exp}"
    elif abs_value >= 1e12:
        # Trillion
        formatted = f"{sign}{abs_value/1e12:.1f}T"
    elif abs_value >= 1e9:
        # Billion
        formatted = f"{sign}{abs_value/1e9:.1f}B"
    elif abs_value >= 1e6:
        # Million
        formatted = f"{sign}{abs_value/1e6:.1f}M"
    elif abs_value >= 1e3:
        # Thousand
        formatted = f"{sign}{abs_value/1e3:.1f}K"
    elif abs_value >= 1:
        # Between 1 and 1000
        if abs_value >= 100:
            formatted = f"{sign}{abs_value:.0f}"
        elif abs_value >= 10:
            formatted = f"{sign}{abs_value:.1f}"
        else:
            formatted = f"{sign}{abs_value:.2f}"
    elif abs_value >= 0.01:
        # Small decimals (0.01 to 1)
        formatted = f"{sign}{abs_value:.2f}"
    elif abs_value >= 0.001:
        # Very small decimals (0.001 to 0.01)
        formatted = f"{sign}{abs_value:.3f}"
    else:
        # Extremely small numbers - use scientific notation
        exp = int(np.floor(np.log10(abs_value)))
        mantissa = abs_value / (10 ** exp)
        formatted = f"{sign}{mantissa:.1f}e{exp}"
    
    # Clean up trailing zeros and decimal points (but not for scientific notation)
    if '.' in formatted and 'e' not in formatted:
        formatted = formatted.rstrip('0').rstrip('.')
        # Re-add the suffix if it was removed
        if not formatted.endswith(('K', 'M', 'B', 'T')) and formatted[-1].isdigit():
            # Need to re-add suffix
            if abs_value >= 1e12:
                formatted += "T"
            elif abs_value >= 1e9:
                formatted += "B"
            elif abs_value >= 1e6:
                formatted += "M"
            elif abs_value >= 1e3:
                formatted += "K"
    
    return formatted

def get_mbt_tick_values_and_labels(y_values):
    """Generate tick values and labels using M/B/T notation for log scale axes
    
    Args:
        y_values: Array of y-axis values to determine appropriate tick range
        
    Returns:
        tuple: (tick_values, tick_labels) for plotly tickmode='array'
    """
    if len(y_values) == 0:
        return [], []
    
    # Filter out non-positive values for log scale
    positive_values = y_values[y_values > 0]
    if len(positive_values) == 0:
        return [], []
    
    min_val = np.min(positive_values)
    max_val = np.max(positive_values)
    
    # Generate tick values on log scale
    min_exp = int(np.floor(np.log10(min_val)))
    max_exp = int(np.ceil(np.log10(max_val)))
    
    tick_values = []
    tick_labels = []
    
    # Generate ticks at powers of 10 and some intermediate values
    for exp in range(min_exp, max_exp + 1):
        base_val = 10 ** exp
        
        # Add the power of 10 if it's within or close to our range
        if base_val >= min_val * 0.1 and base_val <= max_val * 10:
            tick_values.append(base_val)
            tick_labels.append(format_number_mbt(base_val))
            
            # Add intermediate values (2x and 5x) for better granularity
            for multiplier in [2, 5]:
                intermediate_val = base_val * multiplier
                if intermediate_val >= min_val * 0.5 and intermediate_val <= max_val * 2:
                    tick_values.append(intermediate_val)
                    tick_labels.append(format_number_mbt(intermediate_val))
    
    # Sort tick values
    if tick_values:
        sorted_pairs = sorted(zip(tick_values, tick_labels))
        tick_values, tick_labels = zip(*sorted_pairs)
        return list(tick_values), list(tick_labels)
    
    return [], []

def get_generic_mbt_ticks_for_log_axis(min_range=None, max_range=None):
    """Generate generic MBT tick values for log axes when data is not available
    
    Args:
        min_range: Minimum value for the range (default: 1e-3)
        max_range: Maximum value for the range (default: 1e15)
        
    Returns:
        tuple: (tick_values, tick_labels) for plotly tickmode='array'
    """
    if min_range is None:
        min_range = 1e-3
    if max_range is None:
        max_range = 1e15
    
    min_exp = int(np.floor(np.log10(min_range)))
    max_exp = int(np.ceil(np.log10(max_range)))
    
    tick_values = []
    tick_labels = []
    
    # Generate ticks at powers of 10
    for exp in range(min_exp, max_exp + 1):
        base_val = 10 ** exp
        if base_val >= min_range and base_val <= max_range:
            tick_values.append(base_val)
            tick_labels.append(format_number_mbt(base_val))
    
    return tick_values, tick_labels

def get_time_tick_values_and_labels():
    """Generate tick values and labels for time duration y-axis"""
    # Define key time boundaries in minutes (matching the image)
    tick_values = [
        0.033333,  # 2 sec
        0.5,       # 30 sec
        2,         # 2 min
        8,         # 8 min
        30,        # 30 min
        120,       # 2 hrs
        480,       # 8 hrs
        2400,      # 1 week
        10380,     # 1 month  
        41520,    # 4 months
        124560,    # 1 year
        622800,    # 5 years
        2491200,  # 20 years
        12456000,   # 100 years
        49824000,   # 400 years
        199296000,  # 1600 years
        797184000,  # 6400 years
        3188736000,  # 25,600 years
        14947200000,   # 120,000 years
    ]
    
    tick_labels = [
        "2 sec",
        "30 sec",
        "2 min",
        "8 min", 
        "30 min",
        "2 hrs",
        "8 hrs",
        "1 week", 
        "1 month",
        "4 months",
        "1 year",
        "5 years",
        "20 years",
        "100 years",
        "400 years",
        "1,600 years",
        "6,400 years",
        "25,600 years",
        "120,000 years"
    ]
    
    return tick_values, tick_labels

def plot_horizon_lengths(fig, times, horizon_lengths, row, col, metr_data=None, aa_time_horizon_minutes=None):
    """Plot horizon lengths over time with METR benchmark points"""
    # Debug: log horizon length values
    logger.info(f"Horizon lengths range: min={np.min(horizon_lengths):.6f}, max={np.max(horizon_lengths):.6f}, median={np.median(horizon_lengths):.6f}")
    
    # Cap horizon lengths at 1 million minutes to prevent scale distortion
    max_horizon = PARAMETER_BOUNDS['aa_time_horizon_minutes'][1]  # 1 million minutes
    min_horizon = 0.001  # 0.001 minutes
    capped_horizon_lengths = np.clip(horizon_lengths, min_horizon, max_horizon)
    num_capped = np.sum(horizon_lengths > max_horizon)
    if num_capped > 0:
        logger.info(f"Capped {num_capped} horizon length values above {max_horizon} minutes")
    
    # Plot the model trajectory with capped values
    fig.add_trace(
        go.Scatter(x=times.tolist(), y=capped_horizon_lengths.tolist(),
                  name='Model Prediction',
                  line=dict(color='#1f77b4', width=3),
                  mode='lines+markers', marker=dict(size=4, color='#1f77b4'),
                  hovertemplate='Year: %{customdata[1]}<br>Horizon: %{customdata[0]}<extra></extra>',
                  customdata=[[format_time_duration(h), format_decimal_year_to_month_year(t)] for h, t in zip(capped_horizon_lengths, times)]),
        row=row, col=col
    )
    
    # Add METR data points if available
    if metr_data is not None and len(metr_data) > 0:
        # Only plot p80 horizon length for SOTA points
        sota_p80_points = [p for p in metr_data if p.get('is_sota', False) and p.get('p80_horizon_length') is not None]
        if sota_p80_points:
            sota_p80_times = [p['decimal_year'] for p in sota_p80_points]
            sota_p80 = [p['p80_horizon_length'] for p in sota_p80_points]
            sota_p80_labels = [p['model_name'] for p in sota_p80_points]
            sota_p80_formatted = [format_time_duration(h) for h in sota_p80]
            
            # Create color and symbol mapping for different model types
            model_colors = {
                'gpt-3.5': '#003000',  # Dark green for GPT models
                'gpt-4': '#003000',    # Dark green for GPT models 
                'claude': '#832000',   # Dark red/brown for Claude
                'o1': '#003000',       # Dark green for O1 (GPT family)
                'agent': '#40c040',    # Bright green for agents
                'grok': '#000000',     # Black for X.ai Grok models
                'default': '#215F9A'   # Dark blue for others
            }
            
            model_symbols = {
                'gpt-3.5': 'triangle-up',
                'gpt-4': 'square', 
                'claude': 'diamond',
                'o1': 'cross',
                'agent': 'circle',
                'grok': 'x',
                'default': 'circle'
            }
            
            # Group points by model type for better visualization
            for i, (time, horizon, label, formatted) in enumerate(zip(sota_p80_times, sota_p80, sota_p80_labels, sota_p80_formatted)):
                # Determine model type from label
                model_type = 'default'
                label_lower = label.lower()
                if ('gpt_3_5' in label_lower or 'gpt-3.5' in label_lower or 'gpt3.5' in label_lower or
                    'gpt_4' in label_lower or 'gpt-4' in label_lower or 'gpt4' in label_lower or
                    'gpt_5' in label_lower or 'gpt-5' in label_lower or 'gpt5' in label_lower or
                    'o1' in label_lower or 'o3' in label_lower or 
                    'davinci_002' in label_lower or 'gpt2' in label_lower or 'gpt_2' in label_lower):
                    model_type = 'gpt-4'  # All GPT family models use the same color
                elif 'claude' in label_lower:
                    model_type = 'claude'
                elif 'grok' in label_lower:
                    model_type = 'grok'
                elif 'agent' in label_lower:
                    model_type = 'agent'
                
                fig.add_trace(
                    go.Scatter(x=[time], y=[horizon],
                              name=label,
                              mode='markers',
                              marker=dict(
                                  color=model_colors[model_type], 
                                  size=10, 
                                  symbol=model_symbols[model_type],
                                  line=dict(width=1, color='white')
                              ),
                              text=[label],
                              customdata=[[formatted, format_decimal_year_to_month_year(time)]],
                              hovertemplate='<b>%{text}</b><br>Year: %{customdata[1]}<br>p80 Horizon: %{customdata[0]}<extra></extra>',
                              showlegend=True),
                    row=row, col=col
                )
    
    # Add horizontal dashed line for superhuman coder time horizon
    if aa_time_horizon_minutes is not None:
        sc_formatted = format_time_duration(aa_time_horizon_minutes)
        fig.add_trace(
            go.Scatter(x=[times.min(), times.max()], 
                      y=[aa_time_horizon_minutes, aa_time_horizon_minutes],
                      name='ACD-AI Time Horizon',
                      line=dict(color='#d62728', width=2, dash='dash'),
                      mode='lines',
                      hovertemplate=f'ACD-AI Time Horizon: {sc_formatted}<extra></extra>'),
            row=row, col=col
        )
    
    # Get custom tick values and labels for time formatting
    tick_values, tick_labels = get_time_tick_values_and_labels()
    
    # Set Y-axis range and custom ticks: 0.001 minutes to 1 million minutes
    # fig.update_yaxes(
    #     range=[-3, 6], 
    #     autorange=False, 
    #     tickmode='array',
    #     tickvals=[val for val in tick_values if 0.001 <= val <= 1_000_000],
    #     ticktext=[label for val, label in zip(tick_values, tick_labels) if 0.001 <= val <= 1_000_000],
    #     row=row, col=col
    # )
    # logger.info(f"Set y-axis range to [-3, 6] with custom time formatting for horizon length plot at row={row}, col={col}")

def plot_horizon_lengths_vs_progress(fig, progress_values, horizon_lengths, row, col, metr_data=None, aa_time_horizon_minutes=None, progress_at_sc=None):
    """Plot horizon lengths vs progress with METR benchmark points"""
    # Cap horizon lengths at 1 million minutes to prevent scale distortion
    max_horizon = PARAMETER_BOUNDS['aa_time_horizon_minutes'][1]  # 1 million minutes
    min_horizon = 0.001  # 0.001 minutes
    capped_horizon_lengths = np.clip(horizon_lengths, min_horizon, max_horizon)
    num_capped = np.sum(horizon_lengths > max_horizon)
    if num_capped > 0:
        logger.info(f"Capped {num_capped} horizon length values above {max_horizon} minutes in vs_progress plot")
    
    # Plot the model trajectory with capped values
    fig.add_trace(
        go.Scatter(x=progress_values.tolist(), y=capped_horizon_lengths.tolist(),
                  name='Model Prediction',
                  line=dict(color='#1f77b4', width=3),
                  mode='lines+markers', marker=dict(size=4, color='#1f77b4'),
                  hovertemplate='Progress: %{x:.3f}<br>Horizon: %{customdata}<extra></extra>',
                  customdata=[format_time_duration(h) for h in capped_horizon_lengths]),
        row=row, col=col
    )
    
    # Add METR data points if available
    if metr_data is not None and len(metr_data) > 0:
        # Only plot p80 horizon length for SOTA points
        sota_p80_points = [p for p in metr_data if p.get('is_sota', False) and p.get('p80_horizon_length') is not None]
        if sota_p80_points:
            sota_p80_progress = [p['interpolated_progress'] for p in sota_p80_points]
            sota_p80 = [p['p80_horizon_length'] for p in sota_p80_points]
            sota_p80_labels = [p['model_name'] for p in sota_p80_points]
            sota_p80_formatted = [format_time_duration(h) for h in sota_p80]
            
            # Create color and symbol mapping for different model types
            model_colors = {
                'gpt-3.5': '#003000',  # Dark green for GPT models
                'gpt-4': '#003000',    # Dark green for GPT models 
                'claude': '#832000',   # Dark red/brown for Claude
                'o1': '#003000',       # Dark green for O1 (GPT family)
                'agent': '#40c040',    # Bright green for agents
                'grok': '#000000',     # Black for X.ai Grok models
                'default': '#215F9A'   # Dark blue for others
            }
            
            model_symbols = {
                'gpt-3.5': 'triangle-up',
                'gpt-4': 'square', 
                'claude': 'diamond',
                'o1': 'cross',
                'agent': 'circle',
                'grok': 'x',
                'default': 'circle'
            }
            
            # Group points by model type for better visualization
            for i, (progress, horizon, label, formatted) in enumerate(zip(sota_p80_progress, sota_p80, sota_p80_labels, sota_p80_formatted)):
                # Determine model type from label
                model_type = 'default'
                label_lower = label.lower()
                if ('gpt_3_5' in label_lower or 'gpt-3.5' in label_lower or 'gpt3.5' in label_lower or
                    'gpt_4' in label_lower or 'gpt-4' in label_lower or 'gpt4' in label_lower or
                    'gpt_5' in label_lower or 'gpt-5' in label_lower or 'gpt5' in label_lower or
                    'o1' in label_lower or 'o3' in label_lower or 
                    'davinci_002' in label_lower or 'gpt2' in label_lower or 'gpt_2' in label_lower):
                    model_type = 'gpt-4'  # All GPT family models use the same color
                elif 'claude' in label_lower:
                    model_type = 'claude'
                elif 'grok' in label_lower:
                    model_type = 'grok'
                elif 'agent' in label_lower:
                    model_type = 'agent'
                
                fig.add_trace(
                    go.Scatter(x=[progress], y=[horizon],
                              name=label,
                              mode='markers',
                              marker=dict(
                                  color=model_colors[model_type], 
                                  size=10, 
                                  symbol=model_symbols[model_type],
                                  line=dict(width=1, color='white')
                              ),
                              text=[label],
                              customdata=[formatted],
                              hovertemplate='<b>%{text}</b><br>Progress: %{x:.3f}<br>p80 Horizon: %{customdata}<extra></extra>',
                              showlegend=True),
                    row=row, col=col
                )
    
    # Add horizontal dashed line for superhuman coder time horizon
    if aa_time_horizon_minutes is not None:
        sc_formatted = format_time_duration(aa_time_horizon_minutes)
        fig.add_trace(
            go.Scatter(x=[progress_values.min(), progress_values.max()], 
                      y=[aa_time_horizon_minutes, aa_time_horizon_minutes],
                      name='ACD-AI Time Horizon',
                      line=dict(color='#d62728', width=2, dash='dash'),
                      mode='lines',
                      hovertemplate=f'ACD-AI Time Horizon: {sc_formatted}<extra></extra>'),
            row=row, col=col
        )
    
    # Add vertical dashed line for progress at superhuman coder
    if progress_at_sc is not None:
        fig.add_trace(
            go.Scatter(x=[progress_at_sc, progress_at_sc], 
                      y=[horizon_lengths.min(), horizon_lengths.max()],
                      name='Progress at SC',
                      line=dict(color='#ff7f0e', width=2, dash='dash'),
                      mode='lines',
                      hovertemplate=f'Progress at ACD-AI: {progress_at_sc:.1f}<extra></extra>'),
            row=row, col=col
        )
    
    # Get custom tick values and labels for time formatting
    tick_values, tick_labels = get_time_tick_values_and_labels()
    
    # Set Y-axis range and custom ticks: 0.001 minutes to 1 million minutes
    fig.update_yaxes(
        range=[-3, 6], 
        autorange=False, 
        tickmode='array',
        tickvals=[val for val in tick_values],
        ticktext=[label for val, label in zip(tick_values, tick_labels)],
        row=row, col=col
    )
    logger.info(f"Set y-axis range to [-3, 6] with custom time formatting for horizon length vs progress plot at row={row}, col={col}")

# Tab Configuration
def get_tab_configurations():
    """
    Get the configuration for all tabs and their plots using centralized metadata.
    
    This function now generates tab configurations from the centralized TAB_CONFIGURATIONS
    in model_config.py, providing a true single source of truth for plot organization.
    """
    
    # Mapping of plot function names to actual functions
    plot_function_map = {
        'plot_horizon_lengths': lambda fig, data, r, c: plot_horizon_lengths(fig, data['metrics']['times'], data['metrics']['horizon_lengths'], r, c, data.get('metr_data'), data.get('parameters', {}).get('aa_time_horizon_minutes')),
        'plot_horizon_lengths_vs_progress': lambda fig, data, r, c: plot_horizon_lengths_vs_progress(fig, data['metrics']['progress'], data['metrics']['horizon_lengths'], r, c, data.get('metr_data'), data.get('parameters', {}).get('aa_time_horizon_minutes'), data.get('progress_at_sc')),
        'plot_human_labor': lambda fig, data, r, c: plot_human_labor(fig, data['time_series'].time, data['time_series'].L_HUMAN, r, c),
        'plot_ai_labor': lambda fig, data, r, c: plot_ai_labor(fig, data['time_series'].time, data['time_series'].inference_compute, r, c),
        'plot_experiment_compute': lambda fig, data, r, c: plot_experiment_compute(fig, data['time_series'].time, data['time_series'].experiment_compute, r, c),
        'plot_training_compute_growth_rate': lambda fig, data, r, c: plot_training_compute_growth_rate(fig, data['time_series'].time, data['time_series'].training_compute_growth_rate, r, c),
        'plot_software_efficiency': lambda fig, data, r, c: plot_software_efficiency(fig, data['metrics']['times'], data['metrics']['software_efficiency'], r, c),
        'plot_labor_comparison': lambda fig, data, r, c: plot_labor_comparison(fig, data['time_series'], r, c),
        'plot_compute_comparison': lambda fig, data, r, c: plot_compute_comparison(fig, data['time_series'], r, c),
        'plot_automation_fraction': lambda fig, data, r, c: plot_automation_fraction(fig, data['metrics']['times'], data['metrics']['automation_fraction'], r, c),
        'plot_progress_vs_automation': lambda fig, data, r, c: plot_progress_vs_automation(fig, data['metrics']['progress'], data['metrics']['automation_fraction'], r, c),
        'plot_ai_research_taste': lambda fig, data, r, c: plot_ai_research_taste(fig, data['metrics']['times'], data['metrics']['ai_research_taste'], r, c),
        'plot_ai_research_taste_sd': lambda fig, data, r, c: plot_ai_research_taste_sd(fig, data['metrics']['times'], data['metrics']['ai_research_taste_sd'], r, c),
        'plot_ai_research_taste_quantile': lambda fig, data, r, c: plot_ai_research_taste_quantile(fig, data['metrics']['times'], data['metrics']['ai_research_taste_quantile'], r, c),
        'plot_aggregate_research_taste': lambda fig, data, r, c: plot_aggregate_research_taste(fig, data['metrics']['times'], data['metrics']['aggregate_research_taste'], r, c),
        'plot_ai_vs_aggregate_research_taste': lambda fig, data, r, c: plot_ai_vs_aggregate_research_taste(fig, data['metrics']['ai_research_taste'], data['metrics']['aggregate_research_taste'], r, c),
        'plot_coding_labor_with_compute': lambda fig, data, r, c: plot_coding_labor_with_compute(fig, data['metrics']['times'], data['metrics']['coding_labors'], r, c),
        'plot_coding_labor_with_present_resources': lambda fig, data, r, c: plot_coding_labor_with_present_resources(fig, data['metrics']['times'], data['metrics']['coding_labors_with_present_resources'], r, c),
        'plot_cognitive_components': lambda fig, data, r, c: plot_cognitive_components(fig, data['metrics']['times'], data['metrics']['coding_labors'], data['metrics']['human_labor_contributions'], r, c),
        'plot_ai_coding_labor_multiplier': lambda fig, data, r, c: plot_ai_coding_labor_multiplier(fig, data['metrics']['times'], data['metrics']['ai_coding_labor_multipliers'], r, c),
        'plot_ai_coding_labor_mult_ref_present_day': lambda fig, data, r, c: plot_ai_coding_labor_mult_ref_present_day(fig, data['metrics']['times'], data['metrics']['ai_coding_labor_mult_ref_present_day'], r, c),
        'plot_research_stock': lambda fig, data, r, c: plot_research_stock(fig, data['metrics']['times'], data['metrics']['research_stock'], r, c),
        'plot_research_effort': lambda fig, data, r, c: plot_research_effort(fig, data['metrics']['times'], data['metrics']['research_efforts'], r, c),
        'plot_experiment_capacity': lambda fig, data, r, c: plot_experiment_capacity(fig, data['metrics']['times'], data['metrics']['experiment_capacity'], r, c),
        'plot_software_progress_rate': lambda fig, data, r, c: plot_software_progress_rate(fig, data['metrics']['times'], data['metrics']['software_progress_rates'], r, c),
        'plot_cumulative_progress': lambda fig, data, r, c: plot_cumulative_progress(fig, data['metrics']['times'], data['metrics']['effective_compute'], r, c),
        'plot_progress_rate': lambda fig, data, r, c: plot_progress_rate(fig, data['metrics']['times'], data['metrics']['progress_rates'], r, c),
        'plot_rate_components': lambda fig, data, r, c: plot_rate_components(fig, data['metrics']['times'], data['metrics']['progress_rates'], np.interp(data['metrics']['times'], data['time_series'].time, data['time_series'].training_compute_growth_rate), r, c),
        'plot_ai_research_stock_multiplier': lambda fig, data, r, c: plot_ai_research_stock_multiplier(fig, data['metrics']['times'], data['metrics']['ai_research_stock_multipliers'], r, c),
        'plot_ai_software_progress_multiplier': lambda fig, data, r, c: plot_ai_software_progress_multiplier(fig, data['metrics']['times'], data['metrics']['ai_software_progress_multipliers'], r, c),
        'plot_ai_overall_progress_multiplier': lambda fig, data, r, c: plot_ai_overall_progress_multiplier(fig, data['metrics']['times'], data['metrics']['ai_overall_progress_multipliers'], r, c),
        'plot_ai_sw_progress_mult_ref_present_day': lambda fig, data, r, c: plot_ai_sw_progress_mult_ref_present_day(fig, data['metrics']['times'], data['metrics']['ai_sw_progress_mult_ref_present_day'], r, c),
        'plot_all_ai_multipliers': lambda fig, data, r, c: plot_all_ai_multipliers(fig, data['metrics']['times'], data['metrics']['ai_coding_labor_multipliers'], data['metrics']['ai_research_stock_multipliers'], data['metrics']['ai_software_progress_multipliers'], data['metrics']['ai_overall_progress_multipliers'], r, c),
        'plot_human_only_progress_rate': lambda fig, data, r, c: plot_human_only_progress_rate(fig, data['metrics']['times'], data['metrics']['human_only_progress_rates'], r, c),
        'plot_automation_multiplier': lambda fig, data, r, c: plot_automation_multiplier(fig, data['metrics']['times'], data['metrics']['automation_multipliers'], r, c),
    }
    
    # Generate tab configurations from centralized config
    tabs = []
    for tab_key, tab_config in TAB_CONFIGURATIONS.items():
        plots = []
        for plot_def in tab_config['plots']:
            function_name = plot_def['function']
            row, col = plot_def['position']
            
            if function_name not in plot_function_map:
                logger.warning(f"Unknown plot function: {function_name}")
                continue
                
            plot_config = PlotConfig(
                function_name=function_name,
                plot_func=plot_function_map[function_name],
                row=row,
                col=col
            )
            plots.append(plot_config)
            
        tab = TabConfig(
            tab_id=tab_config['id'],
            tab_name=tab_config['name'], 
            plots=plots,
            rows=tab_config['rows'],
            cols=tab_config['cols'],
            subplot_titles=tab_config.get('subplot_titles'),
            explanation=tab_config.get('explanation')
        )
        tabs.append(tab)
    
    return tabs

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
    
    # Update layout with consistent height per row (aligned with front-end sizing)
    height_per_row = 320  # pixels per row
    total_height = height_per_row * tab_config.rows
    
    fig.update_layout(
        height=total_height,
        autosize=True,  # Allow width to be responsive
        showlegend=False,
        title_text=tab_config.tab_name,
        plot_bgcolor='white',
        margin=dict(t=60, b=40, l=100, r=40),  # Increased left margin for y-axis labels
        template=get_professional_plotly_template(),
    )
    
    # Update axes per subplot (no forced matching; some plots use non-time x-axes)
    update_axes_for_tab(fig, tab_config, data)
    
    return fig

def update_axes_for_tab(fig: go.Figure, tab_config: TabConfig, data: Dict[str, Any]):
    """Update axis labels and scaling for a specific tab using plot configurations"""
    
    # Apply axis configuration from each plot
    for plot_config in tab_config.plots:
        row, col = plot_config.row, plot_config.col
        
        # Update x-axis
        fig.update_xaxes(
            title_text=plot_config.x_axis_title,
            row=row, col=col,
            gridcolor='#ebedf0',
            ticks='outside', tickcolor='#bdbdbd', ticklen=6,
        )
        # Ensure time axes render correctly (we use decimal years, not dates)
        if isinstance(plot_config.x_axis_title, str) and plot_config.x_axis_title.strip().lower() == 'time':
            # Use integer year ticks with adaptive spacing to avoid clutter
            dtick_years = 1
            try:
                # Prefer session time series range
                ts = session_data.get('time_series')
                if ts is not None and hasattr(ts, 'time') and len(ts.time) > 1:
                    tmin = float(np.nanmin(ts.time))
                    tmax = float(np.nanmax(ts.time))
                else:
                    # Fallback to metrics times if present
                    mt = data.get('metrics', {}).get('times')
                    tmin = float(np.nanmin(mt)) if mt is not None and len(mt) > 1 else None
                    tmax = float(np.nanmax(mt)) if mt is not None and len(mt) > 1 else None
                if tmin is not None and tmax is not None and np.isfinite(tmin) and np.isfinite(tmax):
                    span = max(0.0, tmax - tmin)
                    # Use more frequent ticks for Time Horizon Lengths graph
                    if plot_config.title == "Time Horizon Lengths":
                        target_ticks = 20.0  # More frequent ticks for time horizon plot
                    else:
                        target_ticks = 6.0  # Default for other plots
                    dtick_years = max(1, int(round(span / target_ticks)))
            except Exception:
                dtick_years = 2
            # Set x-axis range for specific plots to ensure alignment
            x_range = None
            if plot_config.title == "Time Horizon Lengths":
                if tmin is not None and tmax is not None and np.isfinite(tmin) and np.isfinite(tmax):
                    # Force x-axis to start at 2019 for time horizons plot
                    range_start = 2019
                    # Use end_year from time_range if available, otherwise use tmax
                    time_range = data.get('time_range')
                    range_end = time_range[1] if time_range and len(time_range) >= 2 else tmax
                    x_range = [range_start, range_end]
            elif plot_config.title in ["Components of Effective Compute", "Rate Components"]:
                # Ensure both effective compute graphs have the same x-axis range for proper alignment
                # since one is the derivative of the other
                if tmin is not None and tmax is not None and np.isfinite(tmin) and np.isfinite(tmax):
                    # Use simulation end year from time_range if available, otherwise use tmax
                    time_range = data.get('time_range')
                    range_end = time_range[1] if time_range and len(time_range) >= 2 else tmax
                    x_range = [tmin, range_end]
            
            fig.update_xaxes(
                type='linear', tickformat='d', dtick=dtick_years,
                tickangle=0, automargin=True,
                range=x_range,
                row=row, col=col,
            )
        
        # Update primary y-axis using centralized configuration
        y_axis_type = plot_config.y_axis_type if plot_config.y_axis_type != "linear" else None
        y_axis_range_mode = 'tozero' if plot_config.y_axis_type == "linear" else None
        y_axis_kwargs = dict(
            title_text=plot_config.y_axis_title,
            type=y_axis_type,
            rangemode=y_axis_range_mode,
            row=row, col=col,
            gridcolor='#ebedf0',
            ticks='outside', tickcolor='#bdbdbd', ticklen=6,
            secondary_y=False,
        )
        
        # Handle custom y-axis ranges (e.g., for horizon length plots)
        if hasattr(plot_config, 'y_axis_range') and plot_config.y_axis_range is not None:
            y_axis_kwargs['range'] = plot_config.y_axis_range
            y_axis_kwargs['autorange'] = False
            
        # Handle custom ticks (e.g., for time-formatted axes)
        custom_ticks_applied = False
        if hasattr(plot_config, 'y_axis_custom_ticks') and plot_config.y_axis_custom_ticks:
            tick_values, tick_labels = get_time_tick_values_and_labels()
            y_axis_kwargs.update({
                'tickmode': 'array',
                'tickvals': [val for val in tick_values],
                'ticktext': [label for val, label in zip(tick_values, tick_labels)],
            })
            custom_ticks_applied = True
        
        # Use M/B/T notation on log scale axes (instead of SI notation like giga, tera)
        # But only if custom ticks haven't already been applied
        if plot_config.y_axis_type == 'log' and not custom_ticks_applied:
            # Try to use custom M/B/T formatting for all log plots
            data_keys = plot_config.metadata.get('data_keys', [])
            custom_formatting_applied = False
            
            if data_keys and len(data_keys) >= 2:
                y_data_key = data_keys[1]  # Second key is usually y-axis data
                if y_data_key in data and len(data[y_data_key]) > 0:
                    y_values = np.array(data[y_data_key])
                    # Filter out non-positive values for log scale
                    y_values_positive = y_values[y_values > 0]
                    if len(y_values_positive) > 0:
                        tick_values, tick_labels = get_mbt_tick_values_and_labels(y_values_positive)
                        if tick_values:  # Only apply if we got valid tick values
                            y_axis_kwargs.update({
                                'tickmode': 'array',
                                'tickvals': tick_values,
                                'ticktext': tick_labels,
                            })
                            custom_formatting_applied = True
            
            # If custom formatting wasn't applied, use generic MBT formatting as fallback
            if not custom_formatting_applied:
                # Use generic MBT tick formatting for all log plots
                tick_values, tick_labels = get_generic_mbt_ticks_for_log_axis()
                if tick_values:
                    y_axis_kwargs.update({
                        'tickmode': 'array',
                        'tickvals': tick_values,
                        'ticktext': tick_labels,
                    })
                    custom_formatting_applied = True
                
                # Final fallback if generic formatting also fails
                if not custom_formatting_applied:
                    # Use power notation without SI prefixes, avoiding scientific notation
                    y_axis_kwargs.update(dict(
                        exponentformat='power',  # Use 10^n format instead of 1e+n
                        tickformat='.0f'        # No decimal places, no SI notation
                    ))
            
        fig.update_yaxes(**y_axis_kwargs)
        
        # Update secondary y-axis if present
        if plot_config.secondary_y and plot_config.y_axis_secondary_title:
            secondary_type = plot_config.y_axis_secondary_type if plot_config.y_axis_secondary_type != "linear" else None
            secondary_range_mode = 'tozero' if plot_config.y_axis_secondary_type == "linear" else None
            secondary_kwargs = dict(
                title_text=plot_config.y_axis_secondary_title,
                type=secondary_type,
                rangemode=secondary_range_mode,
                row=row, col=col,
                gridcolor='#ebedf0',
                ticks='outside', tickcolor='#bdbdbd', ticklen=6,
                secondary_y=True,
            )
            if plot_config.y_axis_secondary_type == 'log':
                # Use MBT notation for secondary axis too
                tick_values, tick_labels = get_generic_mbt_ticks_for_log_axis()
                if tick_values:
                    secondary_kwargs.update({
                        'tickmode': 'array',
                        'tickvals': tick_values,
                        'ticktext': tick_labels,
                    })
                else:
                    # Fallback to power notation for secondary axis, avoiding scientific notation
                    secondary_kwargs.update(dict(exponentformat='power', tickformat='.0f'))
            fig.update_yaxes(**secondary_kwargs)

    # Apply consistent trace styling for scatter line charts (excluding METR data points)
    try:
        fig.update_traces(line=dict(width=2.5), marker=dict(size=3), selector=dict(type='scatter', mode='lines+markers'))
    except Exception:
        pass

def create_multi_tab_dashboard(metrics: Dict[str, Any], time_range: List[float] = None) -> Dict[str, go.Figure]:
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
    
    # Special handling for AI multipliers - default to 1.0 instead of 0
    ai_multiplier_keys = [
        'ai_coding_labor_multipliers',
        'ai_research_stock_multipliers', 
        'ai_software_progress_multipliers',
        'ai_sw_progress_mult_ref_present_day',
        'ai_overall_progress_multipliers'
    ]
    
    for key in ai_multiplier_keys:
        if key in cleaned_metrics:
            cleaned_metrics[key] = np.where(
                np.isfinite(cleaned_metrics[key]),
                cleaned_metrics[key],
                1.0
            )
    
    # Process METR data for plotting
    metr_data = process_metr_data()
    
    # Prepare data for plotting
    plot_data = {
        'time_series': session_data['time_series'],
        'metrics': cleaned_metrics,
        'metr_data': metr_data,
        'parameters': params_to_dict(session_data.get('current_params', Parameters())) if session_data.get('current_params') else {},
        'time_range': time_range
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
            inference_compute = np.array([float(row['inference_compute']) for row in data])
            experiment_compute = np.array([float(row['experiment_compute']) for row in data])
            training_compute_growth_rate = np.array([float(row['training_compute_growth_rate']) for row in data])
            
            logger.info(f"Loaded time series data: {len(data)} points from {time[0]} to {time[-1]}")
            return TimeSeriesData(time, L_HUMAN, inference_compute, experiment_compute, training_compute_growth_rate)
        else:
            logger.warning("input_data.csv not found, falling back to synthetic data")
            raise FileNotFoundError("input_data.csv not found")
            
    except Exception as e:
        logger.warning(f"Error loading input_data.csv: {e}, falling back to synthetic data")
        # Fallback to synthetic data if CSV loading fails
        time = np.linspace(2029, 2030, 12)
        L_HUMAN = np.ones_like(time) * 1e6
        inference_compute = np.logspace(3, 8, len(time))
        experiment_compute = np.logspace(6, 10, len(time))  # Use exponential growth as fallback
        training_compute_growth_rate = np.logspace(6, 10, len(time))
        
        logger.info("Using synthetic fallback data")
        return TimeSeriesData(time, L_HUMAN, inference_compute, experiment_compute, training_compute_growth_rate)

def create_default_parameters():
    """Create default model parameters"""
    return Parameters()

def load_time_series_from_csv_path(csv_path):
    """Load time series data from a given CSV file path"""
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)
    time = np.array([float(row['time']) for row in data])
    L_HUMAN = np.array([float(row['L_HUMAN']) for row in data])
    inference_compute = np.array([float(row['inference_compute']) for row in data])
    experiment_compute = np.array([float(row['experiment_compute']) for row in data])
    training_compute_growth_rate = np.array([float(row['training_compute_growth_rate']) for row in data])
    return TimeSeriesData(time, L_HUMAN, inference_compute, experiment_compute, training_compute_growth_rate)

def time_series_to_dict(data: TimeSeriesData):
    """Convert TimeSeriesData to dictionary for JSON serialization"""
    return {
        'time': data.time.tolist(),
        'L_HUMAN': data.L_HUMAN.tolist(),
        'inference_compute': data.inference_compute.tolist(),
        'experiment_compute': data.experiment_compute.tolist(),
        'training_compute_growth_rate': data.training_compute_growth_rate.tolist()
    }

def params_to_dict(params: Parameters):
    """Convert parameters to dictionary for JSON serialization""" 
    param_dict = {
        'rho_coding_labor': params.rho_coding_labor,
        'rho_experiment_capacity': params.rho_experiment_capacity,
        'alpha_experiment_capacity': params.alpha_experiment_capacity,
        'r_software': params.r_software,
        'automation_fraction_at_coding_automation_anchor': params.automation_fraction_at_coding_automation_anchor,
        'automation_interp_type': getattr(params, 'automation_interp_type', 'exponential'),

        'coding_labor_normalization': params.coding_labor_normalization,
        'coding_labor_mode': getattr(params, 'coding_labor_mode', 'simple_ces'),
        'coding_automation_efficiency_slope': getattr(params, 'coding_automation_efficiency_slope', 1.0),
        'optimal_ces_eta_init': getattr(params, 'optimal_ces_eta_init', 1.0),
        'optimal_ces_grid_size': getattr(params, 'optimal_ces_grid_size', 4096),
        'experiment_compute_exponent': params.experiment_compute_exponent,
            'ai_research_taste_at_coding_automation_anchor': params.ai_research_taste_at_coding_automation_anchor,
            'ai_research_taste_at_coding_automation_anchor_sd': params.ai_research_taste_at_coding_automation_anchor_sd,
            'ai_research_taste_slope': params.ai_research_taste_slope,
        'taste_schedule_type': params.taste_schedule_type,
        'progress_at_sc': params.progress_at_sc,
        'aa_time_horizon_minutes': params.aa_time_horizon_minutes,
        'pre_gap_aa_time_horizon': getattr(params, 'pre_gap_aa_time_horizon', None),
        'horizon_extrapolation_type': params.horizon_extrapolation_type,
        # Manual horizon fitting parameters
        'present_day': params.present_day,
        'present_horizon': params.present_horizon,
        'present_doubling_time': params.present_doubling_time,
        'doubling_difficulty_growth_factor': params.doubling_difficulty_growth_factor,
        # Baseline Annual Compute Multiplier
        'baseline_annual_compute_multiplier': params.baseline_annual_compute_multiplier,
        # Gap mode selection
        'include_gap': getattr(params, 'include_gap', 'no gap'),
        'gap_years': float(getattr(params, 'gap_years', 0.0)),
        # Research taste distribution parameter
        'median_to_top_taste_multiplier': getattr(params, 'median_to_top_taste_multiplier', 3.25),
        'top_percentile': getattr(params, 'top_percentile', 0.999),
    }
    
    # Add calculated SC information if available from the current session
    if session_data.get('results'):
        results = session_data['results']
        if results.get('sc_time') is not None:
            param_dict['calculated_sc_time'] = float(results['sc_time'])
        if results.get('sc_progress_level') is not None:
            param_dict['calculated_sc_progress_level'] = float(results['sc_progress_level'])
            
    return param_dict



@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/compute', methods=['POST'])
def compute_model():
    """Compute model with given parameters"""
    # try:
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
    initial_progress = data.get('initial_progress', 0.0)
    model = ProgressModel(params, time_series)
    
    times, progress_values, research_stock_values = model.compute_progress_trajectory(
            time_range, initial_progress
        )
    
    # All metrics are now computed by ProgressModel - use them directly
    all_metrics = model.results
    
    # Store results (including auxiliary metrics for potential export)
    session_data['current_params'] = params
    session_data['results'] = all_metrics
    # Persist current direct/derived mode from parameters for diagnostics
    try:
        session_data['use_direct_ces_parameters'] = bool(getattr(params, 'direct_input_exp_cap_ces_params', False))
    except Exception:
        session_data['use_direct_ces_parameters'] = False
    
    # Create multi-tab dashboard using the calculated metrics
    figures = create_multi_tab_dashboard(all_metrics, time_range)
    
    # Convert figures to JSON format
    plots = {}
    for tab_id, fig in figures.items():
        plots[tab_id] = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
    
    # Get tab metadata including row information for proper sizing
    tab_configs = get_tab_configurations()
    tabs_info = [{'id': tab.tab_id, 'name': tab.tab_name, 'rows': tab.rows, 'explanation': tab.explanation} for tab in tab_configs]
    
    # Prepare summary focusing on SC metrics
    summary = {
        'time_range': time_range
    }
    
    # Add anchor taste slope metrics if available
    if model.results.get('ai_research_taste_slope_per_anchor_progress_year') is not None:
        summary['ai_research_taste_slope_per_anchor_progress_year'] = float(model.results['ai_research_taste_slope_per_anchor_progress_year'])
    if model.results.get('ai_research_taste_slope_per_effective_oom') is not None:
        summary['ai_research_taste_slope_per_effective_oom'] = float(model.results['ai_research_taste_slope_per_effective_oom'])

    # Add anchor progress rate if available
    if model.results.get('anchor_progress_rate') is not None:
        summary['anchor_progress_rate'] = float(model.results['anchor_progress_rate'])

    # Add r_software (calibrated value) if available
    if model.results.get('r_software') is not None:
        summary['r_software'] = float(model.results['r_software'])
    
    # Add beta_software (inverse of r_software) if available
    if model.results.get('beta_software') is not None:
        summary['beta_software'] = float(model.results['beta_software'])
    
    # Add top taste percentile metrics if available
    if model.results.get('top_taste_percentile') is not None:
        summary['top_taste_percentile'] = float(model.results['top_taste_percentile'])
    if model.results.get('top_taste_num_sds') is not None:
        summary['top_taste_num_sds'] = float(model.results['top_taste_num_sds'])
    if model.results.get('f_multiplier_per_sd') is not None:
        summary['f_multiplier_per_sd'] = float(model.results['f_multiplier_per_sd'])
    if model.results.get('slope_times_log_f') is not None:
        summary['slope_times_log_f'] = float(model.results['slope_times_log_f'])

    # Add instantaneous anchor doubling time (years) if available
    if model.results.get('instantaneous_anchor_doubling_time_years') is not None:
        summary['instantaneous_anchor_doubling_time_years'] = float(model.results['instantaneous_anchor_doubling_time_years'])

    # Add SC timing information if available
    if model.results.get('sc_progress_level') is not None and model.results.get('sc_sw_multiplier') is not None:
        summary['sc_time'] = float(model.results['sc_time'])
        summary['sc_progress_level'] = float(model.results['sc_progress_level'])
        summary['sc_sw_multiplier'] = float(model.results['sc_sw_multiplier']) 
        logger.info(f"SC time: {summary['sc_time']}, SC progress level: {summary['sc_progress_level']}, SC SW multiplier: {summary['sc_sw_multiplier']}")
    # Add AI2027 SC time if computed
    if model.results.get('ai2027_sc_time') is not None:
        try:
            summary['ai2027_sc_time'] = float(model.results.get('ai2027_sc_time'))
        except Exception:
            summary['ai2027_sc_time'] = None
    return jsonify({
        'success': True,
        'plots': plots,
        'tabs': tabs_info,
        'summary': summary,
        'milestones': model.results['milestones'],
        # Surface computed exp capacity params so UI can display when using pseudoparameters
        'exp_capacity_params': {
            'rho': float(all_metrics.get('exp_capacity_params', {}).get('rho')) if all_metrics.get('exp_capacity_params') else None,
            'alpha': float(all_metrics.get('exp_capacity_params', {}).get('alpha')) if all_metrics.get('exp_capacity_params') else None,
            'experiment_compute_exponent': float(all_metrics.get('exp_capacity_params', {}).get('experiment_compute_exponent')) if all_metrics.get('exp_capacity_params') else None,
        }
    })
        
    # except Exception as e:
    #     logger.error(f"Error computing model: {e}")
    #     return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/parameter-config', methods=['GET'])
def get_parameter_config():
    """Get parameter configuration including bounds, defaults, and metadata"""
    try:
        import model_config as cfg
        
        parameter_config = {
            'bounds': cfg.PARAMETER_BOUNDS,
            'defaults': cfg.DEFAULT_PARAMETERS,
            'validation_thresholds': cfg.PARAM_VALIDATION_THRESHOLDS,
            'taste_schedule_types': cfg.TASTE_SCHEDULE_TYPES,
            'taste_slope_defaults': cfg.TASTE_SLOPE_DEFAULTS,
            'horizon_extrapolation_types': cfg.HORIZON_EXTRAPOLATION_TYPES,
            'automation_interp_types': ['exponential', 'linear'],
            'coding_labor_modes': ['simple_ces', 'optimal_ces'],
            'pre_gap_aa_time_horizon': cfg.DEFAULT_PARAMETERS['pre_gap_aa_time_horizon'],
            'descriptions': {
                'rho_coding_labor': {
                    'name': 'Cognitive Elasticity (ρ_cognitive)',
                    'description': 'Elasticity of substitution between AI and human cognitive labor',
                    'units': 'dimensionless'
                },
                'rho_experiment_capacity': {
                    'name': 'Progress Elasticity (ρ_progress)', 
                    'description': 'Elasticity of substitution in progress production function',
                    'units': 'dimensionless'
                },
                'alpha_experiment_capacity': {
                    'name': 'Compute Weight (α)',
                    'description': 'Weight of compute vs cognitive output in progress production',
                    'units': 'dimensionless'
                },
                'r_software': {
                    'name': 'Software Scale',
                    'description': 'Scale factor for software progress',
                    'units': 'dimensionless'
                },
                'automation_fraction_at_coding_automation_anchor': {
                    'name': 'Max Automation',
                    'description': 'Automation fraction when AI reaches superhuman coding ability',
                    'units': 'fraction'
                },
                'automation_interp_type': {
                    'name': 'Automation Interpolation Type',
                    'description': 'Method for interpolating automation between anchors',
                    'units': 'categorical'
                },
                
                'swe_multiplier_at_present_day': {
                    'name': 'SWE Multiplier at Anchor Time',
                    'description': 'Software engineering productivity multiplier at the anchor time',
                    'units': 'dimensionless'
                },
                'coding_labor_normalization': {
                    'name': 'Cognitive Output Normalization',
                    'description': 'Normalization factor for cognitive output',
                    'units': 'dimensionless'
                },
                'coding_labor_mode': {
                    'name': 'Coding Labor Mode',
                    'description': 'Select between simple CES and optimal frontier CES',
                    'units': 'categorical'
                },
                'coding_automation_efficiency_slope': {
                    'name': 'Automation Efficiency Slope',
                    'description': 'Slope of automation multiplier above threshold (η ∝ E^θ)',
                    'units': 'dimensionless'
                },
                'optimal_ces_eta_init': {
                    'name': 'Optimal CES η_init',
                    'description': 'Automation multiplier at the threshold',
                    'units': 'dimensionless'
                },
                'optimal_ces_grid_size': {
                    'name': 'Optimal CES Grid Size',
                    'description': 'Number of grid points for frontier precompute',
                    'units': 'count'
                },
                'experiment_compute_exponent': {
                    'name': 'Experiment Compute Discounting (ζ)',
                    'description': 'Diminishing returns factor for experiment compute',
                    'units': 'dimensionless'
                },

                'ai_research_taste_at_coding_automation_anchor': {
                    'name': 'Max AI Research Taste',
                    'description': 'AI research taste when AI reaches superhuman coding ability',
                    'units': 'fraction'
                },
                'ai_research_taste_at_coding_automation_anchor_sd': {
                    'name': 'Max AI Research Taste (SD)',
                    'description': 'AI research taste at superhuman coder specified in human-range standard deviations',
                    'units': 'SD'
                },
                
                'ai_research_taste_slope': {
                    'name': 'AI Research Taste Slope',
                    'description': 'Steepness of AI research taste curve',
                    'units': 'dimensionless'
                },
                'taste_schedule_type': {
                    'name': 'AI Research Taste Schedule Type',
                    'description': 'Unit convention for SD-based schedule',
                    'units': 'categorical'
                },
                'progress_at_sc': {
                    'name': 'Progress at ACD-AI',
                    'description': 'Progress level where AI reaches superhuman coding ability (exponential mode)',
                    'units': 'dimensionless'
                },
                'aa_time_horizon_minutes': {
                    'name': 'Time Horizon to ACD-AI',
                    'description': 'Time horizon length corresponding to superhuman coder achievement',
                    'units': 'minutes'
                },
                'include_gap': {
                    'name': 'Include Gap',
                    'description': 'Choose whether to include a gap after pre-gap SC horizon',
                    'units': 'choice (gap | no gap)'
                },
                'gap_years': {
                    'name': 'Gap Size (YEAR-progress-years)',
                    'description': 'Additional gap after saturation horizon to reach SC, measured in YEAR-progress-years (converted using anchor_progress_rate). UI displays YEAR based on Anchor Time.',
                    'units': 'YEAR-progress-years'
                },
                'horizon_extrapolation_type': {
                    'name': 'Horizon Extrapolation Type',
                    'description': 'Method for extrapolating progress beyond the time horizon',
                    'units': 'categorical'
                },
                'parallel_penalty': {
                    'name': 'Parallel Penalty',
                    'description': 'Penalty applied to parallel coding labor contribution in experiment capacity calculations',
                    'units': 'dimensionless'
                },
                # Manual horizon fitting parameters
                'present_day': {
                    'name': 'Anchor Time',
                    'description': 'Reference time point for manual horizon fitting',
                    'units': 'year'
                },
                'present_horizon': {
                    'name': 'Anchor Horizon',
                    'description': 'Time horizon length at the anchor time (leave empty for auto-fit)',
                    'units': 'minutes'
                },
                'present_doubling_time': {
                    'name': 'Anchor Doubling Time',
                    'description': 'Doubling time parameter at the anchor point (leave empty for auto-fit)',
                    'units': 'progress units'
                },
                'doubling_difficulty_growth_factor': {
                    'name': 'Doubling Difficulty Growth Factor',
                    'description': 'Rate of growth for doubling difficulty (1 - decay rate, leave empty for auto-fit)',
                    'units': 'dimensionless'
                },
                'baseline_annual_compute_multiplier': {
                    'name': 'Baseline Annual Compute Multiplier',
                    'description': 'Annual multiplier for baseline compute growth (effective compute = multiplier^progress)',
                    'units': 'dimensionless'
                },
                # exp capacity pseudoparameters
                'inf_labor_asymptote': {
                    'name': 'Infinite Labor Asymptote',
                    'description': 'Asymptotic experiment capacity as labor → ∞ (holding compute finite)',
                    'units': 'capacity units'
                },
                'inf_compute_asymptote': {
                    'name': 'Infinite Compute Asymptote',
                    'description': 'Asymptotic experiment capacity as compute → ∞ (holding labor finite)',
                    'units': 'capacity units'
                },
                'labor_anchor_exp_cap': {
                    'name': 'Labor Anchor Experiment Capacity',
                    'description': 'Experiment capacity at the labor anchor used to infer CES parameters',
                    'units': 'capacity units'
                },
                'compute_anchor_exp_cap': {
                    'name': 'Compute Anchor Experiment Capacity',
                    'description': 'Experiment capacity at the compute anchor used to infer CES parameters (direct form, deprecated in UI)',
                    'units': 'capacity units'
                },
                'inv_compute_anchor_exp_cap': {
                    'name': 'Exp. Cap. Reduction with 0.1x compute',
                    'description': 'Inverse form: capacity at 0.1x compute relative to baseline (i.e., reduction factor). The model uses 1 / value internally for the direct anchor.',
                    'units': 'x relative to baseline'
                },
                'median_to_top_taste_multiplier': {
                    'name': 'Median to Top Taste Multiplier',
                    'description': 'Ratio of top percentile researcher taste to median researcher taste',
                    'units': 'ratio'
                },
                'top_percentile': {
                    'name': 'Top ResearcherPercentile',
                    'description': 'Quantile for the "top" researcher considered in the human range (e.g., 0.999 = 99.9th percentile)',
                    'units': 'quantile (0-1)'
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
        inference_compute = np.array([float(row['inference_compute']) for row in data])
        experiment_compute = np.array([float(row['experiment_compute']) for row in data])
        training_compute_growth_rate = np.array([float(row['training_compute_growth_rate']) for row in data])
        
        time_series = TimeSeriesData(time, L_HUMAN, inference_compute, experiment_compute, training_compute_growth_rate)
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

@app.route('/api/select-data', methods=['POST'])
def select_data():
    """Select a built-in dataset by actor and update session time series"""
    try:
        data = request.json or {}
        dataset = data.get('dataset', 'default')

        import os
        base_dir = os.path.dirname(__file__)
        inputs_dir = os.path.join(base_dir, 'inputs')

        # Map datasets to absolute file paths. Note: the "default" CSV lives at project root.
        dataset_to_file = {
            'default': os.path.join(base_dir, 'input_data.csv'),
            'updated_default': os.path.join(inputs_dir, 'updated_default.csv'),
            'truncated_default': os.path.join(inputs_dir, 'truncated_default.csv'),
            'pretrain_russia': os.path.join(inputs_dir, 'pretrain_russia.csv'),
            'pretrain_us_black_site': os.path.join(inputs_dir, 'pretrain_us_black_site.csv'),
            'pretrain_cn_black_site': os.path.join(inputs_dir, 'pretrain_cn_black_site.csv'),
            'rl_russia': os.path.join(inputs_dir, 'rl_russia.csv'),
            'rl_us_black_site': os.path.join(inputs_dir, 'rl_us_black_site.csv'),
            'rl_cn_black_site': os.path.join(inputs_dir, 'rl_cn_black_site.csv')
        }

        if dataset not in dataset_to_file:
            return jsonify({'success': False, 'error': f"Unknown dataset: {dataset}"}), 400

        csv_path = dataset_to_file[dataset]

        if not os.path.exists(csv_path):
            return jsonify({'success': False, 'error': f"Dataset file not found: {os.path.basename(csv_path)}"}), 400

        time_series = load_time_series_from_csv_path(csv_path)
        session_data['time_series'] = time_series

        time_array = time_series.time
        return jsonify({
            'success': True,
            'data_summary': {
                'time_range': [float(time_array.min()), float(time_array.max())],
                'data_points': len(time_array),
                'preview': time_series_to_dict(time_series)
            }
        })
    except Exception as e:
        logger.error(f"Error selecting dataset: {e}")
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
        writer.writerow(['time', 'cumulative_progress', 'automation_fraction', 'progress_rate', 'software_progress_rate', 'coding_labor', 'research_stock', 'research_effort', 'experiment_capacity', 'human_only_progress_rate', 'ai_labor_contribution', 'human_labor_contribution'])
        
        # Write metadata header with SC information if available
        if results.get('sc_time') is not None:
            writer.writerow(['# ACD-AI Level Reached at:', f"Time: {results['sc_time']:.4f}", f"Progress Level: {results.get('sc_progress_level', 'N/A'):.4f}" if results.get('sc_progress_level') is not None else 'Progress Level: N/A'])
            writer.writerow([])  # Empty row for separation
        
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
            if 'coding_labors' in results and i < len(results['coding_labors']):
                row.append(results['coding_labors'][i])
            else:
                row.append(0.0)

            if 'research_stock' in results and i < len(results['research_stock']):
                row.append(results['research_stock'][i])
            else:
                row.append(0.0)

            if 'research_efforts' in results and i < len(results['research_efforts']):
                row.append(results['research_efforts'][i])
            else:
                row.append(0.0)

            if 'experiment_capacity' in results and i < len(results['experiment_capacity']):
                row.append(results['experiment_capacity'][i])
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

def process_metr_data():
    """Process METR benchmark data and return structured data for plotting"""
    try:
        # Load benchmark results from YAML
        try:
            with open('benchmark_results.yaml', 'r') as f:
                benchmark_data = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning('benchmark_results.yaml file not found')
            return None
        except Exception as e:
            logger.warning(f'Error reading benchmark_results.yaml: {str(e)}')
            return None
        
        if session_data['results'] is None:
            logger.warning('No model results available for METR data processing')
            return None
            
        results = session_data['results']
        times = np.array(results['times'])
        progress_values = np.array(results['progress'])
        
        # Calculate progress offset so that progress at start_year equals initial_progress
        # Use the first time point and progress value from the computed results
        start_year = times[0]  # First time point in the results
        target_initial_progress = progress_values[0]  # This should be the initial_progress that was used
        
        # Find actual progress at start_year from the computed trajectory
        progress_at_start = np.interp(start_year, times, progress_values)
        
        # Calculate offset to normalize progress so that at start_year, progress = target_initial_progress
        progress_offset = target_initial_progress - progress_at_start
        
        # Apply offset to all progress values
        adjusted_progress_values = progress_values + progress_offset
        
        # Process METR data points
        metr_points = []
        
        for model_name, model_info in benchmark_data['results'].items():
            # Convert release date to decimal year
            release_date_obj = model_info['release_date']
            try:
                # Handle both string and date objects
                if isinstance(release_date_obj, str):
                    # Parse date in YYYY-MM-DD format
                    release_date = datetime.strptime(release_date_obj, '%Y-%m-%d').date()
                    release_date_str = release_date_obj
                else:
                    # Already a date object (PyYAML auto-parsed it)
                    release_date = release_date_obj
                    release_date_str = release_date.strftime('%Y-%m-%d')
                
                decimal_year = release_date.year + (release_date.timetuple().tm_yday - 1) / 365.25
            except (ValueError, AttributeError) as e:
                logger.warning(f"Could not parse release date for {model_name}: {release_date_obj} ({e})")
                continue
            
            # Interpolate progress value at the release date using adjusted progress
            if decimal_year >= times.min() and decimal_year <= times.max():
                interpolated_progress = np.interp(decimal_year, times, adjusted_progress_values)
            elif decimal_year < times.min():
                # If release date is before our time series, use the first adjusted progress value
                interpolated_progress = adjusted_progress_values[0]
            else:
                # If release date is after our time series, use the last adjusted progress value
                interpolated_progress = adjusted_progress_values[-1]
            
            # Process each agent configuration for this model
            for agent_name, agent_data in model_info['agents'].items():
                point = {
                    'model_name': model_name,
                    'agent_configuration': agent_name,
                    'release_date': release_date_str,
                    'decimal_year': decimal_year,
                    'interpolated_progress': interpolated_progress,
                    'is_sota': agent_data.get('is_sota', False)
                }
                
                # Add horizon length data if available
                if 'p50_horizon_length' in agent_data:
                    p50_data = agent_data['p50_horizon_length']
                    point['p50_horizon_length'] = p50_data.get('estimate')
                    point['p50_horizon_length_ci_low'] = p50_data.get('ci_low')
                    point['p50_horizon_length_ci_high'] = p50_data.get('ci_high')
                
                if 'p80_horizon_length' in agent_data:
                    p80_data = agent_data['p80_horizon_length']
                    point['p80_horizon_length'] = p80_data.get('estimate')
                    point['p80_horizon_length_ci_low'] = p80_data.get('ci_low')
                    point['p80_horizon_length_ci_high'] = p80_data.get('ci_high')
                
                if 'average_score' in agent_data:
                    avg_score_data = agent_data['average_score']
                    if isinstance(avg_score_data, dict):
                        point['average_score'] = avg_score_data.get('estimate')
                        point['average_score_ci_low'] = avg_score_data.get('ci_low')
                        point['average_score_ci_high'] = avg_score_data.get('ci_high')
                    else:
                        # Handle case where average_score is just a number
                        point['average_score'] = avg_score_data
                
                metr_points.append(point)
        
        return metr_points
        
    except Exception as e:
        logger.warning(f"Error processing METR data: {e}")
        return None

@app.route('/api/export-metr-data')
def export_metr_data():
    """Export progress-adjusted METR benchmark data as CSV"""
    try:
        metr_points = process_metr_data()
        if metr_points is None:
            return jsonify({'success': False, 'error': 'No model results to interpolate progress values. Please run the model first.'}), 400
        
        # Convert processed METR points to CSV rows
        csv_rows = []
        
        for point in metr_points:
            row = {
                'model_name': point['model_name'],
                'agent_configuration': point['agent_configuration'],
                'release_date_original': point['release_date'],
                'release_date_decimal_year': point['decimal_year'],
                'is_sota': 1 if point['is_sota'] else 0,
                'interpolated_progress': point['interpolated_progress']
            }
            
            # Add performance metrics if available
            if 'p50_horizon_length' in point and point['p50_horizon_length'] is not None:
                row.update({
                    'p50_horizon_length_estimate': point['p50_horizon_length'],
                    'p50_horizon_length_ci_low': point.get('p50_horizon_length_ci_low'),
                    'p50_horizon_length_ci_high': point.get('p50_horizon_length_ci_high')
                })
            
            if 'p80_horizon_length' in point and point['p80_horizon_length'] is not None:
                row.update({
                    'p80_horizon_length_estimate': point['p80_horizon_length'],
                    'p80_horizon_length_ci_low': point.get('p80_horizon_length_ci_low'),
                    'p80_horizon_length_ci_high': point.get('p80_horizon_length_ci_high')
                })
            
            if 'average_score' in point and point['average_score'] is not None:
                row.update({
                    'average_score_estimate': point['average_score'],
                    'average_score_ci_low': point.get('average_score_ci_low'),
                    'average_score_ci_high': point.get('average_score_ci_high')
                })
            
            csv_rows.append(row)
        
        if not csv_rows:
            return jsonify({'success': False, 'error': 'No valid benchmark data found to export'}), 400
        
        # Create CSV content
        output = io.StringIO()
        
        # Get all possible column names from all rows
        all_columns = set()
        for row in csv_rows:
            all_columns.update(row.keys())
        
        # Define column order for better readability
        preferred_order = [
            'model_name', 'agent_configuration', 'release_date_original', 'release_date_decimal_year', 
            'is_sota', 'interpolated_progress',
            'p50_horizon_length_estimate', 'p80_horizon_length_estimate',
            'p50_horizon_length_ci_low', 'p50_horizon_length_ci_high',
            'p80_horizon_length_ci_low', 'p80_horizon_length_ci_high',
            'average_score_estimate', 'average_score_ci_low', 'average_score_ci_high'
        ]
        
        # Add any remaining columns not in preferred order
        columns = [col for col in preferred_order if col in all_columns]
        remaining_cols = [col for col in all_columns if col not in columns]
        columns.extend(sorted(remaining_cols))
        
        writer = csv.DictWriter(output, fieldnames=columns)
        
        # Write header
        writer.writeheader()
        
        # Write data rows
        for row in csv_rows:
            # Fill missing values with None/empty
            complete_row = {col: row.get(col, '') for col in columns}
            writer.writerow(complete_row)
        
        # Prepare file for download
        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        output.close()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"progress_adjusted_metr_data_{timestamp}.csv"
        
        return send_file(
            mem,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Error exporting METR data: {e}")
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
            'rho_coding_labor': params.rho_coding_labor,
            'rho_experiment_capacity': params.rho_experiment_capacity,
            'alpha_experiment_capacity': params.alpha_experiment_capacity,
            'r_software': params.r_software,
            'automation_fraction_at_coding_automation_anchor': params.automation_fraction_at_coding_automation_anchor,
            'coding_labor_normalization': params.coding_labor_normalization,
            'experiment_compute_exponent': params.experiment_compute_exponent,
            'ai_research_taste_at_coding_automation_anchor': params.ai_research_taste_at_coding_automation_anchor,
            'ai_research_taste_slope': params.ai_research_taste_slope,
            'taste_schedule_type': params.taste_schedule_type,
            'progress_at_sc': params.progress_at_sc,
            'aa_time_horizon_minutes': params.aa_time_horizon_minutes,
            'horizon_extrapolation_type': params.horizon_extrapolation_type,
            # Manual horizon fitting parameters
            'present_day': params.present_day,
            'present_horizon': params.present_horizon,
            'present_doubling_time': params.present_doubling_time,
            'doubling_difficulty_growth_factor': params.doubling_difficulty_growth_factor,
            # Baseline Annual Compute Multiplier
            'baseline_annual_compute_multiplier': params.baseline_annual_compute_multiplier,
            # Research taste distribution parameter
            'median_to_top_taste_multiplier': params.median_to_top_taste_multiplier,
            'top_percentile': params.top_percentile
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
                'plots': plot_info,
                'explanation': tab_config.explanation
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