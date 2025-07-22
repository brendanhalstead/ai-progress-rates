#!/usr/bin/env python3
"""
Visualization Module for AI Progress Modeling

Provides modular, clean visualization functionality for analyzing AI progress trajectories,
automation fractions, component contributions, and parameter sensitivity.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlotConfig:
    """Configuration for plot styling and layout"""
    figsize: Tuple[float, float] = (12, 8)
    dpi: int = 100
    style: str = 'seaborn-v0_8'
    color_palette: List[str] = None
    grid_alpha: float = 0.3
    line_width: float = 2.0
    font_size: int = 12
    title_size: int = 14
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


class ProgressVisualizer:
    """Main visualization class for AI progress modeling results"""
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self._setup_matplotlib()
    
    def _setup_matplotlib(self):
        """Configure matplotlib with consistent styling"""
        try:
            plt.style.use(self.config.style)
        except OSError:
            plt.style.use('default')
        
        plt.rcParams.update({
            'figure.dpi': self.config.dpi,
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.title_size,
            'axes.labelsize': self.config.font_size,
            'legend.fontsize': self.config.font_size,
            'xtick.labelsize': self.config.font_size - 1,
            'ytick.labelsize': self.config.font_size - 1,
            'lines.linewidth': self.config.line_width,
            'axes.grid': True,
            'axes.axisbelow': True,
            'grid.alpha': self.config.grid_alpha
        })
    
    def plot_progress_trajectory(self, times: np.ndarray, progress: np.ndarray, 
                               title: str = "AI Progress Over Time", 
                               ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot cumulative progress over time
        
        Args:
            times: Time points
            progress: Cumulative progress values
            title: Plot title
            ax: Existing axes to plot on
        
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize)
        
        ax.plot(times, progress, color=self.config.color_palette[0], 
                linewidth=self.config.line_width, label='Cumulative Progress')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative Progress')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=self.config.grid_alpha)
        
        return ax
    
    def plot_automation_fraction(self, times: np.ndarray, automation: np.ndarray,
                                title: str = "Automation Fraction Over Time",
                                ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot automation fraction over time
        
        Args:
            times: Time points
            automation: Automation fraction values [0,1]
            title: Plot title
            ax: Existing axes to plot on
        
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize)
        
        ax.plot(times, automation, color=self.config.color_palette[1], 
                linewidth=self.config.line_width, label='Automation Fraction')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Automation Fraction')
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=self.config.grid_alpha)
        
        # Add percentage formatting to y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        return ax
    
    def plot_progress_rate(self, times: np.ndarray, rates: np.ndarray,
                          title: str = "Progress Rate Over Time",
                          ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot instantaneous progress rate over time
        
        Args:
            times: Time points
            rates: Progress rate values
            title: Plot title
            ax: Existing axes to plot on
        
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize)
        
        ax.plot(times, rates, color=self.config.color_palette[2], 
                linewidth=self.config.line_width, label='Progress Rate')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Progress Rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=self.config.grid_alpha)
        
        return ax
    
    def plot_component_contributions(self, times: np.ndarray, 
                                   software_progress: np.ndarray,
                                   training_progress: np.ndarray,
                                   title: str = "Progress Component Contributions",
                                   ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot software vs training contributions to overall progress
        
        Args:
            times: Time points
            software_progress: Software progress component
            training_progress: Training progress component
            title: Plot title
            ax: Existing axes to plot on
        
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize)
        
        ax.plot(times, software_progress, color=self.config.color_palette[0],
                linewidth=self.config.line_width, label='Software Progress')
        ax.plot(times, training_progress, color=self.config.color_palette[1],
                linewidth=self.config.line_width, label='Training Progress')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Progress Rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=self.config.grid_alpha)
        
        return ax
    
    def plot_labor_contributions(self, times: np.ndarray,
                               L_AI: np.ndarray, L_HUMAN: np.ndarray,
                               automation_fraction: np.ndarray,
                               title: str = "Labor Supply and Automation",
                               ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot AI vs human labor with automation overlay
        
        Args:
            times: Time points
            L_AI: AI labor supply
            L_HUMAN: Human labor supply
            automation_fraction: Automation fraction [0,1]
            title: Plot title
            ax: Existing axes to plot on
        
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Plot labor supplies on log scale
        ax.semilogy(times, L_AI, color=self.config.color_palette[0],
                    linewidth=self.config.line_width, label='AI Labor')
        ax.semilogy(times, L_HUMAN, color=self.config.color_palette[1],
                    linewidth=self.config.line_width, label='Human Labor')
        
        # Add automation fraction on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(times, automation_fraction, color=self.config.color_palette[2],
                 linewidth=self.config.line_width, linestyle='--', 
                 label='Automation Fraction')
        ax2.set_ylabel('Automation Fraction')
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Labor Supply (log scale)')
        ax.set_title(title)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center left')
        
        ax.grid(True, alpha=self.config.grid_alpha)
        
        return ax
    
    def plot_comprehensive_dashboard(self, results: Dict[str, np.ndarray],
                                   time_series_data: Optional[Dict[str, np.ndarray]] = None,
                                   title: str = "AI Progress Modeling Dashboard") -> plt.Figure:
        """
        Create comprehensive dashboard with multiple subplots
        
        Args:
            results: Dictionary containing 'times', 'progress', 'automation_fraction', etc.
            time_series_data: Optional input time series data
            title: Overall figure title
        
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Create 2x3 subplot grid
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        times = results['times']
        
        # 1. Progress trajectory
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_progress_trajectory(times, results['progress'], 
                                    "Cumulative Progress", ax1)
        
        # 2. Automation fraction
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_automation_fraction(times, results['automation_fraction'],
                                    "Automation Fraction", ax2)
        
        # 3. Progress rate (if available)
        if 'progress_rate' in results:
            ax3 = fig.add_subplot(gs[1, 0])
            self.plot_progress_rate(times, results['progress_rate'],
                                  "Progress Rate", ax3)
        
        # 4. Component contributions (if available)
        if 'software_progress' in results and 'training_progress' in results:
            ax4 = fig.add_subplot(gs[1, 1])
            self.plot_component_contributions(times, results['software_progress'],
                                            results['training_progress'],
                                            "Progress Components", ax4)
        
        # 5. Labor contributions (if time series data available)
        if time_series_data is not None:
            ax5 = fig.add_subplot(gs[2, :])
            self.plot_labor_contributions(times, 
                                        np.interp(times, time_series_data['time'], time_series_data['L_AI']),
                                        np.interp(times, time_series_data['time'], time_series_data['L_HUMAN']),
                                        results['automation_fraction'],
                                        "Labor Supply and Automation", ax5)
        
        return fig
    
    def plot_parameter_sensitivity(self, param_name: str, param_values: np.ndarray,
                                 progress_trajectories: List[np.ndarray],
                                 times: np.ndarray,
                                 title: Optional[str] = None) -> plt.Figure:
        """
        Plot sensitivity analysis for a parameter
        
        Args:
            param_name: Name of the parameter being varied
            param_values: Array of parameter values
            progress_trajectories: List of progress trajectories for each parameter value
            times: Time points
            title: Plot title
        
        Returns:
            Matplotlib figure object
        """
        if title is None:
            title = f"Sensitivity Analysis: {param_name}"
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot all trajectories
        colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
        
        for i, (val, traj) in enumerate(zip(param_values, progress_trajectories)):
            ax1.plot(times, traj, color=colors[i], alpha=0.7,
                    label=f'{param_name}={val:.3f}')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Cumulative Progress')
        ax1.set_title(f'Progress Trajectories vs {param_name}')
        ax1.grid(True, alpha=self.config.grid_alpha)
        
        # Add colorbar legend
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                  norm=plt.Normalize(vmin=param_values.min(), 
                                                   vmax=param_values.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1)
        cbar.set_label(param_name)
        
        # Plot final progress vs parameter value
        final_progress = [traj[-1] for traj in progress_trajectories]
        ax2.plot(param_values, final_progress, 'o-', color=self.config.color_palette[0],
                linewidth=self.config.line_width, markersize=6)
        ax2.set_xlabel(param_name)
        ax2.set_ylabel(f'Final Progress (at t={times[-1]:.1f})')
        ax2.set_title(f'Final Progress vs {param_name}')
        ax2.grid(True, alpha=self.config.grid_alpha)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def save_plot(self, fig: plt.Figure, filename: str, format: str = 'png', 
                  dpi: Optional[int] = None, bbox_inches: str = 'tight'):
        """
        Save plot to file with consistent formatting
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            format: File format ('png', 'pdf', 'svg', etc.)
            dpi: Resolution (uses config default if None)
            bbox_inches: Bounding box specification
        """
        if dpi is None:
            dpi = self.config.dpi
        
        fig.savefig(filename, format=format, dpi=dpi, bbox_inches=bbox_inches)
        logger.info(f"Plot saved to {filename}")
    
    def show_plot(self, fig: Optional[plt.Figure] = None):
        """Display plot"""
        if fig is not None:
            plt.figure(fig.number)
        plt.show()


def create_default_visualizer(style: str = 'seaborn-v0_8', figsize: Tuple[float, float] = (12, 8)) -> ProgressVisualizer:
    """Convenience function to create visualizer with common settings"""
    config = PlotConfig(style=style, figsize=figsize)
    return ProgressVisualizer(config)


def quick_plot_results(results: Dict[str, np.ndarray], 
                      time_series_data: Optional[Dict[str, np.ndarray]] = None,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Quick function to plot results with minimal setup
    
    Args:
        results: Results dictionary from ProgressModel
        time_series_data: Optional input time series
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    visualizer = create_default_visualizer()
    fig = visualizer.plot_comprehensive_dashboard(results, time_series_data)
    
    if save_path:
        visualizer.save_plot(fig, save_path)
    
    return fig