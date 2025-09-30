#!/usr/bin/env python3
"""
Module for generating TimeSeriesData instances with uncertainty.

This module provides functionality to generate input time series data with parametric
uncertainty for Monte Carlo simulations. It takes a base CSV file and allows varying
certain time series according to sampled parameters.

Current methodology:
1. Human labor (L_HUMAN) - copied from base CSV
2. Inference compute - copied from base CSV
3. Experiment compute - copied from base CSV
4. Training compute growth rate - constant up to slowdown_year, then switches to
   post_slowdown_training_compute_growth_rate

This is designed to be extensible as more sophisticated uncertainty models are developed.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from progress_model import TimeSeriesData, load_time_series_data


@dataclass
class TimeSeriesGenerationParams:
    """Parameters controlling time series generation with uncertainty.

    Attributes:
        base_csv_path: Path to the base CSV file to use as template
        constant_training_compute_growth_rate: Growth rate before slowdown
        slowdown_year: Year at which training compute growth rate changes
        post_slowdown_training_compute_growth_rate: Growth rate after slowdown
    """
    base_csv_path: str
    constant_training_compute_growth_rate: float
    slowdown_year: float
    post_slowdown_training_compute_growth_rate: float


def generate_time_series_data(
    params: TimeSeriesGenerationParams,
    base_data: Optional[TimeSeriesData] = None
) -> TimeSeriesData:
    """Generate a TimeSeriesData instance with parametric uncertainty.

    This function creates a new TimeSeriesData instance by:
    1. Loading (or using provided) base time series data
    2. Keeping L_HUMAN, inference_compute, and experiment_compute unchanged
    3. Modifying training_compute_growth_rate based on the parameters

    Args:
        params: Parameters controlling the generation
        base_data: Optional pre-loaded base time series. If None, will load from
                   params.base_csv_path

    Returns:
        A new TimeSeriesData instance with modified training compute growth rate
    """
    # Load base data if not provided
    if base_data is None:
        base_data = load_time_series_data(params.base_csv_path)

    # Copy the base arrays (these remain unchanged)
    time = base_data.time.copy()
    L_HUMAN = base_data.L_HUMAN.copy()
    inference_compute = base_data.inference_compute.copy()
    experiment_compute = base_data.experiment_compute.copy()

    # Generate new training compute growth rate based on parameters
    training_compute_growth_rate = np.where(
        time < params.slowdown_year,
        params.constant_training_compute_growth_rate,
        params.post_slowdown_training_compute_growth_rate
    )

    return TimeSeriesData(
        time=time,
        L_HUMAN=L_HUMAN,
        inference_compute=inference_compute,
        experiment_compute=experiment_compute,
        training_compute_growth_rate=training_compute_growth_rate
    )


def generate_time_series_from_dict(
    param_dict: dict,
    base_csv_path: str,
    base_data: Optional[TimeSeriesData] = None
) -> TimeSeriesData:
    """Convenience function to generate time series from a parameter dictionary.

    This is useful for integration with Monte Carlo sampling where parameters
    are stored as dictionaries.

    Args:
        param_dict: Dictionary containing time series generation parameters:
            - constant_training_compute_growth_rate (required)
            - slowdown_year (required)
            - post_slowdown_training_compute_growth_rate (required)
        base_csv_path: Path to base CSV file
        base_data: Optional pre-loaded base time series

    Returns:
        A new TimeSeriesData instance

    Raises:
        KeyError: If required parameters are missing from param_dict
    """
    params = TimeSeriesGenerationParams(
        base_csv_path=base_csv_path,
        constant_training_compute_growth_rate=param_dict['constant_training_compute_growth_rate'],
        slowdown_year=param_dict['slowdown_year'],
        post_slowdown_training_compute_growth_rate=param_dict['post_slowdown_training_compute_growth_rate']
    )

    return generate_time_series_data(params, base_data)
