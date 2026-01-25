"""
Plot utility functions and helpers for lifecycle visualization.

This module provides common plotting utilities used across visualization modules.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def apply_wealth_log_scale(ax: plt.Axes, linthresh: float = 50, linscale: float = 0.5) -> None:
    """
    Apply symmetric log scale to a wealth chart's Y-axis.

    Uses symlog which handles values near zero gracefully and works with
    the transition from small to large wealth values.

    Args:
        ax: The matplotlib axes to modify
        linthresh: Values between -linthresh and +linthresh display linearly
        linscale: Controls visual transition between linear and log regions
    """
    ax.set_yscale('symlog', linthresh=linthresh, linscale=linscale)


def setup_figure(figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """Create a figure with consistent styling."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def add_retirement_line(ax: plt.Axes, retirement_x: float, use_years: bool = True) -> None:
    """Add a vertical line at retirement."""
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.7,
               label='Retirement', linewidth=1.5)


def get_x_axis(ages: np.ndarray, start_age: int, retirement_age: int, use_years: bool = True):
    """
    Get x-axis values and labels.

    Args:
        ages: Array of ages
        start_age: Age at career start
        retirement_age: Age at retirement
        use_years: If True, use years from start on x-axis

    Returns:
        Tuple of (x_values, xlabel, retirement_x)
    """
    if use_years:
        x = np.arange(len(ages))
        xlabel = 'Years from Career Start'
        retirement_x = retirement_age - start_age
    else:
        x = ages
        xlabel = 'Age'
        retirement_x = retirement_age
    return x, xlabel, retirement_x


def add_zero_line(ax: plt.Axes, alpha: float = 0.3) -> None:
    """Add a horizontal line at y=0."""
    ax.axhline(y=0, color='gray', linestyle='-', alpha=alpha)


def format_currency_axis(ax: plt.Axes, axis: str = 'y') -> None:
    """Format axis labels as currency."""
    def currency_formatter(x, pos):
        return f'${x:,.0f}k'

    if axis == 'y':
        ax.yaxis.set_major_formatter(plt.FuncFormatter(currency_formatter))
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(currency_formatter))


def set_standard_xlim(ax: plt.Axes, x: np.ndarray, padding: int = 2) -> None:
    """Set standard x-axis limits with padding."""
    ax.set_xlim(x[0] - padding, x[-1] + padding)


def add_legend(ax: plt.Axes, loc: str = 'upper right', fontsize: int = 8) -> None:
    """Add a legend with standard formatting."""
    ax.legend(loc=loc, fontsize=fontsize)
