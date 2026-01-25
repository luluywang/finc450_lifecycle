"""
Plot utility functions and helpers for lifecycle visualization.

This module provides common plotting utilities used across visualization modules.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from core import LifecycleParams, LifecycleResult


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


# =============================================================================
# Generic Plotting Helpers (DRY refactoring)
# =============================================================================

def plot_fan_chart(
    ax: plt.Axes,
    paths: np.ndarray,
    x: np.ndarray,
    percentiles: List[int] = None,
    color: str = 'blue',
    label_prefix: str = '',
    alpha_outer: float = 0.2,
    alpha_inner: float = 0.35,
    show_median_label: bool = True,
) -> np.ndarray:
    """
    Plot a generic fan chart with percentile bands.

    This is the single source of truth for fan chart plotting, replacing
    duplicate implementations across dashboard.py, monte_carlo_plots.py,
    and comparison_plots.py.

    Args:
        ax: Matplotlib axes to plot on
        paths: 2D array of shape (n_simulations, n_periods)
        x: X-axis values
        percentiles: List of percentiles to plot [5, 25, 50, 75, 95]
        color: Color for the bands and median line
        label_prefix: Prefix for legend labels (e.g., 'LDI', 'RoT')
        alpha_outer: Transparency for outer band (5-95%)
        alpha_inner: Transparency for inner band (25-75%)
        show_median_label: Whether to add median to legend

    Returns:
        Array of percentile values at each time point (shape: n_percentiles x n_periods)
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    # Compute percentiles
    pctl_values = np.percentile(paths, percentiles, axis=0)

    # Create percentile index lookup
    p_idx = {p: i for i, p in enumerate(percentiles)}

    # Plot outer band (5-95%)
    if 5 in p_idx and 95 in p_idx:
        ax.fill_between(x, pctl_values[p_idx[5]], pctl_values[p_idx[95]],
                        alpha=alpha_outer, color=color)

    # Plot inner band (25-75%)
    if 25 in p_idx and 75 in p_idx:
        ax.fill_between(x, pctl_values[p_idx[25]], pctl_values[p_idx[75]],
                        alpha=alpha_inner, color=color)

    # Plot median line
    if 50 in p_idx:
        label = f'{label_prefix} Median' if show_median_label else None
        ax.plot(x, pctl_values[p_idx[50]], color=color, linewidth=2, label=label)

    return pctl_values


def plot_allocation_stack(
    ax: plt.Axes,
    x: np.ndarray,
    stock_weights: np.ndarray,
    bond_weights: np.ndarray,
    cash_weights: np.ndarray,
    retirement_x: float = None,
    colors: dict = None,
    as_percentage: bool = True,
    show_legend: bool = True,
) -> None:
    """
    Plot a stacked area chart for portfolio allocation.

    This is the single source of truth for allocation stacked charts, replacing
    duplicate implementations across multiple files.

    Args:
        ax: Matplotlib axes to plot on
        x: X-axis values
        stock_weights: Stock allocation weights (0-1 or 0-100)
        bond_weights: Bond allocation weights
        cash_weights: Cash allocation weights
        retirement_x: X-coordinate for retirement vertical line
        colors: Dict with 'stock', 'bond', 'cash' colors
        as_percentage: If True, multiply weights by 100 for display
        show_legend: Whether to show legend
    """
    if colors is None:
        colors = {
            'stock': '#F4A261',  # Coral (colorblind-safe)
            'bond': '#9b59b6',   # Purple (unchanged)
            'cash': '#95a5a6',   # Gray (unchanged)
        }

    multiplier = 100 if as_percentage else 1

    # Stack from bottom: cash, bonds, stocks
    ax.stackplot(x,
                 stock_weights * multiplier,
                 bond_weights * multiplier,
                 cash_weights * multiplier,
                 labels=['Stocks', 'Bonds', 'Cash'],
                 colors=[colors['stock'], colors['bond'], colors['cash']],
                 alpha=0.8)

    if retirement_x is not None:
        ax.axvline(x=retirement_x, color='white', linestyle='--', linewidth=2)

    if as_percentage:
        ax.set_ylim(0, 100)
        ax.set_ylabel('Allocation (%)')

    if show_legend:
        ax.legend(loc='upper right', fontsize=8)


def setup_comparison_axes(
    result_or_ages: Union['LifecycleResult', np.ndarray],
    params: 'LifecycleParams' = None,
    start_age: int = None,
    retirement_age: int = None,
    use_years: bool = True,
) -> Tuple[np.ndarray, str, float]:
    """
    Unified x-axis computation for comparison and strategy plots.

    This replaces all duplicate get_x_axis() implementations across files.

    Args:
        result_or_ages: Either a LifecycleResult object or array of ages
        params: LifecycleParams (used if result_or_ages is LifecycleResult)
        start_age: Start age (used if params not provided)
        retirement_age: Retirement age (used if params not provided)
        use_years: If True, use years from start; if False, use ages

    Returns:
        Tuple of (x_values, xlabel, retirement_x)
    """
    # Extract ages from result if needed
    if hasattr(result_or_ages, 'ages'):
        ages = result_or_ages.ages
    else:
        ages = result_or_ages

    # Get start and retirement ages
    if params is not None:
        start_age = params.start_age
        retirement_age = params.retirement_age
    elif start_age is None or retirement_age is None:
        raise ValueError("Must provide either params or both start_age and retirement_age")

    if use_years:
        x = np.arange(len(ages))
        xlabel = 'Years from Career Start'
        retirement_x = retirement_age - start_age
    else:
        x = ages
        xlabel = 'Age'
        retirement_x = retirement_age

    return x, xlabel, retirement_x


def compute_x_axis_from_ages(
    ages: np.ndarray,
    start_age: int,
    retirement_age: int,
    use_years: bool = True,
) -> Tuple[np.ndarray, str, float]:
    """
    Simplified x-axis computation when ages array is already available.

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


def add_standard_chart_elements(
    ax: plt.Axes,
    retirement_x: float,
    xlabel: str,
    ylabel: str,
    title: str,
    add_zero: bool = True,
    legend_loc: str = 'upper right',
) -> None:
    """
    Add standard chart elements (retirement line, zero line, labels, title, legend).

    Args:
        ax: Matplotlib axes
        retirement_x: X-coordinate for retirement line
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Chart title
        add_zero: Whether to add horizontal line at y=0
        legend_loc: Legend location
    """
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    if add_zero:
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc=legend_loc, fontsize=9)
