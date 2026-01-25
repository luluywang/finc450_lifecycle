"""
Lifecycle visualization plots for median path analysis.

This module provides plotting functions for individual lifecycle charts
showing earnings, expenses, present values, durations, wealth, and allocations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

from .styles import COLORS
from .helpers import add_zero_line, set_standard_xlim

if TYPE_CHECKING:
    from core import LifecycleResult, LifecycleParams


def plot_earnings_expenses_profile(
    result: 'LifecycleResult',
    params: 'LifecycleParams',
    ax: plt.Axes,
    use_years: bool = True
) -> None:
    """Plot 1: Profile of Earnings and Expenses."""
    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result.ages
        xlabel = 'Age'

    ax.plot(x, result.earnings, color=COLORS['blue'], linewidth=2, label='Wages')
    ax.plot(x, result.expenses, color=COLORS['green'], linewidth=2, label='Expenses')

    retirement_x = params.retirement_age - params.start_age if use_years else params.retirement_age
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    add_zero_line(ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Profile of Earnings and Expenses')
    ax.legend(loc='upper right', fontsize=8)
    set_standard_xlim(ax, x)


def plot_forward_present_values(
    result: 'LifecycleResult',
    params: 'LifecycleParams',
    ax: plt.Axes,
    use_years: bool = True
) -> None:
    """Plot 2: Forward Looking Present Values."""
    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result.ages
        xlabel = 'Age'

    ax.plot(x, result.pv_earnings, color=COLORS['blue'], linewidth=2, label='PV of Future Earnings')
    ax.plot(x, -result.pv_expenses, color=COLORS['green'], linewidth=2, label='PV of Future Expenses')

    retirement_x = params.retirement_age - params.start_age if use_years else params.retirement_age
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    add_zero_line(ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Forward Looking Present Values')
    ax.legend(loc='upper right', fontsize=8)
    set_standard_xlim(ax, x)


def plot_durations(
    result: 'LifecycleResult',
    params: 'LifecycleParams',
    ax: plt.Axes,
    use_years: bool = True
) -> None:
    """Plot 3: Durations of Assets."""
    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result.ages
        xlabel = 'Age'

    ax.plot(x, result.duration_earnings, color=COLORS['blue'], linewidth=2, label='Duration of Future Earnings')
    ax.plot(x, -result.duration_expenses, color=COLORS['green'], linewidth=2, label='Duration of Expenses (Liability)')

    retirement_x = params.retirement_age - params.start_age if use_years else params.retirement_age
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    add_zero_line(ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Years')
    ax.set_title('Durations of Assets')
    ax.legend(loc='upper right', fontsize=8)
    set_standard_xlim(ax, x)


def plot_human_vs_financial_wealth(
    result: 'LifecycleResult',
    params: 'LifecycleParams',
    ax: plt.Axes,
    use_years: bool = True
) -> None:
    """Plot 4: Human Capital vs Financial Wealth."""
    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result.ages
        xlabel = 'Age'

    ax.plot(x, result.human_capital, color=COLORS['orange'], linewidth=2, label='Human Capital')
    ax.plot(x, result.financial_wealth, color=COLORS['blue'], linewidth=2, label='Financial Wealth')
    ax.plot(x, result.total_wealth, color=COLORS['green'], linewidth=2, label='Total Wealth')

    retirement_x = params.retirement_age - params.start_age if use_years else params.retirement_age
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    add_zero_line(ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Human Capital vs Financial Wealth')
    ax.legend(loc='upper right', fontsize=8)
    set_standard_xlim(ax, x)


def plot_hc_decomposition(
    result: 'LifecycleResult',
    params: 'LifecycleParams',
    ax: plt.Axes,
    use_years: bool = True
) -> None:
    """Plot 5: Portfolio Decomposition of Human Capital."""
    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result.ages
        xlabel = 'Age'

    ax.plot(x, result.hc_stock_component, color=COLORS['blue'], linewidth=2, label='Stocks in Human Capital')
    ax.plot(x, result.hc_bond_component, color=COLORS['orange'], linewidth=2, label='Bonds in Human Capital')
    ax.plot(x, result.hc_cash_component, color=COLORS['green'], linewidth=2, label='Cash in Human Capital')

    retirement_x = params.retirement_age - params.start_age if use_years else params.retirement_age
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    add_zero_line(ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Portfolio Decomposition of Human Capital')
    ax.legend(loc='upper right', fontsize=8)
    set_standard_xlim(ax, x)


def plot_target_financial_holdings(
    result: 'LifecycleResult',
    params: 'LifecycleParams',
    ax: plt.Axes,
    use_years: bool = True
) -> None:
    """Plot 6: Target Financial Holdings."""
    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result.ages
        xlabel = 'Age'

    ax.plot(x, result.target_fin_stocks, color=COLORS['blue'], linewidth=2, label='Target Financial Stocks')
    ax.plot(x, result.target_fin_bonds, color=COLORS['orange'], linewidth=2, label='Target Financial Bonds')
    ax.plot(x, result.target_fin_cash, color=COLORS['green'], linewidth=2, label='Target Financial Cash')

    retirement_x = params.retirement_age - params.start_age if use_years else params.retirement_age
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    add_zero_line(ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Target Financial Holdings')
    ax.legend(loc='upper right', fontsize=8)
    set_standard_xlim(ax, x)


def plot_portfolio_shares(
    result: 'LifecycleResult',
    params: 'LifecycleParams',
    ax: plt.Axes,
    use_years: bool = True
) -> None:
    """Plot 7: Target Financial Portfolio Shares (no short constraint)."""
    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result.ages
        xlabel = 'Age'

    ax.plot(x, result.stock_weight_no_short, color=COLORS['blue'], linewidth=2, label='Stock Weight - No Short')
    ax.plot(x, result.bond_weight_no_short, color=COLORS['orange'], linewidth=2, label='Bond Weight - No Short')
    ax.plot(x, result.cash_weight_no_short, color=COLORS['green'], linewidth=2, label='Cash Weight - No Short')

    retirement_x = params.retirement_age - params.start_age if use_years else params.retirement_age
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    add_zero_line(ax)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weight')
    ax.set_title('Target Financial Portfolio Shares')
    ax.legend(loc='upper right', fontsize=8)
    set_standard_xlim(ax, x)
    ax.set_ylim(-0.05, 1.15)


def plot_total_wealth_holdings(
    result: 'LifecycleResult',
    params: 'LifecycleParams',
    ax: plt.Axes,
    use_years: bool = True
) -> None:
    """Plot 8: Target Total Wealth Holdings."""
    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result.ages
        xlabel = 'Age'

    ax.plot(x, result.total_stocks, color=COLORS['blue'], linewidth=2, label='Target Stock')
    ax.plot(x, result.total_bonds, color=COLORS['orange'], linewidth=2, label='Target Bond')
    ax.plot(x, result.total_cash, color=COLORS['green'], linewidth=2, label='Target Cash')

    retirement_x = params.retirement_age - params.start_age if use_years else params.retirement_age
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    add_zero_line(ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Target Total Wealth Holdings')
    ax.legend(loc='upper right', fontsize=8)
    set_standard_xlim(ax, x)


def plot_consumption_dollars(
    result: 'LifecycleResult',
    params: 'LifecycleParams',
    ax: plt.Axes,
    use_years: bool = True
) -> None:
    """Plot 9: Total Consumption in Dollar Terms."""
    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result.ages
        xlabel = 'Age'

    ax.plot(x, result.total_consumption, color=COLORS['blue'], linewidth=2,
            label='Total Consumption')

    retirement_x = params.retirement_age - params.start_age if use_years else params.retirement_age
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    add_zero_line(ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Consumption')
    ax.legend(loc='upper right', fontsize=8)
    set_standard_xlim(ax, x)
    y_max = max(result.total_consumption) * 1.1
    ax.set_ylim(0, y_max)


def plot_consumption_breakdown(
    result: 'LifecycleResult',
    params: 'LifecycleParams',
    ax: plt.Axes,
    use_years: bool = True
) -> None:
    """Plot: Consumption breakdown (subsistence vs variable)."""
    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result.ages
        xlabel = 'Age'

    ax.plot(x, result.subsistence_consumption, color=COLORS['green'], linewidth=2,
            label='Subsistence Consumption')
    ax.plot(x, result.variable_consumption, color=COLORS['orange'], linewidth=2,
            label='Variable (r+1pp of NW, capped)')
    ax.plot(x, result.total_consumption, color=COLORS['blue'], linewidth=2,
            label='Total Consumption')
    ax.plot(x, result.earnings, color='gray', linewidth=1, linestyle='--', alpha=0.7,
            label='Earnings (cap)')

    retirement_x = params.retirement_age - params.start_age if use_years else params.retirement_age
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    add_zero_line(ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Consumption Breakdown')
    ax.legend(loc='upper right', fontsize=8)
    set_standard_xlim(ax, x)
    y_max = max(max(result.total_consumption), max(result.earnings)) * 1.1
    ax.set_ylim(0, y_max)


def create_lifecycle_figure(
    result: 'LifecycleResult',
    params: 'LifecycleParams',
    figsize: tuple = (20, 10),
    use_years: bool = True
) -> plt.Figure:
    """
    Create a single figure with lifecycle strategy charts.

    Layout: 2 rows x 4 columns

    Args:
        result: Lifecycle calculation results
        params: Lifecycle parameters
        figsize: Figure size tuple
        use_years: If True, x-axis shows years from career start; if False, shows age
    """
    fig, axes = plt.subplots(2, 4, figsize=figsize)

    # Row 1
    plot_earnings_expenses_profile(result, params, axes[0, 0], use_years)
    plot_forward_present_values(result, params, axes[0, 1], use_years)
    plot_durations(result, params, axes[0, 2], use_years)
    plot_human_vs_financial_wealth(result, params, axes[0, 3], use_years)

    # Row 2
    plot_target_financial_holdings(result, params, axes[1, 0], use_years)
    plot_total_wealth_holdings(result, params, axes[1, 1], use_years)
    plot_consumption_breakdown(result, params, axes[1, 2], use_years)
    plot_consumption_dollars(result, params, axes[1, 3], use_years)

    plt.tight_layout()
    return fig
