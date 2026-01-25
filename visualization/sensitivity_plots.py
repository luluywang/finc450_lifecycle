"""
Parameter sensitivity visualization plots.

This module provides plotting functions for analyzing how lifecycle strategy
responds to changes in parameters like beta, gamma, initial wealth, equity premium, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core import LifecycleParams, EconomicParams


def create_beta_comparison_figure(
    beta_values: list = [0.0, 0.5, 1.0],
    base_params: 'LifecycleParams' = None,
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (16, 12),
    use_years: bool = True
) -> plt.Figure:
    """
    Create a figure comparing key metrics across different stock beta values.

    Shows how portfolio allocation, human capital decomposition, and target
    holdings change as stock beta varies from 0 to 1.
    """
    from core import LifecycleParams, EconomicParams, compute_lifecycle_median_path

    if base_params is None:
        base_params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    # Compute results for each beta value
    results = {}
    for beta in beta_values:
        params = LifecycleParams(
            start_age=base_params.start_age,
            retirement_age=base_params.retirement_age,
            end_age=base_params.end_age,
            initial_earnings=base_params.initial_earnings,
            earnings_growth=base_params.earnings_growth,
            earnings_hump_age=base_params.earnings_hump_age,
            earnings_decline=base_params.earnings_decline,
            base_expenses=base_params.base_expenses,
            expense_growth=base_params.expense_growth,
            retirement_expenses=base_params.retirement_expenses,
            stock_beta_human_capital=beta,
            gamma=base_params.gamma,
            target_stock_allocation=base_params.target_stock_allocation,
            target_bond_allocation=base_params.target_bond_allocation,
            risk_free_rate=base_params.risk_free_rate,
            equity_premium=base_params.equity_premium,
            initial_wealth=base_params.initial_wealth,
        )
        results[beta] = compute_lifecycle_median_path(params, econ_params)

    # Create figure with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Colors for different beta values
    beta_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green

    # Get x-axis values
    result_0 = results[beta_values[0]]
    if use_years:
        x = np.arange(len(result_0.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result_0.ages
        xlabel = 'Age'

    retirement_x = base_params.retirement_age - base_params.start_age if use_years else base_params.retirement_age

    # Plot 1: Stock Weight comparison
    ax = axes[0, 0]
    for i, beta in enumerate(beta_values):
        ax.plot(x, results[beta].stock_weight_no_short, color=beta_colors[i],
                linewidth=2, label=f'Beta = {beta}')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weight')
    ax.set_title('Stock Weight by Beta')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.05, 1.15)

    # Plot 2: Bond Weight comparison
    ax = axes[0, 1]
    for i, beta in enumerate(beta_values):
        ax.plot(x, results[beta].bond_weight_no_short, color=beta_colors[i],
                linewidth=2, label=f'Beta = {beta}')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weight')
    ax.set_title('Bond Weight by Beta')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.05, 1.15)

    # Plot 3: Cash Weight comparison
    ax = axes[0, 2]
    for i, beta in enumerate(beta_values):
        ax.plot(x, results[beta].cash_weight_no_short, color=beta_colors[i],
                linewidth=2, label=f'Beta = {beta}')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weight')
    ax.set_title('Cash Weight by Beta')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.05, 1.15)

    # Compute shared y-axis range for bottom row (HC decomposition charts)
    hc_min = float('inf')
    hc_max = float('-inf')
    for beta in beta_values:
        for component in [results[beta].hc_stock_component,
                          results[beta].hc_bond_component,
                          results[beta].hc_cash_component]:
            hc_min = min(hc_min, np.min(component))
            hc_max = max(hc_max, np.max(component))
    # Add 5% padding
    hc_range = hc_max - hc_min
    hc_ylim = (hc_min - 0.05 * hc_range, hc_max + 0.05 * hc_range)

    # Plot 4: HC Stock Component comparison
    ax = axes[1, 0]
    for i, beta in enumerate(beta_values):
        ax.plot(x, results[beta].hc_stock_component, color=beta_colors[i],
                linewidth=2, label=f'Beta = {beta}')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Stock Component of Human Capital')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(hc_ylim)

    # Plot 5: HC Bond Component comparison
    ax = axes[1, 1]
    for i, beta in enumerate(beta_values):
        ax.plot(x, results[beta].hc_bond_component, color=beta_colors[i],
                linewidth=2, label=f'Beta = {beta}')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Bond Component of Human Capital')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(hc_ylim)

    # Plot 6: HC Cash Component comparison
    ax = axes[1, 2]
    for i, beta in enumerate(beta_values):
        ax.plot(x, results[beta].hc_cash_component, color=beta_colors[i],
                linewidth=2, label=f'Beta = {beta}')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Cash Component of Human Capital')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(hc_ylim)

    plt.tight_layout()
    return fig


def create_gamma_comparison_figure(
    gamma_values: list = [1.0, 2.0, 4.0, 8.0],
    base_params: 'LifecycleParams' = None,
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (16, 12),
    use_years: bool = True
) -> plt.Figure:
    """
    Create a figure comparing key metrics across different risk aversion (gamma) values.

    Shows how portfolio allocation changes with different risk tolerances.
    """
    from core import LifecycleParams, EconomicParams, compute_lifecycle_median_path

    if base_params is None:
        base_params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    # Compute results for each gamma value
    results = {}
    for gamma in gamma_values:
        params = LifecycleParams(
            start_age=base_params.start_age,
            retirement_age=base_params.retirement_age,
            end_age=base_params.end_age,
            initial_earnings=base_params.initial_earnings,
            earnings_growth=base_params.earnings_growth,
            earnings_hump_age=base_params.earnings_hump_age,
            earnings_decline=base_params.earnings_decline,
            base_expenses=base_params.base_expenses,
            expense_growth=base_params.expense_growth,
            retirement_expenses=base_params.retirement_expenses,
            stock_beta_human_capital=base_params.stock_beta_human_capital,
            gamma=gamma,
            consumption_boost=base_params.consumption_boost,
            initial_wealth=base_params.initial_wealth,
        )
        results[gamma] = compute_lifecycle_median_path(params, econ_params)

    # Create figure with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Colors for different gamma values
    gamma_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(gamma_values)))

    # Get x-axis values
    result_0 = results[gamma_values[0]]
    if use_years:
        x = np.arange(len(result_0.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result_0.ages
        xlabel = 'Age'

    retirement_x = base_params.retirement_age - base_params.start_age if use_years else base_params.retirement_age

    # Plot 1: Stock Weight comparison
    ax = axes[0, 0]
    for i, gamma in enumerate(gamma_values):
        ax.plot(x, results[gamma].stock_weight_no_short, color=gamma_colors[i],
                linewidth=2, label=f'gamma = {gamma}')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weight')
    ax.set_title('Stock Weight by Risk Aversion')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.05, 1.15)

    # Plot 2: Bond Weight comparison
    ax = axes[0, 1]
    for i, gamma in enumerate(gamma_values):
        ax.plot(x, results[gamma].bond_weight_no_short, color=gamma_colors[i],
                linewidth=2, label=f'gamma = {gamma}')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weight')
    ax.set_title('Bond Weight by Risk Aversion')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.05, 1.15)

    # Plot 3: Financial Wealth comparison
    ax = axes[0, 2]
    for i, gamma in enumerate(gamma_values):
        ax.plot(x, results[gamma].financial_wealth, color=gamma_colors[i],
                linewidth=2, label=f'gamma = {gamma}')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth by Risk Aversion')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 4: Total Wealth comparison
    ax = axes[1, 0]
    for i, gamma in enumerate(gamma_values):
        ax.plot(x, results[gamma].total_wealth, color=gamma_colors[i],
                linewidth=2, label=f'gamma = {gamma}')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Wealth by Risk Aversion')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 5: Total Consumption comparison
    ax = axes[1, 1]
    for i, gamma in enumerate(gamma_values):
        ax.plot(x, results[gamma].total_consumption, color=gamma_colors[i],
                linewidth=2, label=f'gamma = {gamma}')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Consumption by Risk Aversion')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 6: Net Worth comparison
    ax = axes[1, 2]
    for i, gamma in enumerate(gamma_values):
        ax.plot(x, results[gamma].net_worth, color=gamma_colors[i],
                linewidth=2, label=f'gamma = {gamma}')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Net Worth by Risk Aversion')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


def create_initial_wealth_comparison_figure(
    wealth_values: list = [-50, 0, 50, 200],
    base_params: 'LifecycleParams' = None,
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (16, 12),
    use_years: bool = True
) -> plt.Figure:
    """
    Create a figure comparing scenarios with different initial wealth levels.

    Useful for comparing student loan debt (-50k) to various savings levels.
    """
    from core import LifecycleParams, EconomicParams, compute_lifecycle_median_path

    if base_params is None:
        base_params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    # Compute results for each initial wealth level
    results = {}
    for wealth in wealth_values:
        params = LifecycleParams(
            start_age=base_params.start_age,
            retirement_age=base_params.retirement_age,
            end_age=base_params.end_age,
            initial_earnings=base_params.initial_earnings,
            earnings_growth=base_params.earnings_growth,
            earnings_hump_age=base_params.earnings_hump_age,
            earnings_decline=base_params.earnings_decline,
            base_expenses=base_params.base_expenses,
            expense_growth=base_params.expense_growth,
            retirement_expenses=base_params.retirement_expenses,
            stock_beta_human_capital=base_params.stock_beta_human_capital,
            gamma=base_params.gamma,
            consumption_boost=base_params.consumption_boost,
            initial_wealth=wealth,
        )
        results[wealth] = compute_lifecycle_median_path(params, econ_params)

    # Create figure with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Colors for different wealth levels
    wealth_colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(wealth_values)))

    # Get x-axis values
    result_0 = results[wealth_values[0]]
    if use_years:
        x = np.arange(len(result_0.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result_0.ages
        xlabel = 'Age'

    retirement_x = base_params.retirement_age - base_params.start_age if use_years else base_params.retirement_age

    # Plot 1: Financial Wealth comparison
    ax = axes[0, 0]
    for i, wealth in enumerate(wealth_values):
        label = f'${wealth:+}k' if wealth != 0 else '$0k'
        ax.plot(x, results[wealth].financial_wealth, color=wealth_colors[i],
                linewidth=2, label=label)
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth by Initial Wealth')
    ax.legend(loc='upper left', fontsize=9)

    # Plot 2: Total Wealth comparison
    ax = axes[0, 1]
    for i, wealth in enumerate(wealth_values):
        label = f'${wealth:+}k' if wealth != 0 else '$0k'
        ax.plot(x, results[wealth].total_wealth, color=wealth_colors[i],
                linewidth=2, label=label)
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Wealth by Initial Wealth')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 3: Net Worth comparison
    ax = axes[0, 2]
    for i, wealth in enumerate(wealth_values):
        label = f'${wealth:+}k' if wealth != 0 else '$0k'
        ax.plot(x, results[wealth].net_worth, color=wealth_colors[i],
                linewidth=2, label=label)
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Net Worth by Initial Wealth')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 4: Total Consumption comparison
    ax = axes[1, 0]
    for i, wealth in enumerate(wealth_values):
        label = f'${wealth:+}k' if wealth != 0 else '$0k'
        ax.plot(x, results[wealth].total_consumption, color=wealth_colors[i],
                linewidth=2, label=label)
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Consumption by Initial Wealth')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 5: Stock Weight comparison
    ax = axes[1, 1]
    for i, wealth in enumerate(wealth_values):
        label = f'${wealth:+}k' if wealth != 0 else '$0k'
        ax.plot(x, results[wealth].stock_weight_no_short, color=wealth_colors[i],
                linewidth=2, label=label)
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weight')
    ax.set_title('Stock Weight by Initial Wealth')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.05, 1.15)

    # Plot 6: Variable Consumption comparison
    ax = axes[1, 2]
    for i, wealth in enumerate(wealth_values):
        label = f'${wealth:+}k' if wealth != 0 else '$0k'
        ax.plot(x, results[wealth].variable_consumption, color=wealth_colors[i],
                linewidth=2, label=label)
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Variable Consumption by Initial Wealth')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


def create_consumption_boost_comparison_figure(
    boost_values: list = [0.0, 0.01, 0.02, 0.03],
    base_params: 'LifecycleParams' = None,
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (16, 12),
    use_years: bool = True
) -> plt.Figure:
    """
    Create a figure comparing scenarios with different consumption boost levels.

    The boost is added to median return to get consumption rate.
    """
    from core import LifecycleParams, EconomicParams, compute_lifecycle_median_path

    if base_params is None:
        base_params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    # Compute results for each boost level
    results = {}
    for boost in boost_values:
        params = LifecycleParams(
            start_age=base_params.start_age,
            retirement_age=base_params.retirement_age,
            end_age=base_params.end_age,
            initial_earnings=base_params.initial_earnings,
            earnings_growth=base_params.earnings_growth,
            earnings_hump_age=base_params.earnings_hump_age,
            earnings_decline=base_params.earnings_decline,
            base_expenses=base_params.base_expenses,
            expense_growth=base_params.expense_growth,
            retirement_expenses=base_params.retirement_expenses,
            stock_beta_human_capital=base_params.stock_beta_human_capital,
            gamma=base_params.gamma,
            consumption_boost=boost,
            initial_wealth=base_params.initial_wealth,
        )
        results[boost] = compute_lifecycle_median_path(params, econ_params)

    # Create figure with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Colors for different boost levels
    boost_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(boost_values)))

    # Get x-axis values
    result_0 = results[boost_values[0]]
    if use_years:
        x = np.arange(len(result_0.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result_0.ages
        xlabel = 'Age'

    retirement_x = base_params.retirement_age - base_params.start_age if use_years else base_params.retirement_age

    # Plot 1: Financial Wealth comparison
    ax = axes[0, 0]
    for i, boost in enumerate(boost_values):
        ax.plot(x, results[boost].financial_wealth, color=boost_colors[i],
                linewidth=2, label=f'+{boost*100:.0f}pp')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth by Consumption Boost')
    ax.legend(loc='upper left', fontsize=9)

    # Plot 2: Total Consumption comparison
    ax = axes[0, 1]
    for i, boost in enumerate(boost_values):
        ax.plot(x, results[boost].total_consumption, color=boost_colors[i],
                linewidth=2, label=f'+{boost*100:.0f}pp')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Consumption by Consumption Boost')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 3: Variable Consumption comparison
    ax = axes[0, 2]
    for i, boost in enumerate(boost_values):
        ax.plot(x, results[boost].variable_consumption, color=boost_colors[i],
                linewidth=2, label=f'+{boost*100:.0f}pp')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Variable Consumption by Consumption Boost')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 4: Savings (Earnings - Consumption)
    ax = axes[1, 0]
    for i, boost in enumerate(boost_values):
        ax.plot(x, results[boost].savings, color=boost_colors[i],
                linewidth=2, label=f'+{boost*100:.0f}pp')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Savings by Consumption Boost')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 5: Net Worth comparison
    ax = axes[1, 1]
    for i, boost in enumerate(boost_values):
        ax.plot(x, results[boost].net_worth, color=boost_colors[i],
                linewidth=2, label=f'+{boost*100:.0f}pp')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Net Worth by Consumption Boost')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 6: Total Wealth comparison
    ax = axes[1, 2]
    for i, boost in enumerate(boost_values):
        ax.plot(x, results[boost].total_wealth, color=boost_colors[i],
                linewidth=2, label=f'+{boost*100:.0f}pp')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Wealth by Consumption Boost')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


def create_equity_premium_comparison_figure(
    premium_values: list = [0.02, 0.04, 0.06, 0.08],
    base_params: 'LifecycleParams' = None,
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (16, 12),
    use_years: bool = True
) -> plt.Figure:
    """
    Create a figure comparing scenarios with different equity risk premiums.
    """
    from core import LifecycleParams, EconomicParams, compute_lifecycle_median_path

    if base_params is None:
        base_params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    # Compute results for each premium level
    results = {}
    for premium in premium_values:
        custom_econ = EconomicParams(
            r_bar=econ_params.r_bar,
            mu_excess=premium,
            mu_bond=econ_params.mu_bond,
            sigma_s=econ_params.sigma_s,
            sigma_r=econ_params.sigma_r,
            rho=econ_params.rho,
            bond_duration=econ_params.bond_duration,
        )
        results[premium] = compute_lifecycle_median_path(base_params, custom_econ)

    # Create figure with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Colors for different premium levels
    premium_colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(premium_values)))

    # Get x-axis values
    result_0 = results[premium_values[0]]
    if use_years:
        x = np.arange(len(result_0.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result_0.ages
        xlabel = 'Age'

    retirement_x = base_params.retirement_age - base_params.start_age if use_years else base_params.retirement_age

    # Plot 1: Stock Weight comparison
    ax = axes[0, 0]
    for i, premium in enumerate(premium_values):
        ax.plot(x, results[premium].stock_weight_no_short, color=premium_colors[i],
                linewidth=2, label=f'{premium*100:.0f}%')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weight')
    ax.set_title('Stock Weight by Equity Premium')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.05, 1.15)

    # Plot 2: Bond Weight comparison
    ax = axes[0, 1]
    for i, premium in enumerate(premium_values):
        ax.plot(x, results[premium].bond_weight_no_short, color=premium_colors[i],
                linewidth=2, label=f'{premium*100:.0f}%')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weight')
    ax.set_title('Bond Weight by Equity Premium')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.05, 1.15)

    # Plot 3: Financial Wealth comparison
    ax = axes[0, 2]
    for i, premium in enumerate(premium_values):
        ax.plot(x, results[premium].financial_wealth, color=premium_colors[i],
                linewidth=2, label=f'{premium*100:.0f}%')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth by Equity Premium')
    ax.legend(loc='upper left', fontsize=9)

    # Plot 4: Total Wealth comparison
    ax = axes[1, 0]
    for i, premium in enumerate(premium_values):
        ax.plot(x, results[premium].total_wealth, color=premium_colors[i],
                linewidth=2, label=f'{premium*100:.0f}%')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Wealth by Equity Premium')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 5: Total Consumption comparison
    ax = axes[1, 1]
    for i, premium in enumerate(premium_values):
        ax.plot(x, results[premium].total_consumption, color=premium_colors[i],
                linewidth=2, label=f'{premium*100:.0f}%')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Consumption by Equity Premium')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 6: HC Stock Component comparison
    ax = axes[1, 2]
    for i, premium in enumerate(premium_values):
        ax.plot(x, results[premium].hc_stock_component, color=premium_colors[i],
                linewidth=2, label=f'{premium*100:.0f}%')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('HC Stock Component by Equity Premium')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


def create_income_comparison_figure(
    earnings_values: list = [60, 100, 150, 200],
    base_params: 'LifecycleParams' = None,
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (16, 12),
    use_years: bool = True
) -> plt.Figure:
    """
    Create a figure comparing scenarios with different initial earnings levels.
    """
    from core import LifecycleParams, EconomicParams, compute_lifecycle_median_path

    if base_params is None:
        base_params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    # Compute results for each earnings level
    results = {}
    for earnings in earnings_values:
        params = LifecycleParams(
            start_age=base_params.start_age,
            retirement_age=base_params.retirement_age,
            end_age=base_params.end_age,
            initial_earnings=earnings,
            earnings_growth=base_params.earnings_growth,
            earnings_hump_age=base_params.earnings_hump_age,
            earnings_decline=base_params.earnings_decline,
            base_expenses=base_params.base_expenses,
            expense_growth=base_params.expense_growth,
            retirement_expenses=base_params.retirement_expenses,
            stock_beta_human_capital=base_params.stock_beta_human_capital,
            gamma=base_params.gamma,
            consumption_boost=base_params.consumption_boost,
            initial_wealth=base_params.initial_wealth,
        )
        results[earnings] = compute_lifecycle_median_path(params, econ_params)

    # Create figure with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Colors for different earnings levels
    earnings_colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(earnings_values)))

    # Get x-axis values
    result_0 = results[earnings_values[0]]
    if use_years:
        x = np.arange(len(result_0.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result_0.ages
        xlabel = 'Age'

    retirement_x = base_params.retirement_age - base_params.start_age if use_years else base_params.retirement_age

    # Plot 1: Earnings Profile comparison
    ax = axes[0, 0]
    for i, earnings in enumerate(earnings_values):
        ax.plot(x, results[earnings].earnings, color=earnings_colors[i],
                linewidth=2, label=f'${earnings}k')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Earnings Profile by Initial Earnings')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 2: Human Capital comparison
    ax = axes[0, 1]
    for i, earnings in enumerate(earnings_values):
        ax.plot(x, results[earnings].human_capital, color=earnings_colors[i],
                linewidth=2, label=f'${earnings}k')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Human Capital by Initial Earnings')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 3: Financial Wealth comparison
    ax = axes[0, 2]
    for i, earnings in enumerate(earnings_values):
        ax.plot(x, results[earnings].financial_wealth, color=earnings_colors[i],
                linewidth=2, label=f'${earnings}k')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth by Initial Earnings')
    ax.legend(loc='upper left', fontsize=9)

    # Plot 4: Total Consumption comparison
    ax = axes[1, 0]
    for i, earnings in enumerate(earnings_values):
        ax.plot(x, results[earnings].total_consumption, color=earnings_colors[i],
                linewidth=2, label=f'${earnings}k')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Consumption by Initial Earnings')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 5: Savings comparison
    ax = axes[1, 1]
    for i, earnings in enumerate(earnings_values):
        ax.plot(x, results[earnings].savings, color=earnings_colors[i],
                linewidth=2, label=f'${earnings}k')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Savings by Initial Earnings')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 6: Stock Weight comparison
    ax = axes[1, 2]
    for i, earnings in enumerate(earnings_values):
        ax.plot(x, results[earnings].stock_weight_no_short, color=earnings_colors[i],
                linewidth=2, label=f'${earnings}k')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weight')
    ax.set_title('Stock Weight by Initial Earnings')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.05, 1.15)

    plt.tight_layout()
    return fig


def create_volatility_comparison_figure(
    volatility_values: list = [0.12, 0.18, 0.24, 0.30],
    base_params: 'LifecycleParams' = None,
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (16, 12),
    use_years: bool = True
) -> plt.Figure:
    """
    Create a figure comparing scenarios with different stock volatility levels.
    """
    from core import LifecycleParams, EconomicParams, compute_lifecycle_median_path

    if base_params is None:
        base_params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    # Compute results for each volatility level
    results = {}
    for vol in volatility_values:
        custom_econ = EconomicParams(
            r_bar=econ_params.r_bar,
            mu_excess=econ_params.mu_excess,
            mu_bond=econ_params.mu_bond,
            sigma_s=vol,
            sigma_r=econ_params.sigma_r,
            rho=econ_params.rho,
            bond_duration=econ_params.bond_duration,
        )
        results[vol] = compute_lifecycle_median_path(base_params, custom_econ)

    # Create figure with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Colors for different volatility levels
    vol_colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(volatility_values)))

    # Get x-axis values
    result_0 = results[volatility_values[0]]
    if use_years:
        x = np.arange(len(result_0.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result_0.ages
        xlabel = 'Age'

    retirement_x = base_params.retirement_age - base_params.start_age if use_years else base_params.retirement_age

    # Plot 1: Stock Weight comparison
    ax = axes[0, 0]
    for i, vol in enumerate(volatility_values):
        ax.plot(x, results[vol].stock_weight_no_short, color=vol_colors[i],
                linewidth=2, label=f'{vol*100:.0f}%')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weight')
    ax.set_title('Stock Weight by Stock Volatility')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.05, 1.15)

    # Plot 2: Bond Weight comparison
    ax = axes[0, 1]
    for i, vol in enumerate(volatility_values):
        ax.plot(x, results[vol].bond_weight_no_short, color=vol_colors[i],
                linewidth=2, label=f'{vol*100:.0f}%')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weight')
    ax.set_title('Bond Weight by Stock Volatility')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.05, 1.15)

    # Plot 3: Cash Weight comparison
    ax = axes[0, 2]
    for i, vol in enumerate(volatility_values):
        ax.plot(x, results[vol].cash_weight_no_short, color=vol_colors[i],
                linewidth=2, label=f'{vol*100:.0f}%')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weight')
    ax.set_title('Cash Weight by Stock Volatility')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.05, 1.15)

    # Plot 4: Financial Wealth comparison
    ax = axes[1, 0]
    for i, vol in enumerate(volatility_values):
        ax.plot(x, results[vol].financial_wealth, color=vol_colors[i],
                linewidth=2, label=f'{vol*100:.0f}%')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth by Stock Volatility')
    ax.legend(loc='upper left', fontsize=9)

    # Plot 5: Total Consumption comparison
    ax = axes[1, 1]
    for i, vol in enumerate(volatility_values):
        ax.plot(x, results[vol].total_consumption, color=vol_colors[i],
                linewidth=2, label=f'{vol*100:.0f}%')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Consumption by Stock Volatility')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 6: Total Wealth comparison
    ax = axes[1, 2]
    for i, vol in enumerate(volatility_values):
        ax.plot(x, results[vol].total_wealth, color=vol_colors[i],
                linewidth=2, label=f'{vol*100:.0f}%')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Wealth by Stock Volatility')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig
