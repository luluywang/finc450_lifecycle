"""
Monte Carlo visualization plots for lifecycle analysis.

This module provides plotting functions for Monte Carlo simulation results
including fan charts, distributions, and teaching scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, TYPE_CHECKING

from .styles import COLORS
from .helpers import apply_wealth_log_scale

if TYPE_CHECKING:
    from core import MonteCarloResult, LifecycleParams, SimulationResult


def create_monte_carlo_fan_chart(
    mc_result: 'MonteCarloResult',
    params: 'LifecycleParams' = None,
    figsize: Tuple[int, int] = (16, 14),
    use_years: bool = True,
    n_sample_paths: int = 50,
) -> plt.Figure:
    """
    Create fan charts showing Monte Carlo simulation results.

    Shows percentile bands for wealth and consumption across simulated paths.

    Args:
        mc_result: Monte Carlo simulation results
        params: Lifecycle parameters
        figsize: Figure size
        use_years: If True, use years from start on x-axis
        n_sample_paths: Number of sample paths to show as thin lines

    Returns:
        matplotlib Figure
    """
    from core import LifecycleParams as LP
    if params is None:
        params = LP()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    total_years = len(mc_result.ages)
    if use_years:
        x = np.arange(total_years)
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = mc_result.ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    percentiles = [10, 25, 50, 75, 90]

    # Panel 1: Financial Wealth Fan Chart
    ax = axes[0, 0]
    fw_percentiles = np.percentile(mc_result.financial_wealth_paths, percentiles, axis=0)

    for i in range(min(n_sample_paths, mc_result.financial_wealth_paths.shape[0])):
        ax.plot(x, mc_result.financial_wealth_paths[i, :], color='blue', alpha=0.1, linewidth=0.5)

    ax.fill_between(x, fw_percentiles[0], fw_percentiles[4], alpha=0.2, color='blue', label='10-90th pctl')
    ax.fill_between(x, fw_percentiles[1], fw_percentiles[3], alpha=0.3, color='blue', label='25-75th pctl')
    ax.plot(x, fw_percentiles[2], color='darkblue', linewidth=2, label='Median')
    ax.plot(x, mc_result.median_result.financial_wealth, color='#2A9D8F', linewidth=2,
            linestyle='--', label='Deterministic')

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='#E07A5F', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth: Monte Carlo Fan Chart')
    ax.legend(loc='upper left', fontsize=8)
    apply_wealth_log_scale(ax)

    # Panel 2: Consumption Fan Chart
    ax = axes[0, 1]
    cons_percentiles = np.percentile(mc_result.total_consumption_paths, percentiles, axis=0)

    for i in range(min(n_sample_paths, mc_result.total_consumption_paths.shape[0])):
        ax.plot(x, mc_result.total_consumption_paths[i, :], color='orange', alpha=0.1, linewidth=0.5)

    ax.fill_between(x, cons_percentiles[0], cons_percentiles[4], alpha=0.2, color='orange', label='10-90th pctl')
    ax.fill_between(x, cons_percentiles[1], cons_percentiles[3], alpha=0.3, color='orange', label='25-75th pctl')
    ax.plot(x, cons_percentiles[2], color='darkorange', linewidth=2, label='Median')
    ax.plot(x, mc_result.median_result.total_consumption, color='#2A9D8F', linewidth=2,
            linestyle='--', label='Deterministic')

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Consumption: Monte Carlo Fan Chart')
    ax.legend(loc='upper right', fontsize=8)
    apply_wealth_log_scale(ax)

    # Panel 3: Final Wealth Distribution
    ax = axes[1, 0]
    final_wealth = mc_result.final_wealth
    non_default = final_wealth[~mc_result.default_flags]

    floor_val = 10
    non_default_floored = np.maximum(non_default, floor_val)
    max_val = np.percentile(non_default_floored, 99)
    bins = np.geomspace(floor_val, max_val, 50)
    ax.hist(non_default_floored, bins=bins, alpha=0.7, color='#1A759F', edgecolor='white', density=True)
    ax.axvline(x=max(np.median(non_default), floor_val), color='#E9C46A', linestyle='--', linewidth=2,
               label=f'Median: ${np.median(non_default):,.0f}k')
    ax.axvline(x=max(np.mean(non_default), floor_val), color='#2A9D8F', linestyle='--', linewidth=2,
               label=f'Mean: ${np.mean(non_default):,.0f}k')

    ax.set_xlabel('Final Wealth ($ 000s)')
    ax.set_ylabel('Density')
    ax.set_title(f'Final Wealth Distribution (Non-Defaulted, n={len(non_default)})')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xscale('log')

    # Panel 4: Summary Statistics
    ax = axes[1, 1]
    ax.axis('off')

    n_sims = len(mc_result.default_flags)
    n_defaults = np.sum(mc_result.default_flags)
    default_rate = 100 * n_defaults / n_sims

    total_cons_median = np.median(mc_result.total_lifetime_consumption)
    total_cons_p10 = np.percentile(mc_result.total_lifetime_consumption, 10)
    total_cons_p90 = np.percentile(mc_result.total_lifetime_consumption, 90)

    final_w_median = np.median(mc_result.final_wealth)
    final_w_p10 = np.percentile(mc_result.final_wealth, 10)
    final_w_p90 = np.percentile(mc_result.final_wealth, 90)

    summary_text = f"""
Monte Carlo Simulation Summary
==============================

Simulation Parameters:
  - Number of Paths: {n_sims:,}
  - Total Years: {total_years}

Target Allocation (MV Optimal):
  - Stocks: {mc_result.target_stock*100:.1f}%
  - Bonds: {mc_result.target_bond*100:.1f}%
  - Cash: {mc_result.target_cash*100:.1f}%

Outcomes:
  - Default Rate: {default_rate:.1f}%
  - Defaults: {n_defaults:,} / {n_sims:,}

Lifetime Consumption ($ 000s):
  - Median: ${total_cons_median:,.0f}k
  - 10th Pctl: ${total_cons_p10:,.0f}k
  - 90th Pctl: ${total_cons_p90:,.0f}k

Final Wealth ($ 000s):
  - Median: ${final_w_median:,.0f}k
  - 10th Pctl: ${final_w_p10:,.0f}k
  - 90th Pctl: ${final_w_p90:,.0f}k
"""

    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    return fig


def create_monte_carlo_detailed_view(
    mc_result: 'MonteCarloResult',
    params: 'LifecycleParams' = None,
    figsize: Tuple[int, int] = (16, 14),
    use_years: bool = True,
) -> plt.Figure:
    """
    Create detailed Monte Carlo visualization with portfolio weights and returns.

    Args:
        mc_result: Monte Carlo simulation results
        params: Lifecycle parameters
        figsize: Figure size
        use_years: If True, use years from start on x-axis

    Returns:
        matplotlib Figure
    """
    from core import LifecycleParams as LP
    if params is None:
        params = LP()

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    total_years = len(mc_result.ages)
    if use_years:
        x = np.arange(total_years)
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = mc_result.ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    percentiles = [10, 50, 90]

    # Panel 1: Stock Returns
    ax = axes[0, 0]
    n_returns = mc_result.stock_return_paths.shape[1]
    ret_percentiles = np.percentile(mc_result.stock_return_paths, percentiles, axis=0)
    x_ret = x[:n_returns] if len(x) > n_returns else x
    ax.fill_between(x_ret, ret_percentiles[0]*100, ret_percentiles[2]*100,
                    alpha=0.3, color='#2A9D8F', label='10-90th pctl')
    ax.plot(x_ret, ret_percentiles[1]*100, color='#264653', linewidth=2, label='Median')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Return (%)')
    ax.set_title('Stock Returns Over Time')
    ax.legend(loc='upper right', fontsize=8)

    # Panel 2: Interest Rates
    ax = axes[0, 1]
    n_rates = mc_result.interest_rate_paths.shape[1]
    rate_percentiles = np.percentile(mc_result.interest_rate_paths, percentiles, axis=0)
    x_rate = x[:n_rates] if len(x) >= n_rates else np.arange(n_rates)
    ax.fill_between(x_rate, rate_percentiles[0]*100, rate_percentiles[2]*100,
                    alpha=0.3, color='blue', label='10-90th pctl')
    ax.plot(x_rate, rate_percentiles[1]*100, color='darkblue', linewidth=2, label='Median')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Rate (%)')
    ax.set_title('Interest Rate Paths')
    ax.legend(loc='upper right', fontsize=8)

    # Panel 3: Stock Weight Evolution
    ax = axes[0, 2]
    weight_percentiles = np.percentile(mc_result.stock_weight_paths, percentiles, axis=0)
    ax.fill_between(x, weight_percentiles[0]*100, weight_percentiles[2]*100,
                    alpha=0.3, color='#F4A261', label='10-90th pctl')
    ax.plot(x, weight_percentiles[1]*100, color='#BC6C25', linewidth=2, label='Median')
    ax.plot(x, mc_result.median_result.stock_weight_no_short*100, color='#2A9D8F',
            linewidth=2, linestyle='--', label='Deterministic')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=mc_result.target_stock*100, color='gray', linestyle='--', alpha=0.5,
               label=f'Target ({mc_result.target_stock*100:.0f}%)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weight (%)')
    ax.set_title('Stock Allocation Over Lifecycle')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 105)

    # Panel 4: Total Wealth
    ax = axes[1, 0]
    tw_percentiles = np.percentile(mc_result.total_wealth_paths, percentiles, axis=0)
    ax.fill_between(x, tw_percentiles[0], tw_percentiles[2],
                    alpha=0.3, color='purple', label='10-90th pctl')
    ax.plot(x, tw_percentiles[1], color='purple', linewidth=2, label='Median')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Wealth (HC + FW)')
    ax.legend(loc='upper right', fontsize=8)
    apply_wealth_log_scale(ax)

    # Panel 5: Cumulative Consumption
    ax = axes[1, 1]
    cum_cons = np.cumsum(mc_result.total_consumption_paths, axis=1)
    cum_percentiles = np.percentile(cum_cons, percentiles, axis=0)
    ax.fill_between(x, cum_percentiles[0], cum_percentiles[2],
                    alpha=0.3, color='orange', label='10-90th pctl')
    ax.plot(x, cum_percentiles[1], color='darkorange', linewidth=2, label='Median')
    cum_det = np.cumsum(mc_result.median_result.total_consumption)
    ax.plot(x, cum_det, color='#2A9D8F', linewidth=2, linestyle='--', label='Deterministic')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Cumulative Lifetime Consumption')
    ax.legend(loc='upper left', fontsize=8)

    # Panel 6: Default Analysis
    ax = axes[1, 2]
    default_ages = mc_result.default_ages[~np.isnan(mc_result.default_ages)]
    if len(default_ages) > 0:
        ax.hist(default_ages, bins=20, alpha=0.7, color='#E07A5F', edgecolor='white')
        ax.axvline(x=params.retirement_age, color='gray', linestyle='--', linewidth=2,
                   label='Retirement Age')
        ax.set_xlabel('Age at Default')
        ax.set_ylabel('Count')
        ax.set_title(f'Default Age Distribution (n={len(default_ages)})')
        ax.legend(loc='upper right', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No Defaults Observed', transform=ax.transAxes,
                fontsize=14, ha='center', va='center')
        ax.set_title('Default Analysis')

    plt.tight_layout()
    return fig


def plot_wealth_paths_spaghetti(
    results: Dict[str, 'SimulationResult'],
    n_paths: int = 100,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Spaghetti plot showing sample wealth trajectories for each strategy.
    """
    from .styles import STRATEGY_COLORS

    strategies = list(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        result = results[strategy]
        n_periods = result.wealth_paths.shape[1]
        years = np.arange(n_periods)

        for i in range(min(n_paths, result.wealth_paths.shape[0])):
            alpha = 0.3 if result.defaulted[i] else 0.15
            color = '#E07A5F' if result.defaulted[i] else STRATEGY_COLORS[idx]
            ax.plot(years, result.wealth_paths[i, :] / 1e6,
                   color=color, alpha=alpha, linewidth=0.5)

        median_path = np.median(result.wealth_paths, axis=0)
        ax.plot(years, median_path / 1e6, 'k-', linewidth=2, label='Median')

        p10 = np.percentile(result.wealth_paths, 10, axis=0)
        p90 = np.percentile(result.wealth_paths, 90, axis=0)
        ax.fill_between(years, p10 / 1e6, p90 / 1e6, alpha=0.2, color=STRATEGY_COLORS[idx])

        ax.set_xlabel('Year')
        ax.set_ylabel('Wealth ($M)')
        ax.set_title(f'{strategy}')
        ax.set_xlim(0, n_periods - 1)
        ax.set_ylim(0, None)
        ax.axhline(y=0, color='#E07A5F', linestyle='--', alpha=0.5)

        default_rate = 100 * result.defaulted.mean()
        ax.annotate(f'Default: {default_rate:.1f}%',
                   xy=(0.02, 0.98), xycoords='axes fraction',
                   fontsize=10, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def plot_final_wealth_distribution(
    results: Dict[str, 'SimulationResult'],
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Histogram of final wealth distribution across strategies.
    """
    from .styles import STRATEGY_COLORS

    fig, ax = plt.subplots(figsize=figsize)

    strategies = list(results.keys())

    for idx, strategy in enumerate(strategies):
        result = results[strategy]
        final_wealth = result.final_wealth[result.final_wealth > 0] / 1e6

        ax.hist(final_wealth, bins=50, alpha=0.5, label=strategy,
               color=STRATEGY_COLORS[idx], density=True)

    ax.set_xlabel('Final Wealth ($M)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Final Wealth (Non-Defaulted Paths)')
    ax.legend()

    plt.tight_layout()
    return fig


def create_teaching_scenarios_figure(
    scenarios: List,  # List of ScenarioResult
    params: 'LifecycleParams' = None,
    figsize: Tuple[int, int] = (18, 14),
    use_years: bool = True,
) -> plt.Figure:
    """
    Create visualization comparing teaching scenarios.

    Shows how different return sequences affect wealth and consumption outcomes.
    """
    from core import LifecycleParams as LP
    if params is None:
        params = LP()

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    total_years = len(scenarios[0].ages)
    if use_years:
        x = np.arange(total_years)
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = scenarios[0].ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    colors = plt.cm.tab10(np.linspace(0, 0.5, len(scenarios)))

    # Panel 1: Stock Returns by Scenario
    ax = axes[0, 0]
    for i, scenario in enumerate(scenarios):
        ax.plot(x, scenario.stock_returns * 100, color=colors[i], linewidth=1.5,
                label=scenario.name, alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Return (%)')
    ax.set_title('Stock Returns by Scenario')
    ax.legend(loc='lower left', fontsize=8)

    # Panel 2: Financial Wealth
    ax = axes[0, 1]
    for i, scenario in enumerate(scenarios):
        ax.plot(x, scenario.financial_wealth, color=colors[i], linewidth=2,
                label=scenario.name)
    ax.axhline(y=0, color='#E07A5F', linestyle='-', alpha=0.3)
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth by Scenario')
    ax.legend(loc='upper left', fontsize=8)

    # Panel 3: Annual Consumption
    ax = axes[0, 2]
    for i, scenario in enumerate(scenarios):
        ax.plot(x, scenario.total_consumption, color=colors[i], linewidth=2,
                label=scenario.name)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Annual Consumption by Scenario')
    ax.legend(loc='upper right', fontsize=8)

    # Panel 4: Cumulative Consumption
    ax = axes[1, 0]
    for i, scenario in enumerate(scenarios):
        ax.plot(x, scenario.cumulative_consumption, color=colors[i], linewidth=2,
                label=scenario.name)
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Cumulative Consumption by Scenario')
    ax.legend(loc='upper left', fontsize=8)

    # Panel 5: Stock Weight Evolution
    ax = axes[1, 1]
    for i, scenario in enumerate(scenarios):
        ax.plot(x, scenario.stock_weight * 100, color=colors[i], linewidth=2,
                label=scenario.name)
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weight (%)')
    ax.set_title('Stock Allocation by Scenario')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 105)

    # Panel 6: Summary Table
    ax = axes[1, 2]
    ax.axis('off')

    table_data = []
    headers = ['Scenario', 'Final Wealth', 'Total Cons', 'Peak Cons']

    for scenario in scenarios:
        final_w = scenario.financial_wealth[-1]
        total_c = scenario.cumulative_consumption[-1]
        peak_c = np.max(scenario.total_consumption)
        table_data.append([
            scenario.name[:15],
            f'${final_w:,.0f}k',
            f'${total_c:,.0f}k',
            f'${peak_c:,.0f}k'
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Scenario Outcomes Summary', pad=20)

    plt.tight_layout()
    return fig


def create_sequence_of_returns_figure(
    params: 'LifecycleParams' = None,
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (16, 10),
    use_years: bool = True,
) -> plt.Figure:
    """
    Create visualization specifically illustrating sequence of returns risk.

    Compares two scenarios with identical average returns but different sequencing.
    """
    from core import LifecycleParams as LP, EconomicParams as EP, compute_lifecycle_median_path

    if params is None:
        params = LP()
    if econ_params is None:
        econ_params = EP()

    total_years = params.end_age - params.start_age
    working_years = params.retirement_age - params.start_age
    retirement_years = total_years - working_years

    good_return = 0.15
    bad_return = -0.05
    avg_return = (good_return + bad_return) / 2

    # Sequence 1: Good early in retirement, bad late
    good_early = np.full(total_years, avg_return)
    good_early[working_years:working_years + retirement_years // 2] = good_return
    good_early[working_years + retirement_years // 2:] = bad_return

    # Sequence 2: Bad early in retirement, good late
    bad_early = np.full(total_years, avg_return)
    bad_early[working_years:working_years + retirement_years // 2] = bad_return
    bad_early[working_years + retirement_years // 2:] = good_return

    # Sequence 3: Constant average
    constant = np.full(total_years, avg_return)

    # Import the scenario creation function
    from lifecycle_strategy import create_teaching_scenario

    scenarios = [
        create_teaching_scenario("Good Early", "Strong returns at start of retirement",
                                 good_early, params, econ_params),
        create_teaching_scenario("Bad Early (Seq Risk)", "Poor returns at start of retirement",
                                 bad_early, params, econ_params),
        create_teaching_scenario("Constant", "Same average return every year",
                                 constant, params, econ_params),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    if use_years:
        x = np.arange(total_years)
        xlabel = 'Years from Career Start'
        retirement_x = working_years
    else:
        x = scenarios[0].ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    colors = ['#1A759F', '#E9C46A', '#2A9D8F']  # Colorblind-safe: blue, amber, teal

    # Panel 1: Return Sequences
    ax = axes[0, 0]
    for i, scenario in enumerate(scenarios):
        ax.plot(x, scenario.stock_returns * 100, color=colors[i], linewidth=2,
                label=scenario.name)
    ax.axhline(y=avg_return * 100, color='gray', linestyle='--', alpha=0.7,
               label=f'Avg: {avg_return*100:.0f}%')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5, label='Retirement')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Return (%)')
    ax.set_title('Return Sequences (Same Average)')
    ax.legend(loc='lower left', fontsize=9)

    # Panel 2: Financial Wealth
    ax = axes[0, 1]
    for i, scenario in enumerate(scenarios):
        ax.plot(x, scenario.financial_wealth, color=colors[i], linewidth=2,
                label=scenario.name)
    ax.axhline(y=0, color='#E07A5F', linestyle='-', alpha=0.3)
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth: Sequence of Returns Risk')
    ax.legend(loc='upper left', fontsize=9)

    # Panel 3: Consumption
    ax = axes[1, 0]
    for i, scenario in enumerate(scenarios):
        ax.plot(x, scenario.total_consumption, color=colors[i], linewidth=2,
                label=scenario.name)
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Annual Consumption')
    ax.legend(loc='upper right', fontsize=9)

    # Panel 4: Key Insight Text
    ax = axes[1, 1]
    ax.axis('off')

    insight_text = f"""
SEQUENCE OF RETURNS RISK
========================

Key Insight:
-----------
The ORDER of returns matters, not just the average!

All three scenarios have the same average return
({avg_return*100:.0f}% per year), but vastly different outcomes.

Results:
--------
Good Early: Final Wealth = ${scenarios[0].financial_wealth[-1]:,.0f}k
Bad Early:  Final Wealth = ${scenarios[1].financial_wealth[-1]:,.0f}k
Constant:   Final Wealth = ${scenarios[2].financial_wealth[-1]:,.0f}k

Why It Matters:
--------------
1. Early in retirement, the portfolio is largest
2. Poor early returns destroy wealth when it's biggest
3. There's no time to recover from early losses
4. Variable consumption helps but doesn't eliminate risk

Practical Implications:
----------------------
- Reduce equity exposure near retirement
- Consider liability matching for essential expenses
- Buffer assets can help weather early volatility
"""

    ax.text(0.05, 0.95, insight_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Sequence of Returns Risk: Why Timing Matters',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig
