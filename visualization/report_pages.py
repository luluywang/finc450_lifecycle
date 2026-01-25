"""
PDF Report Page Creation Functions.

This module contains functions that create multi-panel figure pages for PDF reports.
These are higher-level layouts that combine multiple charts into a single page.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from core import LifecycleParams, EconomicParams, LifecycleResult, MonteCarloResult

from .helpers import apply_wealth_log_scale


# Standard color palette for charts
REPORT_COLORS = {
    'earnings': '#27ae60',
    'expenses': '#e74c3c',
    'stock': '#3498db',
    'bond': '#9b59b6',
    'cash': '#f1c40f',
    'fw': '#2ecc71',
    'hc': '#e67e22',
    'subsistence': '#95a5a6',
    'variable': '#e74c3c',
    'consumption': '#e74c3c',
    'nw': '#9b59b6',
    'rate': '#f39c12',
    'optimal': '#2ecc71',
    'rot': '#3498db',
}


def create_base_case_page(
    result: 'LifecycleResult',
    params: 'LifecycleParams',
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (20, 24),
    use_years: bool = True
) -> plt.Figure:
    """
    Create Page 1: BASE CASE (Deterministic Median Path).

    Layout with 4 sections, 10 charts total:
    - Section 1: Assumptions (2 charts: Earnings, Expenses)
    - Section 2: Forward-Looking Values (2 charts: Present Values, Durations)
    - Section 3: Wealth (4 charts: HC vs FW, HC Decomposition, Expense Decomposition, Net HC minus Expenses)
    - Section 4: Choices (2 charts: Consumption Path, Portfolio Allocation)

    Args:
        result: LifecycleResult from compute_lifecycle_median_path
        params: LifecycleParams
        econ_params: EconomicParams (optional)
        figsize: Figure size tuple
        use_years: If True, x-axis shows years from career start

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.25)

    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = result.ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    COLORS = REPORT_COLORS

    # ===== Section 1: Assumptions =====
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, result.earnings, color=COLORS['earnings'], linewidth=2, label='Earnings')
    ax.plot(x, result.expenses, color=COLORS['expenses'], linewidth=2, label='Expenses')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5, label='Retirement')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Income & Expenses ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    savings = result.earnings - result.expenses
    ax.fill_between(x, 0, savings, where=savings >= 0, alpha=0.7, color=COLORS['earnings'], label='Savings')
    ax.fill_between(x, 0, savings, where=savings < 0, alpha=0.7, color=COLORS['expenses'], label='Drawdown')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Cash Flow: Earnings - Expenses ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # ===== Section 2: Forward-Looking Values =====
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x, result.pv_earnings, color=COLORS['earnings'], linewidth=2, label='PV Earnings')
    ax.plot(x, result.pv_expenses, color=COLORS['expenses'], linewidth=2, label='PV Expenses')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Present Values ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x, result.duration_earnings, color=COLORS['earnings'], linewidth=2, label='Duration (Earnings)')
    ax.plot(x, result.duration_expenses, color=COLORS['expenses'], linewidth=2, label='Duration (Expenses)')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Years')
    ax.set_title('Durations (years)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # ===== Section 3: Wealth =====
    ax = fig.add_subplot(gs[2, 0])
    ax.fill_between(x, 0, result.financial_wealth, alpha=0.7, color=COLORS['fw'], label='Financial Wealth')
    ax.fill_between(x, result.financial_wealth, result.financial_wealth + result.human_capital,
                   alpha=0.7, color=COLORS['hc'], label='Human Capital')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Human Capital vs Financial Wealth ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(x, result.hc_cash_component, color=COLORS['cash'], linewidth=2, label='HC Cash')
    ax.plot(x, result.hc_bond_component, color=COLORS['bond'], linewidth=2, label='HC Bond')
    ax.plot(x, result.hc_stock_component, color=COLORS['stock'], linewidth=2, label='HC Stock')
    ax.plot(x, result.human_capital, color='black', linewidth=1.5, linestyle='--', label='Total HC')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Human Capital Decomposition ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    ax = fig.add_subplot(gs[3, 0])
    ax.plot(x, result.exp_cash_component, color=COLORS['cash'], linewidth=2, label='Expense Cash')
    ax.plot(x, result.exp_bond_component, color=COLORS['bond'], linewidth=2, label='Expense Bond')
    ax.plot(x, result.pv_expenses, color='black', linewidth=1.5, linestyle='--', label='Total Expenses')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Expense Liability Decomposition ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    ax = fig.add_subplot(gs[3, 1])
    net_stock = result.hc_stock_component
    net_bond = result.hc_bond_component - result.exp_bond_component
    net_cash = result.hc_cash_component - result.exp_cash_component
    net_total = net_stock + net_bond + net_cash

    ax.plot(x, net_cash, color=COLORS['cash'], linewidth=2, label='Net Cash')
    ax.plot(x, net_bond, color=COLORS['bond'], linewidth=2, label='Net Bond')
    ax.plot(x, net_stock, color=COLORS['stock'], linewidth=2, label='Net Stock')
    ax.plot(x, net_total, color='black', linewidth=1.5, linestyle='--', label='Net Total')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Net HC minus Expenses ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # ===== Section 4: Choices =====
    ax = fig.add_subplot(gs[4, 0])
    ax.fill_between(x, 0, result.subsistence_consumption, alpha=0.7, color=COLORS['subsistence'], label='Subsistence')
    ax.fill_between(x, result.subsistence_consumption, result.total_consumption,
                   alpha=0.7, color=COLORS['variable'], label='Variable')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Consumption Path ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    ax = fig.add_subplot(gs[4, 1])
    stock_pct = result.stock_weight_no_short * 100
    bond_pct = result.bond_weight_no_short * 100
    cash_pct = result.cash_weight_no_short * 100

    ax.fill_between(x, 0, cash_pct, alpha=0.7, color=COLORS['cash'], label='Cash')
    ax.fill_between(x, cash_pct, cash_pct + bond_pct, alpha=0.7, color=COLORS['bond'], label='Bonds')
    ax.fill_between(x, cash_pct + bond_pct, cash_pct + bond_pct + stock_pct,
                   alpha=0.7, color=COLORS['stock'], label='Stocks')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Allocation (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Portfolio Allocation (%)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    fig.suptitle('PAGE 1: BASE CASE (Deterministic Median Path)', fontsize=16, fontweight='bold', y=0.995)
    return fig


def create_monte_carlo_page(
    mc_result: 'MonteCarloResult',
    params: 'LifecycleParams',
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (20, 22),
    use_years: bool = True,
    percentiles: List[int] = None,
) -> plt.Figure:
    """
    Create Page 2: MONTE CARLO simulation results.

    Layout with 6 chart panels:
    - Consumption Distribution (percentile lines)
    - Financial Wealth Distribution (percentile lines)
    - Net Worth Distribution (percentile lines)
    - Terminal Values Grid (text summary)
    - Cumulative Stock Returns (percentile bands)
    - Interest Rate Paths (percentile bands)

    Args:
        mc_result: MonteCarloResult from run_lifecycle_monte_carlo
        params: LifecycleParams
        econ_params: EconomicParams (optional)
        figsize: Figure size tuple
        use_years: If True, x-axis shows years from career start
        percentiles: List of percentiles to show (default: [5, 25, 50, 75, 95])

    Returns:
        matplotlib Figure
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    n_sims = mc_result.financial_wealth_paths.shape[0]
    total_years = len(mc_result.ages)

    if use_years:
        x = np.arange(total_years)
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = mc_result.ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    COLORS = REPORT_COLORS

    # Compute percentiles
    consumption_pctls = np.percentile(mc_result.total_consumption_paths, percentiles, axis=0)
    fw_pctls = np.percentile(mc_result.financial_wealth_paths, percentiles, axis=0)

    net_worth_paths = (mc_result.human_capital_paths + mc_result.financial_wealth_paths -
                       mc_result.median_result.pv_expenses[np.newaxis, :])
    nw_pctls = np.percentile(net_worth_paths, percentiles, axis=0)

    stock_return_data = mc_result.stock_return_paths[:, :total_years]
    stock_cumulative = np.cumprod(1 + stock_return_data, axis=1)
    stock_pctls = np.percentile(stock_cumulative, percentiles, axis=0)

    rate_data = mc_result.interest_rate_paths[:, :total_years]
    rate_pctls = np.percentile(rate_data * 100, percentiles, axis=0)

    line_styles = {0: ':', 1: '--', 2: '-', 3: '--', 4: ':'}
    line_widths = {0: 1, 1: 1, 2: 2.5, 3: 1, 4: 1}

    # ===== Consumption Distribution =====
    ax = fig.add_subplot(gs[0, 0])
    for i, p in enumerate(percentiles):
        ax.plot(x, consumption_pctls[i], color=COLORS['consumption'],
               linestyle=line_styles[i], linewidth=line_widths[i],
               label=f'{p}th %ile' if p != 50 else 'Median')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Consumption Distribution ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    apply_wealth_log_scale(ax)

    # ===== Financial Wealth Distribution =====
    ax = fig.add_subplot(gs[0, 1])
    for i, p in enumerate(percentiles):
        ax.plot(x, fw_pctls[i], color=COLORS['fw'],
               linestyle=line_styles[i], linewidth=line_widths[i],
               label=f'{p}th %ile' if p != 50 else 'Median')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth Distribution ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    apply_wealth_log_scale(ax)

    # ===== Net Worth Distribution =====
    ax = fig.add_subplot(gs[1, 0])
    for i, p in enumerate(percentiles):
        ax.plot(x, nw_pctls[i], color=COLORS['nw'],
               linestyle=line_styles[i], linewidth=line_widths[i],
               label=f'{p}th %ile' if p != 50 else 'Median')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Net Worth Distribution (HC + FW - Expenses) ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    apply_wealth_log_scale(ax)

    # ===== Terminal Values Grid =====
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')

    terminal_fw = fw_pctls[:, -1]
    terminal_consumption = consumption_pctls[:, -1]
    depleted_count = np.sum(mc_result.financial_wealth_paths[:, -1] < 10)

    terminal_text = f"""
Terminal Values at Age {params.end_age - 1}
{'='*50}

Financial Wealth ($k):
  5th percentile:  ${terminal_fw[0]:>10,.0f}
  25th percentile: ${terminal_fw[1]:>10,.0f}
  Median:          ${terminal_fw[2]:>10,.0f}
  75th percentile: ${terminal_fw[3]:>10,.0f}
  95th percentile: ${terminal_fw[4]:>10,.0f}

Annual Consumption ($k):
  5th percentile:  ${terminal_consumption[0]:>10,.0f}
  25th percentile: ${terminal_consumption[1]:>10,.0f}
  Median:          ${terminal_consumption[2]:>10,.0f}
  75th percentile: ${terminal_consumption[3]:>10,.0f}
  95th percentile: ${terminal_consumption[4]:>10,.0f}

Runs depleted (FW < $10k): {depleted_count} of {n_sims}
Default Rate: {np.mean(mc_result.default_flags)*100:.1f}%
"""
    ax.text(0.1, 0.9, terminal_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    ax.set_title('Terminal Values Grid', fontweight='bold')

    # ===== Cumulative Stock Returns =====
    ax = fig.add_subplot(gs[2, 0])
    log_stock_pctls = np.log(stock_pctls)

    ax.fill_between(x, log_stock_pctls[0], log_stock_pctls[1], alpha=0.15, color=COLORS['stock'])
    ax.fill_between(x, log_stock_pctls[1], log_stock_pctls[3], alpha=0.3, color=COLORS['stock'])
    ax.fill_between(x, log_stock_pctls[3], log_stock_pctls[4], alpha=0.15, color=COLORS['stock'])

    ax.plot(x, log_stock_pctls[2], color=COLORS['stock'], linewidth=2, label='Median')
    ax.plot(x, log_stock_pctls[1], color=COLORS['stock'], linewidth=1, linestyle='--', label='25th/75th')
    ax.plot(x, log_stock_pctls[3], color=COLORS['stock'], linewidth=1, linestyle='--')

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Log Cumulative Return')
    ax.set_title('Cumulative Stock Returns (Log Scale)', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)

    def log_to_pct(y, pos):
        return f'{np.exp(y)*100:.0f}%'
    ax.yaxis.set_major_formatter(plt.FuncFormatter(log_to_pct))

    # ===== Interest Rate Paths =====
    ax = fig.add_subplot(gs[2, 1])
    ax.fill_between(x, rate_pctls[0], rate_pctls[1], alpha=0.15, color=COLORS['rate'])
    ax.fill_between(x, rate_pctls[1], rate_pctls[3], alpha=0.3, color=COLORS['rate'])
    ax.fill_between(x, rate_pctls[3], rate_pctls[4], alpha=0.15, color=COLORS['rate'])

    ax.plot(x, rate_pctls[2], color=COLORS['rate'], linewidth=2, label='Median')
    ax.plot(x, rate_pctls[1], color=COLORS['rate'], linewidth=1, linestyle='--', label='25th/75th')
    ax.plot(x, rate_pctls[3], color=COLORS['rate'], linewidth=1, linestyle='--')

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Interest Rate (%)')
    ax.set_title('Interest Rate Paths (%)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(f'PAGE 2: MONTE CARLO ({n_sims} Runs)', fontsize=16, fontweight='bold', y=0.995)
    return fig


def create_scenario_page(
    scenario_type: str,
    params: 'LifecycleParams',
    econ_params: 'EconomicParams',
    figsize: Tuple[int, int] = (20, 18),
    use_years: bool = True,
    n_simulations: int = 50,
    random_seed: int = 42,
    rate_shock_age: int = None,
    rate_shock_magnitude: float = -0.02,
) -> plt.Figure:
    """
    Create Teaching Scenario Pages (3a, 3b, 3c).

    Supports three scenario types:
    - 'normal': Optimal vs Rule of Thumb (normal market conditions)
    - 'sequenceRisk': Bad early returns stress test (-20% for 5 years at retirement)
    - 'rateShock': Interest rate shock at configurable age

    Each page shows:
    - Default Risk comparison
    - PV Consumption comparison (at time 0)
    - Financial Wealth percentile charts
    - Consumption percentile charts
    - Portfolio Allocation or market condition visualization

    Args:
        scenario_type: One of 'normal', 'sequenceRisk', 'rateShock'
        params: LifecycleParams
        econ_params: EconomicParams
        figsize: Figure size tuple
        use_years: If True, x-axis shows years from career start
        n_simulations: Number of Monte Carlo simulations
        random_seed: Random seed for reproducibility
        rate_shock_age: Age at which rate shock occurs (default: retirement_age)
        rate_shock_magnitude: Magnitude of rate shock (default: -2%)

    Returns:
        matplotlib Figure
    """
    from core import (
        generate_correlated_shocks,
        simulate_interest_rates,
        simulate_stock_returns,
        run_strategy_comparison,
    )

    if rate_shock_age is None:
        rate_shock_age = params.retirement_age

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    total_years = params.end_age - params.start_age
    working_years = params.retirement_age - params.start_age
    ages = np.arange(params.start_age, params.end_age)

    if use_years:
        x = np.arange(total_years)
        xlabel = 'Years from Career Start'
        retirement_x = working_years
    else:
        x = ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    bad_returns_early = scenario_type == 'sequenceRisk'

    rng = np.random.default_rng(random_seed)
    rate_shocks, stock_shocks = generate_correlated_shocks(
        total_years, n_simulations, econ_params.rho, rng
    )

    initial_rate = econ_params.r_bar
    rate_paths = simulate_interest_rates(
        initial_rate, total_years, n_simulations, econ_params, rate_shocks
    )

    if scenario_type == 'rateShock':
        shock_year = rate_shock_age - params.start_age
        if 0 <= shock_year < total_years:
            for sim in range(n_simulations):
                for t in range(shock_year, total_years):
                    rate_paths[sim, t] += rate_shock_magnitude

    stock_return_paths = simulate_stock_returns(rate_paths, econ_params, stock_shocks)

    if bad_returns_early:
        for sim in range(n_simulations):
            for t in range(working_years, min(working_years + 5, total_years)):
                stock_return_paths[sim, t] = -0.20

    comparison = run_strategy_comparison(
        params=params,
        econ_params=econ_params,
        n_simulations=n_simulations,
        random_seed=random_seed,
        bad_returns_early=bad_returns_early,
    )

    COLORS = REPORT_COLORS

    scenario_titles = {
        'normal': 'Normal Market Conditions',
        'sequenceRisk': 'Sequence Risk (Bad Early Returns)',
        'rateShock': f'Interest Rate Shock (at age {rate_shock_age})',
    }

    # ===== Default Risk Bar Chart =====
    ax = fig.add_subplot(gs[0, 0])
    strategies = ['Optimal\n(Variable)', 'Rule of Thumb\n(4% Rule)']
    default_rates = [comparison.optimal_default_rate * 100, comparison.rot_default_rate * 100]
    colors = [COLORS['optimal'], COLORS['rot']]

    bars = ax.bar(strategies, default_rates, color=colors, alpha=0.8, edgecolor='black')
    for bar, rate in zip(bars, default_rates):
        height = bar.get_height()
        ax.annotate(f'{rate:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Default Rate (%)')
    ax.set_title('Default Risk Comparison', fontweight='bold')
    ax.set_ylim(0, max(default_rates) * 1.3 + 5)

    # ===== PV Consumption Comparison =====
    ax = fig.add_subplot(gs[0, 1])
    x_pos = np.arange(len(comparison.percentiles))
    width = 0.35

    ax.bar(x_pos - width/2, comparison.optimal_pv_consumption_percentiles,
          width, label='Optimal', color=COLORS['optimal'], alpha=0.8)
    ax.bar(x_pos + width/2, comparison.rot_pv_consumption_percentiles,
          width, label='Rule of Thumb', color=COLORS['rot'], alpha=0.8)

    ax.set_xlabel('Percentile')
    ax.set_ylabel('PV Consumption ($k)')
    ax.set_title('PV Consumption at Time 0', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{p}th' for p in comparison.percentiles])
    ax.legend(loc='upper left', fontsize=9)

    # ===== Financial Wealth Percentiles =====
    ax = fig.add_subplot(gs[1, 0])
    for i, p in enumerate(comparison.percentiles):
        if p == 50:
            ax.plot(x, comparison.optimal_wealth_percentiles[i], color=COLORS['optimal'],
                   linewidth=2, label='Optimal Median')
            ax.plot(x, comparison.rot_wealth_percentiles[i], color=COLORS['rot'],
                   linewidth=2, linestyle='--', label='RoT Median')
        elif p in [25, 75]:
            ax.plot(x, comparison.optimal_wealth_percentiles[i], color=COLORS['optimal'],
                   linewidth=1, alpha=0.6)
            ax.plot(x, comparison.rot_wealth_percentiles[i], color=COLORS['rot'],
                   linewidth=1, linestyle='--', alpha=0.6)

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth Percentiles', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)

    # ===== Consumption Percentiles =====
    ax = fig.add_subplot(gs[1, 1])
    for i, p in enumerate(comparison.percentiles):
        if p == 50:
            ax.plot(x, comparison.optimal_consumption_percentiles[i], color=COLORS['optimal'],
                   linewidth=2, label='Optimal Median')
            ax.plot(x, comparison.rot_consumption_percentiles[i], color=COLORS['rot'],
                   linewidth=2, linestyle='--', label='RoT Median')
        elif p in [25, 75]:
            ax.plot(x, comparison.optimal_consumption_percentiles[i], color=COLORS['optimal'],
                   linewidth=1, alpha=0.6)
            ax.plot(x, comparison.rot_consumption_percentiles[i], color=COLORS['rot'],
                   linewidth=1, linestyle='--', alpha=0.6)

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Consumption Percentiles', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # ===== Scenario-specific panel =====
    if scenario_type == 'sequenceRisk':
        ax = fig.add_subplot(gs[2, 0])
        stock_data = stock_return_paths[:, :total_years]
        cumulative = np.cumprod(1 + stock_data, axis=1)
        pctls = np.percentile(cumulative, [5, 25, 50, 75, 95], axis=0)

        ax.fill_between(x, pctls[0], pctls[4], alpha=0.2, color='#3498db')
        ax.fill_between(x, pctls[1], pctls[3], alpha=0.3, color='#3498db')
        ax.plot(x, pctls[2], color='#3498db', linewidth=2, label='Median')
        ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)
        ax.axvspan(retirement_x, retirement_x + 5, alpha=0.2, color='red', label='Forced -20% Returns')

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Stock Return Paths (Showing Stress Period)', fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)

    elif scenario_type == 'rateShock':
        ax = fig.add_subplot(gs[2, 0])
        rate_data = rate_paths[:, :total_years] * 100
        pctls = np.percentile(rate_data, [5, 25, 50, 75, 95], axis=0)

        ax.fill_between(x, pctls[0], pctls[4], alpha=0.2, color='#f39c12')
        ax.fill_between(x, pctls[1], pctls[3], alpha=0.3, color='#f39c12')
        ax.plot(x, pctls[2], color='#f39c12', linewidth=2, label='Median')

        shock_x = rate_shock_age - params.start_age if use_years else rate_shock_age
        ax.axvline(x=shock_x, color='red', linestyle='--', linewidth=2, label=f'Rate Shock ({rate_shock_magnitude*100:.0f}%)')
        ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Interest Rate (%)')
        ax.set_title('Interest Rate Paths (Showing Shock)', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)

    else:  # normal
        ax = fig.add_subplot(gs[2, 0])
        ax.plot(x, comparison.rot_stock_weight_sample * 100, color=COLORS['rot'],
               linewidth=2, label='RoT Stocks')
        ax.plot(x, comparison.rot_bond_weight_sample * 100, color='#9b59b6',
               linewidth=2, linestyle='--', label='RoT Bonds')
        ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Allocation (%)')
        ax.set_title('Rule of Thumb Glide Path', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 100)

    # ===== Summary Statistics =====
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')

    summary_text = f"""
Strategy Comparison Summary
{'='*50}

Scenario: {scenario_titles.get(scenario_type, scenario_type)}

Default Rates:
  Optimal (Variable):     {comparison.optimal_default_rate*100:>6.1f}%
  Rule of Thumb (4%):     {comparison.rot_default_rate*100:>6.1f}%

Median Final Wealth ($k):
  Optimal:                ${comparison.optimal_median_final_wealth:>10,.0f}
  Rule of Thumb:          ${comparison.rot_median_final_wealth:>10,.0f}

Median PV Consumption ($k):
  Optimal:                ${np.median(comparison.optimal_pv_consumption):>10,.0f}
  Rule of Thumb:          ${np.median(comparison.rot_pv_consumption):>10,.0f}

Simulations: {n_simulations}
"""
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    page_num = {'normal': '3a', 'sequenceRisk': '3b', 'rateShock': '3c'}.get(scenario_type, '3')
    fig.suptitle(f'PAGE {page_num}: TEACHING SCENARIO - {scenario_titles.get(scenario_type, scenario_type)}',
                fontsize=16, fontweight='bold', y=0.995)

    return fig
