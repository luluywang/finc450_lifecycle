"""
Strategy comparison visualization plots.

This module provides plotting functions for comparing different investment strategies
including optimal vs 4% rule, LDI vs rule-of-thumb, and multi-strategy comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, TYPE_CHECKING

from .styles import COLORS, STRATEGY_COLORS

if TYPE_CHECKING:
    from core import (
        LifecycleParams, EconomicParams, LifecycleResult,
        StrategyComparisonResult, MedianPathComparisonResult, SimulationResult
    )


def create_optimal_vs_4pct_rule_comparison(
    base_params: 'LifecycleParams' = None,
    econ_params: 'EconomicParams' = None,
    withdrawal_rate: float = 0.04,
    figsize: Tuple[int, int] = (16, 14),
    use_years: bool = True
) -> plt.Figure:
    """
    Create a figure comparing optimal variable consumption vs 4% fixed rule.

    Shows:
    - Consumption paths
    - Financial wealth trajectories
    - Default risk under 4% rule
    - Net worth evolution
    """
    from core import LifecycleParams, EconomicParams
    from core import compute_lifecycle_median_path, compute_lifecycle_fixed_consumption

    if base_params is None:
        base_params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    result_optimal = compute_lifecycle_median_path(base_params, econ_params)
    result_4pct = compute_lifecycle_fixed_consumption(base_params, econ_params, withdrawal_rate)

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    if use_years:
        x = np.arange(len(result_optimal.ages))
        xlabel = 'Years from Career Start'
        retirement_x = base_params.retirement_age - base_params.start_age
    else:
        x = result_optimal.ages
        xlabel = 'Age'
        retirement_x = base_params.retirement_age

    color_optimal = '#2ecc71'
    color_4pct = '#e74c3c'

    # Plot 1: Total Consumption Comparison
    ax = axes[0, 0]
    ax.plot(x, result_optimal.total_consumption, color=color_optimal,
            linewidth=2, label='Optimal (Variable)')
    ax.plot(x, result_4pct.total_consumption, color=color_4pct,
            linewidth=2, linestyle='--', label=f'{withdrawal_rate*100:.0f}% Rule (Fixed)')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5, label='Retirement')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Consumption: Optimal vs 4% Rule')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 2: Financial Wealth Comparison
    ax = axes[0, 1]
    ax.plot(x, result_optimal.financial_wealth, color=color_optimal,
            linewidth=2, label='Optimal (Variable)')
    ax.plot(x, result_4pct.financial_wealth, color=color_4pct,
            linewidth=2, linestyle='--', label=f'{withdrawal_rate*100:.0f}% Rule (Fixed)')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth: Optimal vs 4% Rule')
    ax.legend(loc='upper left', fontsize=9)

    # Plot 3: Cumulative Consumption
    ax = axes[1, 0]
    cumulative_optimal = np.cumsum(result_optimal.total_consumption)
    cumulative_4pct = np.cumsum(result_4pct.total_consumption)
    ax.plot(x, cumulative_optimal, color=color_optimal,
            linewidth=2, label='Optimal (Variable)')
    ax.plot(x, cumulative_4pct, color=color_4pct,
            linewidth=2, linestyle='--', label=f'{withdrawal_rate*100:.0f}% Rule (Fixed)')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Cumulative Lifetime Consumption')
    ax.legend(loc='upper left', fontsize=9)

    total_optimal = cumulative_optimal[-1]
    total_4pct = cumulative_4pct[-1]
    ax.annotate(f'Total: ${total_optimal:,.0f}k', xy=(0.98, 0.85),
                xycoords='axes fraction', fontsize=10, ha='right', color=color_optimal)
    ax.annotate(f'Total: ${total_4pct:,.0f}k', xy=(0.98, 0.75),
                xycoords='axes fraction', fontsize=10, ha='right', color=color_4pct)

    # Plot 4: Net Worth Comparison
    ax = axes[1, 1]
    ax.plot(x, result_optimal.net_worth, color=color_optimal,
            linewidth=2, label='Optimal (Variable)')
    ax.plot(x, result_4pct.net_worth, color=color_4pct,
            linewidth=2, linestyle='--', label=f'{withdrawal_rate*100:.0f}% Rule (Fixed)')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Net Worth (HC + FW - PV Expenses)')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 5: Consumption Breakdown - Optimal
    ax = axes[2, 0]
    ax.fill_between(x, 0, result_optimal.subsistence_consumption,
                    alpha=0.7, color='#95a5a6', label='Subsistence')
    ax.fill_between(x, result_optimal.subsistence_consumption,
                    result_optimal.total_consumption,
                    alpha=0.7, color='#f39c12', label='Variable')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Optimal Strategy: Consumption Breakdown')
    ax.legend(loc='upper right', fontsize=9)

    # Plot 6: Consumption Breakdown - 4% Rule
    ax = axes[2, 1]
    ax.fill_between(x, 0, result_4pct.subsistence_consumption,
                    alpha=0.7, color='#95a5a6', label='Subsistence')
    ax.fill_between(x, result_4pct.subsistence_consumption,
                    result_4pct.total_consumption,
                    alpha=0.7, color='#e74c3c', label='Fixed Withdrawal')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title(f'{withdrawal_rate*100:.0f}% Rule: Consumption Breakdown')
    ax.legend(loc='upper right', fontsize=9)

    # Check for default in 4% rule
    default_idx = np.where(result_4pct.financial_wealth[base_params.retirement_age - base_params.start_age:] <= 0)[0]
    if len(default_idx) > 0:
        default_year = default_idx[0] + (base_params.retirement_age - base_params.start_age)
        if use_years:
            ax.axvline(x=default_year, color='darkred', linestyle='--', linewidth=2, label='Default')
        ax.annotate('DEFAULT', xy=(default_year if use_years else base_params.start_age + default_year, 0),
                   fontsize=12, color='darkred', fontweight='bold')

    plt.tight_layout()
    return fig


def create_strategy_comparison_figure(
    comparison_result: 'StrategyComparisonResult',
    params: 'LifecycleParams' = None,
    figsize: Tuple[int, int] = (18, 12),
    use_years: bool = True,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Create a figure comparing optimal vs rule-of-thumb strategies.

    Shows a 2x3 panel layout:
    - (0,0): Default risk bar chart
    - (0,1): Consumption percentile fan charts (both strategies)
    - (0,2): Wealth percentile fan charts (both strategies)
    - (1,0): Rule-of-thumb allocation glide path
    - (1,1): Summary statistics table
    - (1,2): Default age distribution histograms
    """
    from core import LifecycleParams

    if params is None:
        params = LifecycleParams()

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    if use_years:
        x = np.arange(len(comparison_result.ages))
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = comparison_result.ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    color_optimal = '#2ecc71'
    color_rot = '#3498db'
    alpha_fan = 0.3

    # (0,0): Default Risk Bar Chart
    ax = axes[0, 0]
    strategies = ['Optimal\n(Variable Consumption)', 'Rule of Thumb\n(100-Age, 4% Rule)']
    default_rates = [
        comparison_result.optimal_default_rate * 100,
        comparison_result.rot_default_rate * 100
    ]
    colors = [color_optimal, color_rot]
    bars = ax.bar(strategies, default_rates, color=colors, alpha=0.8, edgecolor='black')

    for bar, rate in zip(bars, default_rates):
        height = bar.get_height()
        ax.annotate(f'{rate:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Default Rate (%)')
    ax.set_title('Default Risk Comparison')
    ax.set_ylim(0, max(100, max(default_rates) * 1.2))

    # (0,1): Consumption Percentile Fan Charts
    ax = axes[0, 1]
    p_idx = {p: i for i, p in enumerate(comparison_result.percentiles)}

    if 5 in p_idx and 95 in p_idx:
        ax.fill_between(x,
                        comparison_result.optimal_consumption_percentiles[p_idx[5], :],
                        comparison_result.optimal_consumption_percentiles[p_idx[95], :],
                        alpha=alpha_fan, color=color_optimal, label='Optimal 5-95%')
    if 25 in p_idx and 75 in p_idx:
        ax.fill_between(x,
                        comparison_result.optimal_consumption_percentiles[p_idx[25], :],
                        comparison_result.optimal_consumption_percentiles[p_idx[75], :],
                        alpha=alpha_fan + 0.2, color=color_optimal)
    if 50 in p_idx:
        ax.plot(x, comparison_result.optimal_consumption_percentiles[p_idx[50], :],
                color=color_optimal, linewidth=2, label='Optimal Median')

    if 5 in p_idx and 95 in p_idx:
        ax.fill_between(x,
                        comparison_result.rot_consumption_percentiles[p_idx[5], :],
                        comparison_result.rot_consumption_percentiles[p_idx[95], :],
                        alpha=alpha_fan, color=color_rot, label='RoT 5-95%')
    if 25 in p_idx and 75 in p_idx:
        ax.fill_between(x,
                        comparison_result.rot_consumption_percentiles[p_idx[25], :],
                        comparison_result.rot_consumption_percentiles[p_idx[75], :],
                        alpha=alpha_fan + 0.2, color=color_rot)
    if 50 in p_idx:
        ax.plot(x, comparison_result.rot_consumption_percentiles[p_idx[50], :],
                color=color_rot, linewidth=2, linestyle='--', label='RoT Median')

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Consumption Comparison (Percentile Bands)')
    ax.legend(loc='upper right', fontsize=8)

    # (0,2): Wealth Percentile Fan Charts
    ax = axes[0, 2]

    if 5 in p_idx and 95 in p_idx:
        ax.fill_between(x,
                        comparison_result.optimal_wealth_percentiles[p_idx[5], :],
                        comparison_result.optimal_wealth_percentiles[p_idx[95], :],
                        alpha=alpha_fan, color=color_optimal, label='Optimal 5-95%')
    if 25 in p_idx and 75 in p_idx:
        ax.fill_between(x,
                        comparison_result.optimal_wealth_percentiles[p_idx[25], :],
                        comparison_result.optimal_wealth_percentiles[p_idx[75], :],
                        alpha=alpha_fan + 0.2, color=color_optimal)
    if 50 in p_idx:
        ax.plot(x, comparison_result.optimal_wealth_percentiles[p_idx[50], :],
                color=color_optimal, linewidth=2, label='Optimal Median')

    if 5 in p_idx and 95 in p_idx:
        ax.fill_between(x,
                        comparison_result.rot_wealth_percentiles[p_idx[5], :],
                        comparison_result.rot_wealth_percentiles[p_idx[95], :],
                        alpha=alpha_fan, color=color_rot, label='RoT 5-95%')
    if 25 in p_idx and 75 in p_idx:
        ax.fill_between(x,
                        comparison_result.rot_wealth_percentiles[p_idx[25], :],
                        comparison_result.rot_wealth_percentiles[p_idx[75], :],
                        alpha=alpha_fan + 0.2, color=color_rot)
    if 50 in p_idx:
        ax.plot(x, comparison_result.rot_wealth_percentiles[p_idx[50], :],
                color=color_rot, linewidth=2, linestyle='--', label='RoT Median')

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth Comparison (Percentile Bands)')
    ax.legend(loc='upper left', fontsize=8)

    # (1,0): Rule-of-Thumb Allocation Glide Path
    ax = axes[1, 0]
    ax.stackplot(x,
                 comparison_result.rot_stock_weight_sample * 100,
                 comparison_result.rot_bond_weight_sample * 100,
                 comparison_result.rot_cash_weight_sample * 100,
                 labels=['Stocks', 'Bonds', 'Cash'],
                 colors=['#e74c3c', '#3498db', '#95a5a6'],
                 alpha=0.8)
    ax.axvline(x=retirement_x, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Allocation (%)')
    ax.set_title('Rule-of-Thumb: (100-Age)% Stock Glide Path')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 100)

    # (1,1): Summary Statistics Table
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
Strategy Comparison Summary
{'='*40}
Number of Simulations: {comparison_result.n_simulations}

                    Optimal    Rule-of-Thumb
                    -------    -------------
Default Rate:       {comparison_result.optimal_default_rate*100:6.1f}%    {comparison_result.rot_default_rate*100:6.1f}%
Median Final Wealth: ${comparison_result.optimal_median_final_wealth:,.0f}k   ${comparison_result.rot_median_final_wealth:,.0f}k

Rule-of-Thumb Strategy:
  - Savings Rate: 15% of income
  - Stock Allocation: (100 - age)%
  - FI Duration: 6 years (30% bonds, 70% cash)
  - Retirement: 4% fixed withdrawal

LDI Strategy:
  - Variable consumption based on net worth
  - MV-optimal allocation
  - Adapts to market conditions
"""

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # (1,2): Default Age Distribution
    ax = axes[1, 2]

    opt_default_ages = comparison_result.optimal_default_ages[~np.isnan(comparison_result.optimal_default_ages)]
    rot_default_ages = comparison_result.rot_default_ages[~np.isnan(comparison_result.rot_default_ages)]

    if len(opt_default_ages) > 0 or len(rot_default_ages) > 0:
        bins = np.arange(params.retirement_age, params.end_age + 1, 2)

        if len(opt_default_ages) > 0:
            ax.hist(opt_default_ages, bins=bins, alpha=0.6, color=color_optimal,
                    label=f'Optimal (n={len(opt_default_ages)})', edgecolor='black')
        if len(rot_default_ages) > 0:
            ax.hist(rot_default_ages, bins=bins, alpha=0.6, color=color_rot,
                    label=f'RoT (n={len(rot_default_ages)})', edgecolor='black')

        ax.set_xlabel('Age at Default')
        ax.set_ylabel('Count')
        ax.set_title('Default Age Distribution')
        ax.legend(loc='upper right', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No defaults in\neither strategy',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Default Age Distribution')

    plt.suptitle(f'Optimal vs Rule-of-Thumb Strategy Comparison{title_suffix}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def create_median_path_comparison_figure(
    comparison_result: 'MedianPathComparisonResult',
    params: 'LifecycleParams' = None,
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (18, 14),
    use_years: bool = True,
) -> plt.Figure:
    """
    Create a figure comparing LDI vs Rule-of-Thumb on deterministic median paths.
    """
    from core import LifecycleParams, EconomicParams

    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    if use_years:
        x = np.arange(len(comparison_result.ages))
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = comparison_result.ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    color_ldi = '#2ecc71'
    color_rot = '#3498db'
    color_earnings = '#f39c12'

    # (0,0): Financial Wealth Comparison
    ax = axes[0, 0]
    ax.plot(x, comparison_result.ldi_financial_wealth, color=color_ldi,
            linewidth=2.5, label='LDI Strategy')
    ax.plot(x, comparison_result.rot_financial_wealth, color=color_rot,
            linewidth=2.5, linestyle='--', label='Rule-of-Thumb')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5, label='Retirement')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth (Deterministic Median)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,1): Consumption Comparison
    ax = axes[0, 1]
    ax.plot(x, comparison_result.ldi_total_consumption, color=color_ldi,
            linewidth=2.5, label='LDI Strategy')
    ax.plot(x, comparison_result.rot_total_consumption, color=color_rot,
            linewidth=2.5, linestyle='--', label='Rule-of-Thumb')
    ax.plot(x, comparison_result.earnings, color=color_earnings,
            linewidth=1.5, linestyle=':', alpha=0.7, label='Earnings')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Consumption (Deterministic Median)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,2): Summary Statistics
    ax = axes[0, 2]
    ax.axis('off')

    long_bond_dur = econ_params.bond_duration
    bond_weight_in_fi = min(1.0, comparison_result.rot_target_duration / long_bond_dur) if long_bond_dur > 0 else 0.0

    summary_text = f"""
Median Path Comparison (Expected Returns)
{'='*45}

LDI Strategy:
  - Variable consumption based on net worth
  - MV-optimal allocation (constant targets)
  - Adapts consumption to wealth changes

Rule-of-Thumb Strategy:
  - Savings Rate: {comparison_result.rot_savings_rate*100:.0f}% of income
  - Stock Allocation: (100 - age)%
  - FI Target Duration: {comparison_result.rot_target_duration:.0f} years
  - FI Split: {bond_weight_in_fi*100:.0f}% bonds / {(1-bond_weight_in_fi)*100:.0f}% cash
    (bonds={long_bond_dur:.0f}yr duration)
  - Retirement: {comparison_result.rot_withdrawal_rate*100:.0f}% fixed withdrawal

PV Lifetime Consumption (@ r={econ_params.r_bar*100:.1f}%):
  - LDI Strategy:     ${comparison_result.ldi_pv_consumption:,.0f}k
  - Rule-of-Thumb:    ${comparison_result.rot_pv_consumption:,.0f}k
  - Difference:       ${comparison_result.ldi_pv_consumption - comparison_result.rot_pv_consumption:+,.0f}k
                      ({(comparison_result.ldi_pv_consumption/comparison_result.rot_pv_consumption - 1)*100:+.1f}%)
"""

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # (1,0): LDI Allocation Glide Path
    ax = axes[1, 0]
    ax.stackplot(x,
                 comparison_result.ldi_stock_weight * 100,
                 comparison_result.ldi_bond_weight * 100,
                 comparison_result.ldi_cash_weight * 100,
                 labels=['Stocks', 'Bonds', 'Cash'],
                 colors=['#e74c3c', '#3498db', '#95a5a6'],
                 alpha=0.8)
    ax.axvline(x=retirement_x, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Allocation (%)')
    ax.set_title('LDI: Financial Portfolio Allocation')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 100)

    # (1,1): RoT Allocation Glide Path
    ax = axes[1, 1]
    ax.stackplot(x,
                 comparison_result.rot_stock_weight * 100,
                 comparison_result.rot_bond_weight * 100,
                 comparison_result.rot_cash_weight * 100,
                 labels=['Stocks', 'Bonds', 'Cash'],
                 colors=['#e74c3c', '#3498db', '#95a5a6'],
                 alpha=0.8)
    ax.axvline(x=retirement_x, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Allocation (%)')
    ax.set_title(f'Rule-of-Thumb: (100-Age)% Stock, {comparison_result.rot_target_duration:.0f}yr FI Duration')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 100)

    # (1,2): Fixed Income Breakdown Comparison
    ax = axes[1, 2]

    ax.plot(x, comparison_result.ldi_bond_weight * 100, color=color_ldi,
            linewidth=2, label='LDI Bonds')
    ax.plot(x, comparison_result.rot_bond_weight * 100, color=color_rot,
            linewidth=2, linestyle='--', label='RoT Bonds')

    ax.plot(x, comparison_result.ldi_cash_weight * 100, color=color_ldi,
            linewidth=2, linestyle=':', alpha=0.7, label='LDI Cash')
    ax.plot(x, comparison_result.rot_cash_weight * 100, color=color_rot,
            linewidth=2, linestyle=':', alpha=0.7, label='RoT Cash')

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Allocation (%)')
    ax.set_title('Bond & Cash Allocations')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('LDI vs Rule-of-Thumb: Deterministic Median Path Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_duration_matching_intuition(
    rates: np.ndarray,
    wealth_paths_mm: np.ndarray,
    wealth_paths_dm: np.ndarray,
    consumption_target: float,
    horizon: int,
    sample_idx: int = 0,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Show how duration matching protects funded status against interest rate changes.
    """
    from core import liability_pv_vectorized

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    years = np.arange(horizon + 1)
    sample_rates = rates[sample_idx, :]

    liab_pv = np.array([
        liability_pv_vectorized(consumption_target, sample_rates[t:t+1], horizon - t)[0]
        for t in range(horizon + 1)
    ])

    # Panel 1: Interest rate path
    ax = axes[0, 0]
    ax.plot(years, sample_rates * 100, 'b-', linewidth=2)
    ax.axhline(y=3.0, color='gray', linestyle='--', alpha=0.7, label='Long-run mean')
    ax.set_xlabel('Year')
    ax.set_ylabel('Interest Rate (%)')
    ax.set_title('Interest Rate Path')
    ax.legend()
    ax.set_xlim(0, horizon)

    # Panel 2: Assets vs Liabilities for MM strategy
    ax = axes[0, 1]
    ax.plot(years, wealth_paths_mm[sample_idx, :] / 1e6, 'b-',
            linewidth=2, label='Assets (MM)')
    ax.plot(years, liab_pv / 1e6, 'r--', linewidth=2, label='PV Liabilities')
    ax.set_xlabel('Year')
    ax.set_ylabel('Value ($M)')
    ax.set_title('Money Market Strategy: Assets vs Liabilities')
    ax.legend()
    ax.set_xlim(0, horizon)

    # Panel 3: Assets vs Liabilities for DM strategy
    ax = axes[1, 0]
    ax.plot(years, wealth_paths_dm[sample_idx, :] / 1e6, 'g-',
            linewidth=2, label='Assets (Duration Matched)')
    ax.plot(years, liab_pv / 1e6, 'r--', linewidth=2, label='PV Liabilities')
    ax.set_xlabel('Year')
    ax.set_ylabel('Value ($M)')
    ax.set_title('Duration Matching Strategy: Assets vs Liabilities')
    ax.legend()
    ax.set_xlim(0, horizon)

    # Panel 4: Funded ratios comparison
    ax = axes[1, 1]
    funded_mm = np.where(liab_pv > 0, wealth_paths_mm[sample_idx, :] / liab_pv, 1.0)
    funded_dm = np.where(liab_pv > 0, wealth_paths_dm[sample_idx, :] / liab_pv, 1.0)

    ax.plot(years, funded_mm, 'b-', linewidth=2, label='MM Strategy')
    ax.plot(years, funded_dm, 'g-', linewidth=2, label='Duration Matched')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Fully Funded')
    ax.set_xlabel('Year')
    ax.set_ylabel('Funded Ratio')
    ax.set_title('Funded Status Comparison')
    ax.legend()
    ax.set_xlim(0, horizon)
    ax.set_ylim(0, max(2.5, max(funded_mm.max(), funded_dm.max()) * 1.1))

    plt.tight_layout()
    return fig


def plot_strategy_comparison_bars(
    results: Dict[str, 'SimulationResult'],
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Bar chart comparing key metrics across strategies.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    strategies = list(results.keys())
    x = np.arange(len(strategies))
    width = 0.6

    # Panel 1: Default Rates
    ax = axes[0]
    default_rates = [100 * results[s].defaulted.mean() for s in strategies]
    bars = ax.bar(x, default_rates, width, color=STRATEGY_COLORS[:len(strategies)])
    ax.set_ylabel('Default Rate (%)')
    ax.set_title('Default Probability')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(' + ', '\n') for s in strategies], fontsize=9)
    ax.set_ylim(0, max(default_rates) * 1.2 if max(default_rates) > 0 else 10)

    for bar, val in zip(bars, default_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Panel 2: Average Total Consumption
    ax = axes[1]
    avg_consumption = [results[s].total_consumption.mean() / 1e6 for s in strategies]
    bars = ax.bar(x, avg_consumption, width, color=STRATEGY_COLORS[:len(strategies)])
    ax.set_ylabel('Avg Total Consumption ($M)')
    ax.set_title('Lifetime Consumption')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(' + ', '\n') for s in strategies], fontsize=9)

    for bar, val in zip(bars, avg_consumption):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'${val:.2f}M', ha='center', va='bottom', fontsize=10)

    # Panel 3: Consumption Volatility
    ax = axes[2]
    cons_vol = [results[s].consumption_paths.std(axis=1).mean() / 1000 for s in strategies]
    bars = ax.bar(x, cons_vol, width, color=STRATEGY_COLORS[:len(strategies)])
    ax.set_ylabel('Consumption Volatility ($k/year)')
    ax.set_title('Consumption Uncertainty')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(' + ', '\n') for s in strategies], fontsize=9)

    for bar, val in zip(bars, cons_vol):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'${val:.1f}k', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig
