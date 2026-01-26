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
        SimulationResult, StrategyComparison,
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

    color_optimal = '#1A759F'  # Deep blue (colorblind-safe)
    color_4pct = '#E9C46A'      # Amber (colorblind-safe)

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
                    alpha=0.7, color='#2A9D8F', label='Variable')
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
                    alpha=0.7, color='#E9C46A', label='Fixed Withdrawal')
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
    comparison_result: 'StrategyComparison',
    params: 'LifecycleParams' = None,
    figsize: Tuple[int, int] = (18, 12),
    use_years: bool = True,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Create a figure comparing LDI vs Rule-of-Thumb strategies.

    Shows a 2x3 panel layout:
    - (0,0): Default risk bar chart
    - (0,1): Consumption percentile fan charts (both strategies)
    - (0,2): Wealth percentile fan charts (both strategies)
    - (1,0): Rule-of-thumb allocation glide path
    - (1,1): Summary statistics table
    - (1,2): Default age distribution histograms

    Args:
        comparison_result: StrategyComparison with result_a=LDI, result_b=RoT
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

    color_optimal = '#1A759F'  # Deep blue (colorblind-safe)
    color_rot = '#E9C46A'       # Amber (colorblind-safe)
    alpha_fan = 0.3

    # Compute percentiles on demand
    percentiles = [5, 25, 50, 75, 95]
    p_idx = {p: i for i, p in enumerate(percentiles)}

    ldi_wealth_pct = comparison_result.wealth_percentiles('a', percentiles)
    rot_wealth_pct = comparison_result.wealth_percentiles('b', percentiles)
    ldi_cons_pct = comparison_result.consumption_percentiles('a', percentiles)
    rot_cons_pct = comparison_result.consumption_percentiles('b', percentiles)

    # Get default rates
    ldi_default_rate = comparison_result.default_rate('a')
    rot_default_rate = comparison_result.default_rate('b')

    # (0,0): Default Risk Bar Chart
    ax = axes[0, 0]
    strategies = ['LDI\n(Variable Consumption)', 'Rule of Thumb\n(100-Age, 4% Rule)']
    default_rates = [ldi_default_rate * 100, rot_default_rate * 100]
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

    ax.fill_between(x, ldi_cons_pct[p_idx[5], :], ldi_cons_pct[p_idx[95], :],
                    alpha=alpha_fan, color=color_optimal, label='LDI 5-95%')
    ax.fill_between(x, ldi_cons_pct[p_idx[25], :], ldi_cons_pct[p_idx[75], :],
                    alpha=alpha_fan + 0.2, color=color_optimal)
    ax.plot(x, ldi_cons_pct[p_idx[50], :],
            color=color_optimal, linewidth=2, label='LDI Median')

    ax.fill_between(x, rot_cons_pct[p_idx[5], :], rot_cons_pct[p_idx[95], :],
                    alpha=alpha_fan, color=color_rot, label='RoT 5-95%')
    ax.fill_between(x, rot_cons_pct[p_idx[25], :], rot_cons_pct[p_idx[75], :],
                    alpha=alpha_fan + 0.2, color=color_rot)
    ax.plot(x, rot_cons_pct[p_idx[50], :],
            color=color_rot, linewidth=2, linestyle='--', label='RoT Median')

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Consumption Comparison (Percentile Bands)')
    ax.legend(loc='upper right', fontsize=8)

    # (0,2): Wealth Percentile Fan Charts
    ax = axes[0, 2]

    ax.fill_between(x, ldi_wealth_pct[p_idx[5], :], ldi_wealth_pct[p_idx[95], :],
                    alpha=alpha_fan, color=color_optimal, label='LDI 5-95%')
    ax.fill_between(x, ldi_wealth_pct[p_idx[25], :], ldi_wealth_pct[p_idx[75], :],
                    alpha=alpha_fan + 0.2, color=color_optimal)
    ax.plot(x, ldi_wealth_pct[p_idx[50], :],
            color=color_optimal, linewidth=2, label='LDI Median')

    ax.fill_between(x, rot_wealth_pct[p_idx[5], :], rot_wealth_pct[p_idx[95], :],
                    alpha=alpha_fan, color=color_rot, label='RoT 5-95%')
    ax.fill_between(x, rot_wealth_pct[p_idx[25], :], rot_wealth_pct[p_idx[75], :],
                    alpha=alpha_fan + 0.2, color=color_rot)
    ax.plot(x, rot_wealth_pct[p_idx[50], :],
            color=color_rot, linewidth=2, linestyle='--', label='RoT Median')

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth Comparison (Percentile Bands)')
    ax.legend(loc='upper left', fontsize=8)

    # (1,0): Rule-of-Thumb Allocation Glide Path (sample from first simulation)
    ax = axes[1, 0]
    rot_stock = comparison_result.result_b.stock_weight[0] * 100
    rot_bond = comparison_result.result_b.bond_weight[0] * 100
    rot_cash = comparison_result.result_b.cash_weight[0] * 100
    ax.stackplot(x, rot_stock, rot_bond, rot_cash,
                 labels=['Stocks', 'Bonds', 'Cash'],
                 colors=['#F4A261', '#9b59b6', '#95a5a6'],  # Coral/Purple/Gray (colorblind-safe)
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

    ldi_median_final = comparison_result.median_final_wealth('a')
    rot_median_final = comparison_result.median_final_wealth('b')

    summary_text = f"""
Strategy Comparison Summary
{'='*40}
Number of Simulations: {comparison_result.n_sims}

                    LDI        Rule-of-Thumb
                    -------    -------------
Default Rate:       {ldi_default_rate*100:6.1f}%    {rot_default_rate*100:6.1f}%
Median Final Wealth: ${ldi_median_final:,.0f}k   ${rot_median_final:,.0f}k

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

    ldi_default_ages = comparison_result.result_a.default_age
    rot_default_ages = comparison_result.result_b.default_age

    # Handle both scalar (single sim) and array (MC) cases
    if np.ndim(ldi_default_ages) == 0:
        ldi_default_ages = np.array([ldi_default_ages])
        rot_default_ages = np.array([rot_default_ages])

    ldi_valid = ldi_default_ages[~np.isnan(ldi_default_ages)]
    rot_valid = rot_default_ages[~np.isnan(rot_default_ages)]

    if len(ldi_valid) > 0 or len(rot_valid) > 0:
        bins = np.arange(params.retirement_age, params.end_age + 1, 2)

        if len(ldi_valid) > 0:
            ax.hist(ldi_valid, bins=bins, alpha=0.6, color=color_optimal,
                    label=f'LDI (n={len(ldi_valid)})', edgecolor='black')
        if len(rot_valid) > 0:
            ax.hist(rot_valid, bins=bins, alpha=0.6, color=color_rot,
                    label=f'RoT (n={len(rot_valid)})', edgecolor='black')

        ax.set_xlabel('Age at Default')
        ax.set_ylabel('Count')
        ax.set_title('Default Age Distribution')
        ax.legend(loc='upper right', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No defaults in\neither strategy',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Default Age Distribution')

    plt.suptitle(f'LDI vs Rule-of-Thumb Strategy Comparison{title_suffix}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def create_median_path_comparison_figure(
    comparison_result: 'StrategyComparison',
    params: 'LifecycleParams' = None,
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (14, 12),
    use_years: bool = True,
) -> plt.Figure:
    """
    Create a figure comparing LDI vs Rule-of-Thumb on deterministic median paths.

    Shows a 2x2 panel layout:
    - (0,0): Financial Wealth comparison
    - (0,1): Total Consumption comparison
    - (1,0): LDI Portfolio Allocation stackplot
    - (1,1): RoT Portfolio Allocation stackplot

    Args:
        comparison_result: StrategyComparison with result_a=LDI, result_b=RoT (single sim)
    """
    from core import LifecycleParams, EconomicParams, compute_earnings_profile

    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    if use_years:
        x = np.arange(len(comparison_result.ages))
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = comparison_result.ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    # Extract results
    ldi = comparison_result.result_a
    rot = comparison_result.result_b

    color_ldi = '#1A759F'       # Deep blue (colorblind-safe)
    color_rot = '#E9C46A'        # Amber (colorblind-safe)
    color_earnings = '#f39c12'

    # Compute earnings for display
    earnings_profile = compute_earnings_profile(params)
    working_years = params.retirement_age - params.start_age
    total_years = params.end_age - params.start_age
    earnings = np.zeros(total_years)
    earnings[:working_years] = earnings_profile

    # Get RoT strategy params
    rot_params = comparison_result.strategy_b_params
    rot_target_duration = rot_params.get('target_duration', 6.0)

    # (0,0): Financial Wealth Comparison
    ax = axes[0, 0]
    ax.plot(x, ldi.financial_wealth, color=color_ldi,
            linewidth=2.5, label='LDI Strategy')
    ax.plot(x, rot.financial_wealth, color=color_rot,
            linewidth=2.5, linestyle='--', label='Rule-of-Thumb')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5, label='Retirement')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,1): Consumption Comparison
    ax = axes[0, 1]
    ax.plot(x, ldi.consumption, color=color_ldi,
            linewidth=2.5, label='LDI Strategy')
    ax.plot(x, rot.consumption, color=color_rot,
            linewidth=2.5, linestyle='--', label='Rule-of-Thumb')
    ax.plot(x, earnings, color=color_earnings,
            linewidth=1.5, linestyle=':', alpha=0.7, label='Earnings')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Consumption')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,0): LDI Allocation Glide Path
    ax = axes[1, 0]
    ax.stackplot(x,
                 ldi.stock_weight * 100,
                 ldi.bond_weight * 100,
                 ldi.cash_weight * 100,
                 labels=['Stocks', 'Bonds', 'Cash'],
                 colors=['#F4A261', '#9b59b6', '#95a5a6'],  # Coral/Purple/Gray (colorblind-safe)
                 alpha=0.8)
    ax.axvline(x=retirement_x, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Allocation (%)')
    ax.set_title('LDI: Portfolio Allocation')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 100)

    # (1,1): RoT Allocation Glide Path
    ax = axes[1, 1]
    ax.stackplot(x,
                 rot.stock_weight * 100,
                 rot.bond_weight * 100,
                 rot.cash_weight * 100,
                 labels=['Stocks', 'Bonds', 'Cash'],
                 colors=['#F4A261', '#9b59b6', '#95a5a6'],  # Coral/Purple/Gray (colorblind-safe)
                 alpha=0.8)
    ax.axvline(x=retirement_x, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Allocation (%)')
    ax.set_title(f'Rule-of-Thumb: (100-Age)% Stock')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 100)

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


def create_allocation_comparison_page(
    comparison_beta0: 'StrategyComparison',
    comparison_beta_risky: 'StrategyComparison',
    params_beta0: 'LifecycleParams',
    params_beta_risky: 'LifecycleParams',
    figsize: Tuple[int, int] = (16, 10),
    use_years: bool = True,
) -> plt.Figure:
    """
    Create a single page comparing portfolio allocations for LDI vs RoT at different betas.

    Shows a 2x2 panel layout:
    - (0,0): LDI Allocation (Beta=0)
    - (0,1): LDI Allocation (Beta=risky)
    - (1,0): RoT Allocation (same for both betas)
    - (1,1): Summary text

    Args:
        comparison_beta0: StrategyComparison for Beta=0
        comparison_beta_risky: StrategyComparison for risky beta
        params_beta0: LifecycleParams for Beta=0
        params_beta_risky: LifecycleParams for risky beta
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Get the risky beta value from params
    risky_beta = params_beta_risky.stock_beta_human_capital

    if use_years:
        x = np.arange(len(comparison_beta0.ages))
        xlabel = 'Years from Career Start'
        retirement_x = params_beta0.retirement_age - params_beta0.start_age
    else:
        x = comparison_beta0.ages
        xlabel = 'Age'
        retirement_x = params_beta0.retirement_age

    stock_color = '#F4A261'   # Coral (colorblind-safe)
    bond_color = '#9b59b6'    # Purple (unchanged)
    cash_color = '#95a5a6'    # Gray (unchanged)

    # (0,0): LDI Allocation (Beta=0)
    ax = axes[0, 0]
    ldi_b0 = comparison_beta0.result_a
    ax.stackplot(x,
                 ldi_b0.stock_weight * 100,
                 ldi_b0.bond_weight * 100,
                 ldi_b0.cash_weight * 100,
                 labels=['Stocks', 'Bonds', 'Cash'],
                 colors=[stock_color, bond_color, cash_color],
                 alpha=0.8)
    ax.axvline(x=retirement_x, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Allocation (%)')
    ax.set_title('LDI Strategy (β = 0, Bond-like HC)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 100)

    # (0,1): LDI Allocation (Beta=risky)
    ax = axes[0, 1]
    ldi_risky = comparison_beta_risky.result_a
    ax.stackplot(x,
                 ldi_risky.stock_weight * 100,
                 ldi_risky.bond_weight * 100,
                 ldi_risky.cash_weight * 100,
                 labels=['Stocks', 'Bonds', 'Cash'],
                 colors=[stock_color, bond_color, cash_color],
                 alpha=0.8)
    ax.axvline(x=retirement_x, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Allocation (%)')
    ax.set_title(f'LDI Strategy (β = {risky_beta}, Risky HC)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 100)

    # (1,0): RoT Allocation (Beta=0)
    ax = axes[1, 0]
    rot_b0 = comparison_beta0.result_b
    ax.stackplot(x,
                 rot_b0.stock_weight * 100,
                 rot_b0.bond_weight * 100,
                 rot_b0.cash_weight * 100,
                 labels=['Stocks', 'Bonds', 'Cash'],
                 colors=[stock_color, bond_color, cash_color],
                 alpha=0.8)
    ax.axvline(x=retirement_x, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Allocation (%)')
    ax.set_title('Rule-of-Thumb: (100-Age)% Stock', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 100)

    # (1,1): Summary text panel
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
Portfolio Allocation Summary
════════════════════════════════════════

LDI Strategy adapts allocation based on:
  • Human capital composition (β)
  • Net worth (HC + FW - Expenses)
  • Mean-variance optimal weights

When β = 0 (bond-like human capital):
  → HC acts like a bond, so financial
    portfolio tilts toward stocks

When β = {risky_beta} (risky human capital):
  → HC has stock exposure, so financial
    portfolio reduces stock allocation

Rule-of-Thumb ignores human capital:
  • Stock weight = (100 - age)%
  • Same allocation regardless of β
"""

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.suptitle('PAGE 5: Portfolio Allocation Comparison — LDI vs Rule-of-Thumb',
                 fontsize=14, fontweight='bold', y=1.02)
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


# =============================================================================
# SINGLE-PANEL GAUGE FIGURES (for flexible slide layouts)
# =============================================================================

def create_gauge_net_worth_figure(
    params: 'LifecycleParams' = None,
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (10, 6),
    use_years: bool = True,
) -> plt.Figure:
    """
    Single panel: Net Worth = HC + FW - PV(Expenses).

    This is the key gauge showing "distance to destination" - whether you're
    on track to fund your retirement expenses.
    """
    from core import LifecycleParams, EconomicParams, compute_lifecycle_median_path

    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    result = compute_lifecycle_median_path(params, econ_params)

    fig, ax = plt.subplots(figsize=figsize)

    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = result.ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    color_nw = '#1D3557'  # Dark blue

    net_worth = result.human_capital + result.financial_wealth - result.pv_expenses

    ax.fill_between(x, 0, net_worth, where=net_worth >= 0, alpha=0.7, color=color_nw,
                    label='Net Worth')
    ax.fill_between(x, 0, net_worth, where=net_worth < 0, alpha=0.7, color='#E07A5F',
                    label='Underfunded')
    ax.plot(x, net_worth, color='black', linewidth=2)
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.7, label='Retirement')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Net Worth: HC + FW - PV(Expenses)', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    plt.tight_layout()
    return fig


def create_gauge_wealth_composition_figure(
    params: 'LifecycleParams' = None,
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (10, 6),
    use_years: bool = True,
) -> plt.Figure:
    """
    Single panel: Human Capital vs Financial Wealth composition.

    Shows how wealth composition shifts from HC-dominated to FW-dominated
    over the lifecycle. This drives portfolio allocation decisions.
    """
    from core import LifecycleParams, EconomicParams, compute_lifecycle_median_path

    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    result = compute_lifecycle_median_path(params, econ_params)

    fig, ax = plt.subplots(figsize=figsize)

    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = result.ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    color_hc = '#e67e22'  # Orange
    color_fw = '#457B9D'  # Blue

    # Stacked area showing composition
    ax.fill_between(x, 0, result.financial_wealth,
                    alpha=0.8, color=color_fw, label='Financial Wealth')
    ax.fill_between(x, result.financial_wealth,
                    result.financial_wealth + result.human_capital,
                    alpha=0.8, color=color_hc, label='Human Capital')

    # Total wealth line
    ax.plot(x, result.total_wealth, color='black', linewidth=2,
            linestyle='--', label='Total Wealth')

    ax.axvline(x=retirement_x, color='white', linestyle='--', linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Wealth Composition: Human Capital + Financial Wealth', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(x[0] - 1, x[-1] + 1)
    ax.set_ylim(0, None)

    plt.tight_layout()
    return fig


def create_control_allocation_figure(
    params: 'LifecycleParams' = None,
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (10, 6),
    use_years: bool = True,
) -> plt.Figure:
    """
    Single panel: Portfolio allocation stackplot.

    Shows how the financial portfolio allocation responds to changing
    wealth composition over the lifecycle.
    """
    from core import LifecycleParams, EconomicParams, compute_lifecycle_median_path

    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    result = compute_lifecycle_median_path(params, econ_params)

    fig, ax = plt.subplots(figsize=figsize)

    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = result.ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    color_stock = '#F4A261'  # Coral
    color_bond = '#9b59b6'   # Purple
    color_cash = '#95a5a6'   # Gray

    ax.stackplot(x,
                 result.stock_weight_no_short * 100,
                 result.bond_weight_no_short * 100,
                 result.cash_weight_no_short * 100,
                 labels=['Stocks', 'Bonds', 'Cash'],
                 colors=[color_stock, color_bond, color_cash],
                 alpha=0.8)
    ax.axvline(x=retirement_x, color='white', linestyle='--', linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Allocation (%)')
    ax.set_title('Portfolio Allocation (Responds to Wealth Composition)', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xlim(x[0], x[-1])

    plt.tight_layout()
    return fig


def create_control_consumption_figure(
    params: 'LifecycleParams' = None,
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (10, 6),
    use_years: bool = True,
) -> plt.Figure:
    """
    Single panel: Consumption path with subsistence and variable components.

    Shows how consumption responds to net worth - the variable component
    adjusts automatically to market conditions.
    """
    from core import LifecycleParams, EconomicParams, compute_lifecycle_median_path

    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    result = compute_lifecycle_median_path(params, econ_params)

    fig, ax = plt.subplots(figsize=figsize)

    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = result.ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    color_subsistence = '#95a5a6'  # Gray
    color_variable = '#2A9D8F'     # Teal

    ax.fill_between(x, 0, result.subsistence_consumption,
                    alpha=0.7, color=color_subsistence, label='Subsistence (fixed floor)')
    ax.fill_between(x, result.subsistence_consumption, result.total_consumption,
                    alpha=0.7, color=color_variable, label='Variable (net worth-based)')
    ax.plot(x, result.total_consumption, color='black', linewidth=1.5)
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.7, label='Retirement')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s per year)')
    ax.set_title('Consumption: Subsistence + Variable', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(x[0] - 1, x[-1] + 1)
    ax.set_ylim(0, None)

    plt.tight_layout()
    return fig


def create_gauge_duration_figure(
    params: 'LifecycleParams' = None,
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (10, 6),
    use_years: bool = True,
) -> plt.Figure:
    """
    Duration Gap gauge: interest rate sensitivity of balance sheet.

    Shows Duration of Earnings (asset) vs Duration of Expenses (liability),
    and the Duration Gap (difference). This parallels the Net Worth gauge
    but measures interest rate sensitivity (in years) rather than value (in $).

    Key insight: Duration matching hedges interest rate risk. When the gap is
    positive, assets have longer duration than liabilities (exposed to rate drops).
    When negative, liabilities have longer duration (exposed to rate rises).
    """
    from core import LifecycleParams, EconomicParams, compute_lifecycle_median_path

    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    result = compute_lifecycle_median_path(params, econ_params)

    fig, ax = plt.subplots(figsize=figsize)

    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = result.ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    color_earnings = '#e67e22'   # Orange (asset duration)
    color_expenses = '#E07A5F'   # Coral (liability duration)
    color_gap_pos = '#1D3557'    # Dark blue (positive gap)
    color_gap_neg = '#E07A5F'    # Coral (negative gap)

    # Duration Gap = Duration(Earnings) - Duration(Expenses)
    duration_gap = result.duration_earnings - result.duration_expenses

    # Plot duration lines
    ax.plot(x, result.duration_earnings, color=color_earnings, linewidth=2.5,
            label='Duration of Earnings (Asset)')
    ax.plot(x, result.duration_expenses, color=color_expenses, linewidth=2.5,
            label='Duration of Expenses (Liability)')

    # Fill duration gap (positive/negative regions)
    ax.fill_between(x, result.duration_expenses, result.duration_earnings,
                    where=duration_gap >= 0, alpha=0.3, color=color_gap_pos,
                    label='Positive Gap (rate-drop exposure)')
    ax.fill_between(x, result.duration_expenses, result.duration_earnings,
                    where=duration_gap < 0, alpha=0.3, color=color_gap_neg,
                    label='Negative Gap (rate-rise exposure)')

    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.7, label='Retirement')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Duration (years)')
    ax.set_title('Duration Gap: Interest Rate Sensitivity of Balance Sheet',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    plt.tight_layout()
    return fig
