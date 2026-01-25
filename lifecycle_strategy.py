"""
Lifecycle Investment Strategy Analysis and Visualization

This module implements a full lifecycle model including human capital during working years
and retirement spending, generating PDF outputs showing how optimal portfolio allocation
evolves along the median path.

Author: FINC 450 Life Cycle Investing
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Tuple, Optional, List
from core import (
    # Dataclasses
    EconomicParams,
    LifecycleParams,
    MonteCarloParams,
    LifecycleResult,
    MonteCarloResult,
    RuleOfThumbResult,
    StrategyComparisonResult,
    MedianPathComparisonResult,
    ScenarioResult,  # Now imported from core
    # Economic functions
    effective_duration,
    zero_coupon_price,
    compute_full_merton_allocation_constrained,
    generate_correlated_shocks,
    simulate_interest_rates,
    simulate_stock_returns,
    compute_present_value,
    compute_pv_consumption,
    compute_duration,
    compute_mv_optimal_allocation,
    # Simulation functions
    compute_earnings_profile,
    compute_expense_profile,
    compute_lifecycle_median_path,
    compute_lifecycle_fixed_consumption,
    compute_rule_of_thumb_strategy,
    compute_median_path_comparison,
    run_lifecycle_monte_carlo,
    run_strategy_comparison,
)

# Import visualization functions from the consolidated visualization module
from visualization import (
    # Styles and helpers
    COLORS,
    apply_wealth_log_scale,
    # Lifecycle plots
    plot_earnings_expenses_profile,
    plot_forward_present_values,
    plot_durations,
    plot_human_vs_financial_wealth,
    plot_hc_decomposition,
    plot_target_financial_holdings,
    plot_portfolio_shares,
    plot_total_wealth_holdings,
    plot_consumption_dollars,
    plot_consumption_breakdown,
    create_lifecycle_figure,
    # Monte Carlo plots
    create_monte_carlo_fan_chart,
    create_monte_carlo_detailed_view,
    create_teaching_scenarios_figure,
    create_sequence_of_returns_figure,
    # Comparison plots
    create_strategy_comparison_figure,
    create_median_path_comparison_figure,
    # Sensitivity plots
    create_beta_comparison_figure,
    create_gamma_comparison_figure,
    create_initial_wealth_comparison_figure,
    create_consumption_boost_comparison_figure,
    create_equity_premium_comparison_figure,
    create_income_comparison_figure,
    create_volatility_comparison_figure,
)


# =============================================================================
# Teaching Scenario Analysis
# =============================================================================


def create_teaching_scenario(
    name: str,
    description: str,
    returns_override: np.ndarray,
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
) -> ScenarioResult:
    """
    Create a teaching scenario with specified return sequence.

    Args:
        name: Scenario name
        description: Human-readable description
        returns_override: Array of stock returns for each year
        params: Lifecycle parameters
        econ_params: Economic parameters

    Returns:
        ScenarioResult with wealth and consumption paths
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    # Get baseline from median path
    median_result = compute_lifecycle_median_path(params, econ_params)

    total_years = params.end_age - params.start_age
    working_years = params.retirement_age - params.start_age

    # Compute target allocations
    if params.gamma > 0:
        target_stock, target_bond, target_cash = compute_mv_optimal_allocation(
            mu_stock=econ_params.mu_excess,
            mu_bond=econ_params.mu_bond,
            sigma_s=econ_params.sigma_s,
            sigma_r=econ_params.sigma_r,
            rho=econ_params.rho,
            duration=econ_params.bond_duration,
            gamma=params.gamma
        )
    else:
        target_stock = params.target_stock_allocation
        target_bond = params.target_bond_allocation
        target_cash = 1.0 - target_stock - target_bond

    # Initialize arrays
    financial_wealth = np.zeros(total_years)
    financial_wealth[0] = params.initial_wealth
    total_consumption = np.zeros(total_years)
    stock_weight = np.zeros(total_years)

    # Use median path values
    earnings = median_result.earnings
    expenses = median_result.expenses
    human_capital = median_result.human_capital
    pv_expenses = median_result.pv_expenses
    hc_stock = median_result.hc_stock_component
    hc_bond = median_result.hc_bond_component
    hc_cash = median_result.hc_cash_component

    # Consumption rate
    r = econ_params.r_bar
    avg_return = target_stock * (r + econ_params.mu_excess) + target_bond * r + target_cash * r
    consumption_rate = avg_return + params.consumption_boost

    defaulted = False

    for t in range(total_years):
        fw = financial_wealth[t]
        hc = human_capital[t]
        net_worth = hc + fw - pv_expenses[t]

        # Compute consumption
        subsistence = expenses[t]
        variable = max(0, consumption_rate * net_worth)
        total_cons = subsistence + variable

        # Apply constraints
        if t < working_years:
            if total_cons > earnings[t]:
                total_cons = earnings[t]
        else:
            if defaulted or fw <= 0:
                defaulted = True
                total_cons = 0
            elif total_cons > fw:
                total_cons = fw

        total_consumption[t] = total_cons

        # Compute portfolio weight
        total_wealth = fw + hc
        target_fin_stock = target_stock * total_wealth - hc_stock[t]

        if fw > 1e-6:
            w_stock = target_fin_stock / fw
            w_stock = max(0, min(1, w_stock))  # Constrain to [0, 1]
        else:
            w_stock = target_stock

        stock_weight[t] = w_stock

        # Evolve wealth
        if t < total_years - 1 and not defaulted:
            savings = earnings[t] - total_cons

            # Use overridden stock return
            stock_ret = returns_override[t]
            bond_ret = r + econ_params.mu_bond
            cash_ret = r

            w_b = target_bond / (target_bond + target_cash) * (1 - w_stock) if (target_bond + target_cash) > 0 else 0
            w_c = 1 - w_stock - w_b

            portfolio_return = w_stock * stock_ret + w_b * bond_ret + w_c * cash_ret
            financial_wealth[t + 1] = fw * (1 + portfolio_return) + savings

    cumulative_consumption = np.cumsum(total_consumption)

    return ScenarioResult(
        name=name,
        description=description,
        ages=median_result.ages,
        financial_wealth=financial_wealth,
        total_consumption=total_consumption,
        stock_weight=stock_weight,
        stock_returns=returns_override,
        cumulative_consumption=cumulative_consumption,
    )


def generate_teaching_scenarios(
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
) -> List[ScenarioResult]:
    """
    Generate a set of teaching scenarios illustrating key lifecycle concepts.

    Scenarios include:
    1. Median returns (baseline)
    2. Early crash (sequence risk - bad)
    3. Late crash (sequence risk - less bad)
    4. Bull market
    5. High volatility path

    Returns:
        List of ScenarioResult objects
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    total_years = params.end_age - params.start_age
    working_years = params.retirement_age - params.start_age
    r = econ_params.r_bar
    mu = econ_params.mu_excess

    # Median returns (deterministic)
    median_returns = np.full(total_years, r + mu)

    # Early crash: -30% for 3 years starting at retirement
    early_crash = median_returns.copy()
    early_crash[working_years:working_years + 3] = -0.30

    # Late crash: -30% for 3 years starting 10 years into retirement
    late_crash = median_returns.copy()
    late_start = working_years + 10
    if late_start + 3 <= total_years:
        late_crash[late_start:late_start + 3] = -0.30

    # Bull market: +15% every year
    bull_market = np.full(total_years, 0.15)

    # High volatility: alternating +20% and -10%
    high_vol = np.zeros(total_years)
    for t in range(total_years):
        high_vol[t] = 0.20 if t % 2 == 0 else -0.10

    scenarios = [
        create_teaching_scenario(
            "Median Returns",
            "Expected returns realized every year",
            median_returns, params, econ_params
        ),
        create_teaching_scenario(
            "Early Crash",
            "Market crashes 30% for 3 years at start of retirement",
            early_crash, params, econ_params
        ),
        create_teaching_scenario(
            "Late Crash",
            "Market crashes 30% for 3 years, 10 years into retirement",
            late_crash, params, econ_params
        ),
        create_teaching_scenario(
            "Bull Market",
            "Strong returns (15%) every year",
            bull_market, params, econ_params
        ),
        create_teaching_scenario(
            "High Volatility",
            "Alternating +20% and -10% returns",
            high_vol, params, econ_params
        ),
    ]

    return scenarios


def create_optimal_vs_4pct_rule_comparison(
    base_params: LifecycleParams = None,
    econ_params: EconomicParams = None,
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
    if base_params is None:
        base_params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    # Compute both strategies
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

    # Colors
    color_optimal = '#2ecc71'  # Green
    color_4pct = '#e74c3c'     # Red

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

    # Add annotations for totals
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


# =============================================================================
# NEW Page Functions (Matching TSX Visualizer Layout)
# =============================================================================

def create_base_case_page(
    result: LifecycleResult,
    params: LifecycleParams,
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (20, 24),
    use_years: bool = True
) -> plt.Figure:
    """
    Create Page 1: BASE CASE (Deterministic Median Path).

    Layout matches TSX visualizer with 4 sections, 10 charts total:
    - Section 1: Assumptions (2 charts: Earnings, Expenses)
    - Section 2: Forward-Looking Values (2 charts: Present Values, Durations)
    - Section 3: Wealth (4 charts: HC vs FW, HC Decomposition, Expense Decomposition, Net HC minus Expenses)
    - Section 4: Choices (2 charts: Consumption Path, Portfolio Allocation)
    """
    fig = plt.figure(figsize=figsize)

    # Create 5 rows x 2 columns layout
    # Row 0: Assumptions (Earnings, Expenses)
    # Row 1: Forward-Looking (PV, Duration)
    # Row 2: Wealth (HC vs FW, HC Decomposition)
    # Row 3: Wealth (Expense Decomposition, Net HC minus Expenses)
    # Row 4: Choices (Consumption, Portfolio Allocation)
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.25)

    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = result.ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    # Colors matching TSX
    COLORS = {
        'earnings': '#27ae60',
        'expenses': '#e74c3c',
        'stock': '#3498db',
        'bond': '#9b59b6',
        'cash': '#f1c40f',
        'fw': '#2ecc71',
        'hc': '#e67e22',
        'subsistence': '#95a5a6',
        'variable': '#e74c3c',
    }

    # ===== Section 1: Assumptions =====
    # Income and Expenses on same chart
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, result.earnings, color=COLORS['earnings'], linewidth=2, label='Earnings')
    ax.plot(x, result.expenses, color=COLORS['expenses'], linewidth=2, label='Expenses')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5, label='Retirement')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Income & Expenses ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # Savings = Earnings - Expenses (shows the lifecycle cash flow problem)
    ax = fig.add_subplot(gs[0, 1])
    savings = result.earnings - result.expenses
    ax.fill_between(x, 0, savings, where=savings >= 0, alpha=0.7, color=COLORS['earnings'], label='Savings')
    ax.fill_between(x, 0, savings, where=savings < 0, alpha=0.7, color=COLORS['expenses'], label='Drawdown')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Cash Flow: Earnings − Expenses ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # ===== Section 2: Forward-Looking Values =====
    # Present Values
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x, result.pv_earnings, color=COLORS['earnings'], linewidth=2, label='PV Earnings')
    ax.plot(x, result.pv_expenses, color=COLORS['expenses'], linewidth=2, label='PV Expenses')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Present Values ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # Durations
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x, result.duration_earnings, color=COLORS['earnings'], linewidth=2, label='Duration (Earnings)')
    ax.plot(x, result.duration_expenses, color=COLORS['expenses'], linewidth=2, label='Duration (Expenses)')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Years')
    ax.set_title('Durations (years)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # ===== Section 3: Wealth =====
    # HC vs FW (stacked area)
    ax = fig.add_subplot(gs[2, 0])
    ax.fill_between(x, 0, result.financial_wealth, alpha=0.7, color=COLORS['fw'], label='Financial Wealth')
    ax.fill_between(x, result.financial_wealth, result.financial_wealth + result.human_capital,
                   alpha=0.7, color=COLORS['hc'], label='Human Capital')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Human Capital vs Financial Wealth ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # HC Decomposition (line chart - allows negative values)
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

    # Expense Liability Decomposition (line chart - allows negative values)
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

    # Net HC minus Expenses (line chart - allows negative values)
    ax = fig.add_subplot(gs[3, 1])
    # Net = HC - Expenses for each component
    net_stock = result.hc_stock_component  # Expenses have no stock component
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
    # Consumption Path (stacked area)
    ax = fig.add_subplot(gs[4, 0])
    ax.fill_between(x, 0, result.subsistence_consumption, alpha=0.7, color=COLORS['subsistence'], label='Subsistence')
    ax.fill_between(x, result.subsistence_consumption, result.total_consumption,
                   alpha=0.7, color=COLORS['variable'], label='Variable')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Consumption Path ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # Portfolio Allocation (stacked area, %)
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
    params: LifecycleParams,
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (20, 22),
    use_years: bool = True,
    percentiles: List[int] = None,
) -> plt.Figure:
    """
    Create Page 2: MONTE CARLO (50 Runs).

    Layout matches TSX visualizer with 6 chart panels:
    - Consumption Distribution (percentile lines)
    - Financial Wealth Distribution (percentile lines)
    - Net Worth Distribution (percentile lines)
    - Terminal Values Grid (text summary)
    - Cumulative Stock Returns (percentile bands)
    - Interest Rate Paths (percentile bands)
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

    # Colors
    COLORS = {
        'consumption': '#e74c3c',
        'fw': '#2ecc71',
        'nw': '#9b59b6',
        'stock': '#3498db',
        'rate': '#f39c12',
    }

    # Compute percentiles
    consumption_pctls = np.percentile(mc_result.total_consumption_paths, percentiles, axis=0)
    fw_pctls = np.percentile(mc_result.financial_wealth_paths, percentiles, axis=0)

    # Net Worth = HC + FW - PV_expenses
    net_worth_paths = (mc_result.human_capital_paths + mc_result.financial_wealth_paths -
                       mc_result.median_result.pv_expenses[np.newaxis, :])
    nw_pctls = np.percentile(net_worth_paths, percentiles, axis=0)

    # Stock returns cumulative - ensure shape matches total_years
    stock_return_data = mc_result.stock_return_paths[:, :total_years]
    stock_cumulative = np.cumprod(1 + stock_return_data, axis=1)
    stock_pctls = np.percentile(stock_cumulative, percentiles, axis=0)

    # Interest rate paths - ensure shape matches total_years
    rate_data = mc_result.interest_rate_paths[:, :total_years]
    rate_pctls = np.percentile(rate_data * 100, percentiles, axis=0)  # Convert to %

    # Style for percentile lines
    line_styles = {0: ':', 1: '--', 2: '-', 3: '--', 4: ':'}  # 5th, 25th, 50th, 75th, 95th
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

    # Terminal values at end age
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

    # ===== Cumulative Stock Returns (with bands) =====
    ax = fig.add_subplot(gs[2, 0])
    # Use log scale for stock returns
    log_stock_pctls = np.log(stock_pctls)

    # Fill bands
    ax.fill_between(x, log_stock_pctls[0], log_stock_pctls[1], alpha=0.15, color=COLORS['stock'])
    ax.fill_between(x, log_stock_pctls[1], log_stock_pctls[3], alpha=0.3, color=COLORS['stock'])
    ax.fill_between(x, log_stock_pctls[3], log_stock_pctls[4], alpha=0.15, color=COLORS['stock'])

    # Lines
    ax.plot(x, log_stock_pctls[2], color=COLORS['stock'], linewidth=2, label='Median')
    ax.plot(x, log_stock_pctls[1], color=COLORS['stock'], linewidth=1, linestyle='--', label='25th/75th')
    ax.plot(x, log_stock_pctls[3], color=COLORS['stock'], linewidth=1, linestyle='--')

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Log Cumulative Return')
    ax.set_title('Cumulative Stock Returns (Log Scale)', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)

    # Format y-axis to show as percentage
    def log_to_pct(y, pos):
        return f'{np.exp(y)*100:.0f}%'
    ax.yaxis.set_major_formatter(plt.FuncFormatter(log_to_pct))

    # ===== Interest Rate Paths =====
    ax = fig.add_subplot(gs[2, 1])
    # Fill bands
    ax.fill_between(x, rate_pctls[0], rate_pctls[1], alpha=0.15, color=COLORS['rate'])
    ax.fill_between(x, rate_pctls[1], rate_pctls[3], alpha=0.3, color=COLORS['rate'])
    ax.fill_between(x, rate_pctls[3], rate_pctls[4], alpha=0.15, color=COLORS['rate'])

    # Lines
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
    params: LifecycleParams,
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
    - PV Consumption comparison (at time 0, not total)
    - Financial Wealth percentile charts
    - Consumption percentile charts
    - Portfolio Allocation comparison
    """
    from retirement_simulation import (
        generate_correlated_shocks,
        simulate_interest_rates,
        simulate_stock_returns,
    )

    if rate_shock_age is None:
        rate_shock_age = params.retirement_age  # Default: shock at retirement

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

    # Run simulation based on scenario type
    bad_returns_early = scenario_type == 'sequenceRisk'

    # Generate random shocks
    rng = np.random.default_rng(random_seed)
    rate_shocks, stock_shocks = generate_correlated_shocks(
        total_years, n_simulations, econ_params.rho, rng
    )

    # Simulate paths
    initial_rate = econ_params.r_bar
    rate_paths = simulate_interest_rates(
        initial_rate, total_years, n_simulations, econ_params, rate_shocks
    )

    # Apply rate shock if applicable
    if scenario_type == 'rateShock':
        shock_year = rate_shock_age - params.start_age
        if 0 <= shock_year < total_years:
            for sim in range(n_simulations):
                for t in range(shock_year, total_years):
                    rate_paths[sim, t] += rate_shock_magnitude

    stock_return_paths = simulate_stock_returns(rate_paths, econ_params, stock_shocks)

    # Apply bad returns early if sequenceRisk scenario
    if bad_returns_early:
        for sim in range(n_simulations):
            for t in range(working_years, min(working_years + 5, total_years)):
                stock_return_paths[sim, t] = -0.20

    # Run strategy comparison
    comparison = run_strategy_comparison(
        params=params,
        econ_params=econ_params,
        n_simulations=n_simulations,
        random_seed=random_seed,
        bad_returns_early=bad_returns_early,
    )

    # Colors
    COLORS = {
        'optimal': '#2ecc71',
        'rot': '#3498db',
    }

    # Scenario titles
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
    # Use percentiles for PV consumption
    pv_data = {
        'Optimal': comparison.optimal_pv_consumption_percentiles,
        'Rule of Thumb': comparison.rot_pv_consumption_percentiles,
    }

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

    # ===== Stock Return Paths (for sequenceRisk/rateShock) or Allocation Comparison =====
    if scenario_type == 'sequenceRisk':
        ax = fig.add_subplot(gs[2, 0])
        # Show cumulative stock returns - slice to match total_years
        stock_data = stock_return_paths[:, :total_years]
        cumulative = np.cumprod(1 + stock_data, axis=1)
        pctls = np.percentile(cumulative, [5, 25, 50, 75, 95], axis=0)

        ax.fill_between(x, pctls[0], pctls[4], alpha=0.2, color='#3498db')
        ax.fill_between(x, pctls[1], pctls[3], alpha=0.3, color='#3498db')
        ax.plot(x, pctls[2], color='#3498db', linewidth=2, label='Median')
        ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)

        # Highlight bad return period
        ax.axvspan(retirement_x, retirement_x + 5, alpha=0.2, color='red', label='Forced -20% Returns')

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Stock Return Paths (Showing Stress Period)', fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)

    elif scenario_type == 'rateShock':
        ax = fig.add_subplot(gs[2, 0])
        # Show interest rate paths - slice to match total_years
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
        # Portfolio allocation comparison
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


def generate_lifecycle_pdf(
    output_path: str = 'lifecycle_strategy.pdf',
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    include_legacy_pages: bool = False,
    use_years: bool = True,
    rot_savings_rate: float = 0.15,
    rot_target_duration: float = 6.0,
    rot_withdrawal_rate: float = 0.04,
) -> str:
    """
    Generate a PDF report showing lifecycle investment strategy.

    STRUCTURE:
    - Pages 1-3: Deterministic Median Path for Beta = 0.0, 0.5, 1.0
    - Page 4: Effect of Stock Beta comparison
    - Page 5: LDI vs Rule-of-Thumb Median Path Comparison
    - Page 6: Monte Carlo Strategy Comparison

    Args:
        output_path: Path for output PDF file
        params: Lifecycle parameters (uses defaults if None)
        econ_params: Economic parameters (uses defaults if None)
        include_legacy_pages: If True, include old comparison pages (gamma, wealth, etc.)
        use_years: If True, x-axis shows years from career start; if False, shows age
        rot_savings_rate: Rule-of-Thumb savings rate (default 15%)
        rot_target_duration: Rule-of-Thumb FI target duration (default 6)
        rot_withdrawal_rate: Rule-of-Thumb withdrawal rate (default 4%)

    Returns:
        Path to generated PDF file
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    beta_values = [0.0, 0.5, 1.0]

    with PdfPages(output_path) as pdf:
        # ====================================================================
        # PAGES 1-3: DETERMINISTIC MEDIAN PATH for each Beta
        # ====================================================================
        for page_num, beta in enumerate(beta_values, start=1):
            print(f"Generating Page {page_num}: Deterministic Path (Beta = {beta})...")

            # Create params for this beta
            beta_params = LifecycleParams(
                start_age=params.start_age,
                retirement_age=params.retirement_age,
                end_age=params.end_age,
                initial_earnings=params.initial_earnings,
                earnings_growth=params.earnings_growth,
                earnings_hump_age=params.earnings_hump_age,
                earnings_decline=params.earnings_decline,
                base_expenses=params.base_expenses,
                expense_growth=params.expense_growth,
                retirement_expenses=params.retirement_expenses,
                stock_beta_human_capital=beta,
                gamma=params.gamma,
                target_stock_allocation=params.target_stock_allocation,
                target_bond_allocation=params.target_bond_allocation,
                risk_free_rate=params.risk_free_rate,
                equity_premium=params.equity_premium,
                initial_wealth=params.initial_wealth,
                consumption_boost=params.consumption_boost,
            )
            result = compute_lifecycle_median_path(beta_params, econ_params)

            fig = create_base_case_page(
                result=result,
                params=beta_params,
                econ_params=econ_params,
                figsize=(20, 24),
                use_years=use_years
            )
            # Update the title to show the beta value
            fig.suptitle(f'PAGE {page_num}: DETERMINISTIC MEDIAN PATH (Beta = {beta})',
                        fontsize=16, fontweight='bold', y=0.995)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # ====================================================================
        # PAGE 4: BETA COMPARISON (Effect of Stock Beta)
        # ====================================================================
        print("Generating Page 4: Beta Comparison...")
        fig = create_beta_comparison_figure(
            beta_values=beta_values,
            base_params=params,
            econ_params=econ_params,
            figsize=(16, 10),
            use_years=use_years
        )
        fig.suptitle('Effect of Stock Beta on Portfolio Allocation & Human Capital',
                    fontsize=14, fontweight='bold', y=1.02)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ====================================================================
        # PAGE 5: LDI vs RULE-OF-THUMB MEDIAN PATH COMPARISON
        # ====================================================================
        print("Generating Page 5: LDI vs Rule-of-Thumb Median Path Comparison...")
        median_comparison = compute_median_path_comparison(
            params=params,
            econ_params=econ_params,
            rot_savings_rate=rot_savings_rate,
            rot_target_duration=rot_target_duration,
            rot_withdrawal_rate=rot_withdrawal_rate,
        )
        fig = create_median_path_comparison_figure(
            comparison_result=median_comparison,
            params=params,
            econ_params=econ_params,
            figsize=(18, 14),
            use_years=use_years,
        )
        fig.suptitle('PAGE 5: LDI vs Rule-of-Thumb (Deterministic Median Path)',
                    fontsize=16, fontweight='bold', y=1.02)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ====================================================================
        # PAGE 6: MONTE CARLO STRATEGY COMPARISON
        # ====================================================================
        print("Generating Page 6: Monte Carlo Strategy Comparison...")
        mc_comparison = run_strategy_comparison(
            params=params,
            econ_params=econ_params,
            n_simulations=100,
            random_seed=42,
            rot_savings_rate=rot_savings_rate,
            rot_target_duration=rot_target_duration,
            rot_withdrawal_rate=rot_withdrawal_rate,
        )
        fig = create_strategy_comparison_figure(
            comparison_result=mc_comparison,
            params=params,
            figsize=(18, 12),
            use_years=use_years,
        )
        fig.suptitle('PAGE 6: LDI vs Rule-of-Thumb (Monte Carlo Comparison)',
                    fontsize=16, fontweight='bold', y=1.02)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ====================================================================
        # LEGACY PAGES (optional - for parameter sensitivity analysis)
        # ====================================================================
        if include_legacy_pages:
            print("Generating legacy comparison pages...")

            # Gamma (Risk Aversion) comparison
            fig = create_gamma_comparison_figure(
                gamma_values=[1.0, 2.0, 4.0, 8.0],
                base_params=params,
                econ_params=econ_params,
                figsize=(16, 10),
                use_years=use_years
            )
            fig.suptitle('Effect of Risk Aversion (γ) on Lifecycle Strategy',
                        fontsize=14, fontweight='bold', y=1.02)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Initial Wealth comparison
            fig = create_initial_wealth_comparison_figure(
                wealth_values=[-50, 0, 50, 200],
                base_params=params,
                econ_params=econ_params,
                figsize=(16, 10),
                use_years=use_years
            )
            fig.suptitle('Effect of Initial Wealth on Lifecycle Strategy\n(Negative = Student Loans)',
                        fontsize=14, fontweight='bold', y=1.02)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Equity Premium comparison
            fig = create_equity_premium_comparison_figure(
                premium_values=[0.02, 0.04, 0.06, 0.08],
                base_params=params,
                econ_params=econ_params,
                figsize=(16, 10),
                use_years=use_years
            )
            fig.suptitle('Effect of Equity Risk Premium on Lifecycle Strategy',
                        fontsize=14, fontweight='bold', y=1.02)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Stock Volatility comparison
            fig = create_volatility_comparison_figure(
                volatility_values=[0.12, 0.18, 0.24, 0.30],
                base_params=params,
                econ_params=econ_params,
                figsize=(16, 10),
                use_years=use_years
            )
            fig.suptitle('Effect of Stock Volatility (σ) on Lifecycle Strategy',
                        fontsize=14, fontweight='bold', y=1.02)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Monte Carlo page
            print("Generating Monte Carlo page...")
            mc_params = MonteCarloParams(n_simulations=50, random_seed=42)
            mc_result = run_lifecycle_monte_carlo(params, econ_params, mc_params)
            fig = create_monte_carlo_page(
                mc_result=mc_result,
                params=params,
                econ_params=econ_params,
                figsize=(20, 22),
                use_years=use_years
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Teaching scenarios
            for scenario_type in ['normal', 'sequenceRisk', 'rateShock']:
                print(f"Generating {scenario_type} scenario page...")
                fig = create_scenario_page(
                    scenario_type=scenario_type,
                    params=params,
                    econ_params=econ_params,
                    figsize=(20, 18),
                    use_years=use_years,
                    n_simulations=50,
                    random_seed=42,
                    rate_shock_age=params.retirement_age if scenario_type == 'rateShock' else None,
                    rate_shock_magnitude=-0.02 if scenario_type == 'rateShock' else None
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        # Summary page with parameters
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')

        # Compute MV optimal allocation for summary
        if params.gamma > 0:
            mv_stock, mv_bond, mv_cash = compute_mv_optimal_allocation(
                mu_stock=econ_params.mu_excess,
                mu_bond=econ_params.mu_bond,
                sigma_s=econ_params.sigma_s,
                sigma_r=econ_params.sigma_r,
                rho=econ_params.rho,
                duration=econ_params.bond_duration,
                gamma=params.gamma
            )
            mv_formula = f"w* = (1/gamma) * Sigma^(-1) * mu (Full VCV Merton solution)"
            allocation_source = "Mean-Variance Optimization (Full VCV)"
        else:
            mv_stock = params.target_stock_allocation
            mv_bond = params.target_bond_allocation
            mv_cash = 1 - mv_stock - mv_bond
            mv_formula = "Fixed target allocations (gamma=0)"
            allocation_source = "Fixed Targets"

        summary_text = f"""
Lifecycle Investment Strategy Parameters
========================================

Age Parameters:
  - Career Start: {params.start_age}
  - Retirement Age: {params.retirement_age}
  - Planning Horizon: {params.end_age}

Income Parameters:
  - Initial Earnings: ${params.initial_earnings:,.0f}k
  - Earnings Growth: {params.earnings_growth*100:.1f}%
  - Peak Earnings Age: {params.earnings_hump_age}

Subsistence Expense Parameters:
  - Base Expenses: ${params.base_expenses:,.0f}k
  - Retirement Expenses: ${params.retirement_expenses:,.0f}k

Initial Wealth:
  - Starting Financial Wealth: ${params.initial_wealth:,.0f}k

Consumption Model:
  - Total Consumption = Subsistence + Rate x Net Worth
  - Net Worth = Human Capital + Financial Wealth - PV(Future Expenses)
  - Consumption Rate = Median Return + {params.consumption_boost*100:.1f}pp

Human Capital Allocation:
  - Stock Beta: {params.stock_beta_human_capital:.2f}
  - Bond Duration: {econ_params.bond_duration:.1f} years (used for HC decomposition and MV optimization)

Mean-Variance Optimization (Full VCV):
  - Risk-Free Rate (r_bar): {econ_params.r_bar*100:.1f}%
  - Stock Excess Return (mu_s): {econ_params.mu_excess*100:.1f}%
  - Bond Sharpe Ratio: {econ_params.bond_sharpe:.2f} → mu_b = {econ_params.mu_bond*100:.2f}%
  - Stock Volatility (sigma_s): {econ_params.sigma_s*100:.0f}%
  - Rate Shock Volatility (sigma_r): {econ_params.sigma_r*100:.1f}%
  - Rate/Stock Correlation (rho): {econ_params.rho:.2f}
  - Risk Aversion (gamma): {params.gamma:.1f}
  - Allocation Source: {allocation_source}
  - {mv_formula}

VCV-Based Asset Return Models:
  - Stock: R_s = r + mu_s + sigma_s * eps_s
  - Bond:  R_b = r + mu_b - D * sigma_r * eps_r
  - Bond Vol: D * sigma_r = {econ_params.bond_duration * econ_params.sigma_r*100:.1f}%
  - Cov(R_s,R_b): -D*sigma_s*sigma_r*rho = {-econ_params.bond_duration * econ_params.sigma_s * econ_params.sigma_r * econ_params.rho*100:.3f}%

Target Total Wealth Allocation (from MV):
  - Stocks: {mv_stock*100:.1f}%
  - Bonds: {mv_bond*100:.1f}%
  - Cash: {mv_cash*100:.1f}%

Key Insights:
-------------
1. Portfolio allocation is derived from full Merton
   solution: w* = (1/gamma) * Sigma^(-1) * mu

2. The VCV matrix accounts for bond return volatility
   from duration and rate shock correlation with stocks.

3. Changing gamma, mu, sigma, rho, or duration allows
   studying how portfolios respond to assumptions.

4. Human capital is treated as implicit asset holdings,
   and financial portfolio adjusts to reach total targets.
"""

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    return output_path


# =============================================================================
# Main Entry Point
# =============================================================================

def main(
    output_path: str = 'lifecycle_strategy.pdf',
    start_age: int = 25,
    retirement_age: int = 65,
    end_age: int = 85,
    initial_earnings: float = 150,
    stock_beta_hc: float = 0.0,
    bond_duration: float = 20.0,
    gamma: float = 2.0,
    mu_excess: float = 0.04,
    bond_sharpe: float = 0.0,
    sigma_s: float = 0.18,
    sigma_r: float = 0.006,
    rho: float = 0.0,
    r_bar: float = 0.02,
    consumption_share: float = 0.05,
    consumption_boost: float = 0.0,
    initial_wealth: float = 100.0,
    include_scenarios: bool = True,
    use_years: bool = True,
    verbose: bool = True,
    rot_savings_rate: float = 0.15,
    rot_target_duration: float = 6.0,
    rot_withdrawal_rate: float = 0.04,
):
    """
    Generate lifecycle strategy PDF with configurable parameters.

    Args:
        output_path: Path for output PDF file
        start_age: Age at career start
        retirement_age: Age at retirement
        end_age: Planning horizon end
        initial_earnings: Starting annual earnings in $000s
        stock_beta_hc: Beta of human capital to stocks
        bond_duration: Bond duration for MV optimization (years)
        gamma: Risk aversion coefficient for MV optimization (0 = use fixed targets)
        mu_excess: Equity risk premium (stock excess return)
        bond_sharpe: Bond Sharpe ratio (mu_bond = bond_sharpe * bond_duration * sigma_r)
        sigma_s: Stock return volatility
        sigma_r: Interest rate shock volatility
        rho: Correlation between rate shocks and stock shocks
        r_bar: Long-run real risk-free rate
        consumption_share: Share of net worth consumed above subsistence
        consumption_boost: Boost above median return for consumption rate (default 1%)
        initial_wealth: Initial financial wealth in $000s (can be negative for student loans)
        include_scenarios: If True, include scenario comparison pages in PDF
        use_years: If True, x-axis shows years from start; if False, shows age
        verbose: If True, print progress and statistics
        rot_savings_rate: Rule-of-Thumb savings rate during working years (default 15%)
        rot_target_duration: Rule-of-Thumb target duration for fixed income (default 6)
        rot_withdrawal_rate: Rule-of-Thumb withdrawal rate in retirement (default 4%)
    """
    if verbose:
        print("Computing lifecycle investment strategy...")

    # Create economic parameters with consistent DGP
    econ_params = EconomicParams(
        r_bar=r_bar,
        mu_excess=mu_excess,
        bond_sharpe=bond_sharpe,
        sigma_s=sigma_s,
        sigma_r=sigma_r,
        rho=rho,
        bond_duration=bond_duration,
    )

    # Compute MV optimal allocation for display
    if gamma > 0:
        opt_stock, opt_bond, opt_cash = compute_mv_optimal_allocation(
            mu_stock=mu_excess,
            mu_bond=econ_params.mu_bond,
            sigma_s=sigma_s,
            sigma_r=sigma_r,
            rho=rho,
            duration=bond_duration,
            gamma=gamma
        )
        if verbose:
            print(f"MV Optimal Allocation (gamma={gamma}): "
                  f"Stocks={opt_stock:.1%}, Bonds={opt_bond:.1%}, Cash={opt_cash:.1%}")
    else:
        opt_stock, opt_bond, opt_cash = 0.60, 0.30, 0.10  # fallback

    params = LifecycleParams(
        start_age=start_age,
        retirement_age=retirement_age,
        end_age=end_age,
        initial_earnings=initial_earnings,
        stock_beta_human_capital=stock_beta_hc,
        gamma=gamma,
        target_stock_allocation=opt_stock,
        target_bond_allocation=opt_bond,
        consumption_share=consumption_share,
        consumption_boost=consumption_boost,
        initial_wealth=initial_wealth,
        risk_free_rate=r_bar,
        equity_premium=mu_excess,
    )

    output = generate_lifecycle_pdf(
        output_path=output_path,
        params=params,
        econ_params=econ_params,
        include_legacy_pages=include_scenarios,  # Legacy pages are now optional
        use_years=use_years,
        rot_savings_rate=rot_savings_rate,
        rot_target_duration=rot_target_duration,
        rot_withdrawal_rate=rot_withdrawal_rate,
    )

    if verbose:
        print(f"PDF generated: {output}")

        # Also print some key statistics
        result = compute_lifecycle_median_path(params, econ_params)

        print("\nKey Statistics at Selected Ages:")
        print("-" * 70)
        print(f"{'Age':>5} {'Earnings':>12} {'Human Cap':>12} {'Fin Wealth':>12} {'Stock Wt':>10}")
        print("-" * 70)

        for age_idx in [0, 10, 20, 30, 39, 45, 55]:
            if age_idx < len(result.ages):
                age = result.ages[age_idx]
                print(f"{age:>5} {result.earnings[age_idx]:>12,.0f} {result.human_capital[age_idx]:>12,.0f} "
                      f"{result.financial_wealth[age_idx]:>12,.0f} {result.stock_weight_no_short[age_idx]:>10.1%}")

    return output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate lifecycle investment strategy PDF'
    )
    parser.add_argument('-o', '--output', default='lifecycle_strategy.pdf',
                       help='Output PDF file path')
    parser.add_argument('--start-age', type=int, default=25,
                       help='Age at career start (default: 25)')
    parser.add_argument('--retirement-age', type=int, default=65,
                       help='Retirement age (default: 65)')
    parser.add_argument('--end-age', type=int, default=85,
                       help='Planning horizon end (default: 85)')
    parser.add_argument('--initial-earnings', type=float, default=150,
                       help='Initial earnings in $000s (default: 150)')
    parser.add_argument('--stock-beta', type=float, default=0.0,
                       help='Stock beta of human capital (default: 0.0)')
    parser.add_argument('--bond-duration', type=float, default=20.0,
                       help='Bond duration for MV optimization in years (default: 7.0)')
    parser.add_argument('--gamma', type=float, default=2.0,
                       help='Risk aversion for MV optimization (default: 2.0, 0=use fixed targets)')
    parser.add_argument('--mu-excess', type=float, default=0.04,
                       help='Equity risk premium (default: 0.04 = 4%%)')
    parser.add_argument('--bond-sharpe', type=float, default=0.0,
                       help='Bond Sharpe ratio (default: 0.037); mu_bond = sharpe * duration * sigma_r')
    parser.add_argument('--sigma', type=float, default=0.18,
                       help='Stock return volatility (default: 0.18 = 18%%)')
    parser.add_argument('--sigma-r', type=float, default=0.006,
                       help='Interest rate shock volatility (default: 0.006 = 0.6%%)')
    parser.add_argument('--rho', type=float, default=0.0,
                       help='Correlation between rate and stock shocks (default: 0.0)')
    parser.add_argument('--r-bar', type=float, default=0.02,
                       help='Long-run real risk-free rate (default: 0.02 = 2%%)')
    parser.add_argument('--consumption-share', type=float, default=0.05,
                       help='Share of net worth consumed above subsistence (default: 0.05)')
    parser.add_argument('--consumption-boost', type=float, default=0.0,
                       help='Boost above median return for consumption rate (default: 0.0)')
    parser.add_argument('--initial-wealth', type=float, default=100,
                       help='Initial financial wealth in $000s (default: 100, can be negative for student loans)')
    parser.add_argument('--use-age', action='store_true',
                       help='Use age instead of years from start on x-axis')
    parser.add_argument('--no-scenarios', action='store_true',
                       help='Skip legacy comparison pages (gamma, wealth, etc.) - main 5 pages always included')
    parser.add_argument('--include-legacy', action='store_true',
                       help='Include legacy parameter sensitivity pages (beta, gamma, volatility comparisons)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress output messages')

    # Rule-of-Thumb strategy parameters
    parser.add_argument('--rot-savings-rate', type=float, default=0.15,
                       help='Rule-of-Thumb savings rate (default: 0.15 = 15%%)')
    parser.add_argument('--rot-target-duration', type=float, default=6.0,
                       help='Rule-of-Thumb fixed income target duration in years (default: 6)')
    parser.add_argument('--rot-withdrawal-rate', type=float, default=0.04,
                       help='Rule-of-Thumb retirement withdrawal rate (default: 0.04 = 4%%)')

    args = parser.parse_args()

    main(
        output_path=args.output,
        start_age=args.start_age,
        retirement_age=args.retirement_age,
        end_age=args.end_age,
        initial_earnings=args.initial_earnings,
        stock_beta_hc=args.stock_beta,
        bond_duration=args.bond_duration,
        gamma=args.gamma,
        mu_excess=args.mu_excess,
        bond_sharpe=args.bond_sharpe,
        sigma_s=args.sigma,
        sigma_r=args.sigma_r,
        rho=args.rho,
        r_bar=args.r_bar,
        consumption_share=args.consumption_share,
        consumption_boost=args.consumption_boost,
        initial_wealth=args.initial_wealth,
        include_scenarios=args.include_legacy,  # Legacy pages now opt-in
        use_years=not args.use_age,
        verbose=not args.quiet,
        rot_savings_rate=args.rot_savings_rate,
        rot_target_duration=args.rot_target_duration,
        rot_withdrawal_rate=args.rot_withdrawal_rate,
    )
