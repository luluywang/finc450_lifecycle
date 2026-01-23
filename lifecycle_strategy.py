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
from dataclasses import dataclass
from typing import Tuple, Optional
from retirement_simulation import (
    EconomicParams,
    effective_duration,
    zero_coupon_price,
)


# =============================================================================
# Lifecycle Parameters
# =============================================================================

@dataclass
class LifecycleParams:
    """Parameters for lifecycle model."""
    # Age parameters
    start_age: int = 25          # Age at career start
    retirement_age: int = 65     # Age at retirement
    end_age: int = 85            # Planning horizon (death or planning end)

    # Income parameters (in $000s for cleaner numbers)
    initial_earnings: float = 100    # Starting annual earnings ($100k)
    earnings_growth: float = 0.02    # Real earnings growth rate
    earnings_hump_age: int = 50      # Age at peak earnings
    earnings_decline: float = 0.01   # Decline rate after peak

    # Expense parameters (subsistence/baseline)
    base_expenses: float = 60        # Base annual subsistence expenses ($60k)
    expense_growth: float = 0.01     # Real expense growth rate
    retirement_expenses: float = 80  # Retirement subsistence expenses ($80k)

    # Consumption parameters
    consumption_share: float = 0.05  # Share of net worth consumed above subsistence

    # Asset allocation parameters
    stock_beta_human_capital: float = 0.1    # Beta of human capital to stocks
    bond_duration_benchmark: float = 20.0    # Benchmark bond duration for HC allocation
    target_stock_allocation: float = 0.60    # Target long-run stock allocation
    target_bond_allocation: float = 0.30     # Target long-run bond allocation

    # Economic parameters
    risk_free_rate: float = 0.02     # Long-run real risk-free rate
    equity_premium: float = 0.04     # Equity risk premium

    # Initial financial wealth
    initial_wealth: float = 1        # Starting financial wealth ($1k)


@dataclass
class LifecycleResult:
    """Results from lifecycle calculations along median path."""
    ages: np.ndarray

    # Income/Expense profiles
    earnings: np.ndarray
    expenses: np.ndarray
    savings: np.ndarray

    # Present values (forward looking)
    pv_earnings: np.ndarray
    pv_expenses: np.ndarray

    # Human capital decomposition
    human_capital: np.ndarray
    hc_stock_component: np.ndarray
    hc_bond_component: np.ndarray
    hc_cash_component: np.ndarray

    # Durations
    duration_earnings: np.ndarray
    duration_expenses: np.ndarray

    # Financial wealth
    financial_wealth: np.ndarray
    total_wealth: np.ndarray

    # Target financial portfolio
    target_fin_stocks: np.ndarray
    target_fin_bonds: np.ndarray
    target_fin_cash: np.ndarray

    # Portfolio shares (constrained to no short)
    stock_weight_no_short: np.ndarray
    bond_weight_no_short: np.ndarray
    cash_weight_no_short: np.ndarray

    # Total wealth holdings
    total_stocks: np.ndarray
    total_bonds: np.ndarray
    total_cash: np.ndarray

    # Consumption model
    net_worth: np.ndarray              # HC + FW - PV(future expenses)
    subsistence_consumption: np.ndarray # Baseline/floor consumption
    variable_consumption: np.ndarray    # Share of net worth consumed
    total_consumption: np.ndarray       # Subsistence + variable
    consumption_share_of_fw: np.ndarray # Total consumption / financial wealth


# =============================================================================
# Lifecycle Calculations
# =============================================================================

def compute_earnings_profile(params: LifecycleParams) -> np.ndarray:
    """
    Compute earnings profile over working life.

    Uses a hump-shaped profile: grows until peak age, then declines.
    """
    working_years = params.retirement_age - params.start_age
    ages = np.arange(params.start_age, params.retirement_age)
    earnings = np.zeros(working_years)

    for i, age in enumerate(ages):
        if age <= params.earnings_hump_age:
            # Growth phase
            years_from_start = age - params.start_age
            earnings[i] = params.initial_earnings * (1 + params.earnings_growth) ** years_from_start
        else:
            # Decline phase
            years_from_peak = age - params.earnings_hump_age
            peak_earnings = params.initial_earnings * (1 + params.earnings_growth) ** (params.earnings_hump_age - params.start_age)
            earnings[i] = peak_earnings * (1 - params.earnings_decline) ** years_from_peak

    return earnings


def compute_expense_profile(params: LifecycleParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute expense profile over entire lifecycle.

    Returns:
        Tuple of (working_expenses, retirement_expenses)
    """
    # Working years
    working_years = params.retirement_age - params.start_age
    working_ages = np.arange(params.start_age, params.retirement_age)
    working_expenses = np.array([
        params.base_expenses * (1 + params.expense_growth) ** (age - params.start_age)
        for age in working_ages
    ])

    # Retirement years
    retirement_years = params.end_age - params.retirement_age
    retirement_expenses = np.full(retirement_years, params.retirement_expenses)

    return working_expenses, retirement_expenses


def compute_present_value(
    cashflows: np.ndarray,
    rate: float,
    phi: float = 0.85,
    r_bar: float = None
) -> float:
    """
    Compute present value of cashflow stream.

    If r_bar and phi provided, uses mean-reverting term structure.
    Otherwise uses flat discount rate.
    """
    if r_bar is not None:
        pv = 0.0
        for t, cf in enumerate(cashflows, 1):
            pv += cf * zero_coupon_price(rate, t, r_bar, phi)
        return pv
    else:
        pv = 0.0
        for t, cf in enumerate(cashflows, 1):
            pv += cf / (1 + rate) ** t
        return pv


def compute_duration(
    cashflows: np.ndarray,
    rate: float,
    phi: float = 0.85,
    r_bar: float = None
) -> float:
    """
    Compute effective duration of cashflow stream.

    Under mean reversion, duration is the PV-weighted average of effective durations.
    """
    if len(cashflows) == 0:
        return 0.0

    pv = compute_present_value(cashflows, rate, phi, r_bar)
    if pv < 1e-10:
        return 0.0

    if r_bar is not None:
        weighted_sum = 0.0
        for t, cf in enumerate(cashflows, 1):
            P_t = zero_coupon_price(rate, t, r_bar, phi)
            B_t = effective_duration(t, phi)
            weighted_sum += cf * P_t * B_t
        return weighted_sum / pv
    else:
        weighted_sum = 0.0
        for t, cf in enumerate(cashflows, 1):
            weighted_sum += t * cf / (1 + rate) ** t
        return weighted_sum / pv


def compute_lifecycle_median_path(
    params: LifecycleParams = None,
    econ_params: EconomicParams = None
) -> LifecycleResult:
    """
    Compute lifecycle investment strategy along the median (deterministic) path.

    This computes:
    - Earnings and expense profiles
    - Human capital (PV of future earnings minus expenses during working years)
    - Portfolio decomposition of human capital
    - Optimal financial portfolio to achieve target total allocation
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    r = econ_params.r_bar  # Use long-run rate for median path
    phi = econ_params.phi

    total_years = params.end_age - params.start_age
    ages = np.arange(params.start_age, params.end_age)

    # Initialize arrays
    earnings = np.zeros(total_years)
    expenses = np.zeros(total_years)

    # Fill earnings (only during working years)
    working_years = params.retirement_age - params.start_age
    earnings_profile = compute_earnings_profile(params)
    earnings[:working_years] = earnings_profile

    # Fill expenses (working + retirement)
    working_exp, retirement_exp = compute_expense_profile(params)
    earnings[:working_years] = earnings_profile
    expenses[:working_years] = working_exp
    expenses[working_years:] = retirement_exp

    # Base savings (earnings - subsistence expenses)
    # Note: Actual savings will be computed below after consumption model
    base_savings = earnings - expenses

    # Forward-looking present values at each age
    pv_earnings = np.zeros(total_years)
    pv_expenses = np.zeros(total_years)
    duration_earnings = np.zeros(total_years)
    duration_expenses = np.zeros(total_years)

    for i in range(total_years):
        # Remaining earnings (only during working years)
        if i < working_years:
            remaining_earnings = earnings[i:]
            remaining_earnings = remaining_earnings[:working_years - i]  # Cap at retirement
        else:
            remaining_earnings = np.array([])

        # Remaining expenses (always have some until end)
        remaining_expenses = expenses[i:]

        pv_earnings[i] = compute_present_value(remaining_earnings, r, phi, r)
        pv_expenses[i] = compute_present_value(remaining_expenses, r, phi, r)
        duration_earnings[i] = compute_duration(remaining_earnings, r, phi, r)
        duration_expenses[i] = compute_duration(remaining_expenses, r, phi, r)

    # Human capital = PV(earnings) - PV(expenses during working years only for net HC)
    # For the image style, human capital = PV of future earnings
    human_capital = pv_earnings.copy()

    # Decompose human capital into stock/bond/cash components
    # Stock component: based on stock beta (correlation with equity market)
    # Fixed income (bond/cash): based on duration of human capital
    hc_stock_component = human_capital * params.stock_beta_human_capital

    # Non-stock portion of human capital
    non_stock_hc = human_capital * (1.0 - params.stock_beta_human_capital)

    # Allocate non-stock portion between bonds and cash based on duration
    # Higher duration = more bond-like (interest rate sensitive)
    # Lower duration (near retirement) = more cash-like
    hc_bond_component = np.zeros(total_years)
    hc_cash_component = np.zeros(total_years)

    for i in range(total_years):
        if params.bond_duration_benchmark > 0 and non_stock_hc[i] > 0:
            # Bond fraction based on duration ratio (capped at 1.0)
            bond_fraction = min(1.0, duration_earnings[i] / params.bond_duration_benchmark)
            hc_bond_component[i] = non_stock_hc[i] * bond_fraction
            hc_cash_component[i] = non_stock_hc[i] * (1.0 - bond_fraction)
        else:
            # If no benchmark duration or no HC, treat as cash
            hc_cash_component[i] = non_stock_hc[i]

    # Financial wealth accumulation along median path with consumption model
    financial_wealth = np.zeros(total_years)
    financial_wealth[0] = params.initial_wealth

    # Consumption model arrays
    net_worth = np.zeros(total_years)
    subsistence_consumption = expenses.copy()  # Baseline consumption = subsistence expenses
    variable_consumption = np.zeros(total_years)
    total_consumption = np.zeros(total_years)
    savings = np.zeros(total_years)

    # Expected return on financial portfolio (assume target allocation)
    expected_stock_return = r + econ_params.mu_excess
    expected_bond_return = r
    expected_cash_return = r

    avg_return = (
        params.target_stock_allocation * expected_stock_return +
        params.target_bond_allocation * expected_bond_return +
        (1 - params.target_stock_allocation - params.target_bond_allocation) * expected_cash_return
    )

    # Consumption rate = median return + 1 percentage point
    consumption_rate = avg_return + 0.01

    # Simulate wealth accumulation with consumption model
    # Consumption = subsistence + share × net_worth
    # Net worth = HC + FW - PV(future expenses)
    # During working years: cap total consumption at earnings (no borrowing against HC)
    for i in range(total_years):
        # Compute net worth at start of period
        net_worth[i] = human_capital[i] + financial_wealth[i] - pv_expenses[i]

        # Variable consumption = share of net worth (floor at 0 if net worth negative)
        variable_consumption[i] = max(0, consumption_rate * net_worth[i])

        # Total consumption = subsistence + variable
        total_consumption[i] = subsistence_consumption[i] + variable_consumption[i]

        # During working years, cap consumption at earnings (can't borrow against HC)
        if earnings[i] > 0:
            if total_consumption[i] > earnings[i]:
                # Cap at earnings, reduce variable consumption accordingly
                total_consumption[i] = earnings[i]
                variable_consumption[i] = max(0, earnings[i] - subsistence_consumption[i])

        # Actual savings = earnings - total consumption
        savings[i] = earnings[i] - total_consumption[i]

        # Accumulate wealth for next period
        if i < total_years - 1:
            financial_wealth[i+1] = financial_wealth[i] * (1 + avg_return) + savings[i]
            financial_wealth[i+1] = max(0, financial_wealth[i+1])  # Floor at zero

    # Total wealth = financial wealth + human capital
    total_wealth = financial_wealth + human_capital

    # Target total portfolio allocation
    target_total_stocks = params.target_stock_allocation * total_wealth
    target_total_bonds = params.target_bond_allocation * total_wealth
    target_total_cash = (1 - params.target_stock_allocation - params.target_bond_allocation) * total_wealth

    # Target financial holdings = Total target - Human capital component
    target_fin_stocks = target_total_stocks - hc_stock_component
    target_fin_bonds = target_total_bonds - hc_bond_component
    target_fin_cash = target_total_cash - hc_cash_component

    # Constrained weights (no short selling in financial portfolio)
    # Apply no-short at aggregate level (equity vs fixed income), then split FI
    stock_weight_no_short = np.zeros(total_years)
    bond_weight_no_short = np.zeros(total_years)
    cash_weight_no_short = np.zeros(total_years)

    for i in range(total_years):
        fw = financial_wealth[i]
        if fw > 1e-6:
            # Compute target weights (can be negative or > 1)
            w_stock = target_fin_stocks[i] / fw
            w_bond = target_fin_bonds[i] / fw
            w_cash = target_fin_cash[i] / fw

            # Step 1: Aggregate into equity and fixed income
            equity = w_stock
            fixed_income = w_bond + w_cash

            # Step 2: No-short at aggregate level
            equity = max(0, equity)
            fixed_income = max(0, fixed_income)

            # Step 3: Normalize
            total_agg = equity + fixed_income
            if total_agg > 0:
                equity = equity / total_agg
                fixed_income = fixed_income / total_agg
            else:
                # Fallback to target allocation if both are non-positive
                equity = params.target_stock_allocation
                fixed_income = params.target_bond_allocation + (1.0 - params.target_stock_allocation - params.target_bond_allocation)

            # Step 4: Split FI among positive targets
            if w_bond > 0 and w_cash > 0:
                # Proportional split
                fi_total = w_bond + w_cash
                w_b = fixed_income * (w_bond / fi_total)
                w_c = fixed_income * (w_cash / fi_total)
            elif w_bond > 0:
                # All FI → bonds
                w_b = fixed_income
                w_c = 0.0
            elif w_cash > 0:
                # All FI → cash
                w_b = 0.0
                w_c = fixed_income
            else:
                # Both non-positive: use target proportions for split
                target_fi = params.target_bond_allocation + (1.0 - params.target_stock_allocation - params.target_bond_allocation)
                if target_fi > 0:
                    w_b = fixed_income * (params.target_bond_allocation / target_fi)
                    w_c = fixed_income * ((1.0 - params.target_stock_allocation - params.target_bond_allocation) / target_fi)
                else:
                    w_b = fixed_income / 2.0
                    w_c = fixed_income / 2.0

            stock_weight_no_short[i] = equity
            bond_weight_no_short[i] = w_b
            cash_weight_no_short[i] = w_c

    # Actual total holdings given financial portfolio
    total_stocks = stock_weight_no_short * financial_wealth + hc_stock_component
    total_bonds = bond_weight_no_short * financial_wealth + hc_bond_component
    total_cash = cash_weight_no_short * financial_wealth + hc_cash_component

    # Consumption as share of financial wealth
    consumption_share_of_fw = np.zeros(total_years)
    for i in range(total_years):
        if financial_wealth[i] > 1e-6:
            consumption_share_of_fw[i] = total_consumption[i] / financial_wealth[i]
        else:
            consumption_share_of_fw[i] = np.nan  # Undefined when no financial wealth

    return LifecycleResult(
        ages=ages,
        earnings=earnings,
        expenses=expenses,
        savings=savings,
        pv_earnings=pv_earnings,
        pv_expenses=pv_expenses,
        human_capital=human_capital,
        hc_stock_component=hc_stock_component,
        hc_bond_component=hc_bond_component,
        hc_cash_component=hc_cash_component,
        duration_earnings=duration_earnings,
        duration_expenses=duration_expenses,
        financial_wealth=financial_wealth,
        total_wealth=total_wealth,
        target_fin_stocks=target_fin_stocks,
        target_fin_bonds=target_fin_bonds,
        target_fin_cash=target_fin_cash,
        stock_weight_no_short=stock_weight_no_short,
        bond_weight_no_short=bond_weight_no_short,
        cash_weight_no_short=cash_weight_no_short,
        total_stocks=total_stocks,
        total_bonds=total_bonds,
        total_cash=total_cash,
        net_worth=net_worth,
        subsistence_consumption=subsistence_consumption,
        variable_consumption=variable_consumption,
        total_consumption=total_consumption,
        consumption_share_of_fw=consumption_share_of_fw,
    )


# =============================================================================
# Visualization Functions
# =============================================================================

# Color scheme matching the user's image
COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
}


def plot_earnings_expenses_profile(
    result: LifecycleResult,
    params: LifecycleParams,
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
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Profile of Earnings and Expenses')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-2, len(result.ages) + 2 if use_years else params.end_age + 2)


def plot_forward_present_values(
    result: LifecycleResult,
    params: LifecycleParams,
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
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Forward Looking Present Values')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-2, len(result.ages) + 2 if use_years else params.end_age + 2)


def plot_durations(
    result: LifecycleResult,
    params: LifecycleParams,
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
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Years')
    ax.set_title('Durations of Assets')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-2, len(result.ages) + 2 if use_years else params.end_age + 2)


def plot_human_vs_financial_wealth(
    result: LifecycleResult,
    params: LifecycleParams,
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
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Human Capital vs Financial Wealth')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-2, len(result.ages) + 2 if use_years else params.end_age + 2)


def plot_hc_decomposition(
    result: LifecycleResult,
    params: LifecycleParams,
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
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Portfolio Decomposition of Human Capital')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-2, len(result.ages) + 2 if use_years else params.end_age + 2)


def plot_target_financial_holdings(
    result: LifecycleResult,
    params: LifecycleParams,
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
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Target Financial Holdings')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-2, len(result.ages) + 2 if use_years else params.end_age + 2)


def plot_portfolio_shares(
    result: LifecycleResult,
    params: LifecycleParams,
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
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weight')
    ax.set_title('Target Financial Portfolio Shares')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-2, len(result.ages) + 2 if use_years else params.end_age + 2)
    ax.set_ylim(-0.05, 1.15)


def plot_total_wealth_holdings(
    result: LifecycleResult,
    params: LifecycleParams,
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
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Target Total Wealth Holdings')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-2, len(result.ages) + 2 if use_years else params.end_age + 2)


def plot_consumption_dollars(
    result: LifecycleResult,
    params: LifecycleParams,
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
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Consumption')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-2, len(result.ages) + 2 if use_years else params.end_age + 2)
    # Set y-axis to show data with some padding
    y_max = max(result.total_consumption) * 1.1
    ax.set_ylim(0, y_max)


def plot_consumption_breakdown(
    result: LifecycleResult,
    params: LifecycleParams,
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
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Consumption Breakdown')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-2, len(result.ages) + 2 if use_years else params.end_age + 2)
    # Set y-axis to show all data series with padding
    y_max = max(max(result.total_consumption), max(result.earnings)) * 1.1
    ax.set_ylim(0, y_max)


def create_beta_comparison_figure(
    beta_values: list = [0.0, 0.5, 1.0],
    base_params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    figsize: Tuple[int, int] = (16, 12),
    use_years: bool = True
) -> plt.Figure:
    """
    Create a figure comparing key metrics across different stock beta values.

    Shows how portfolio allocation, human capital decomposition, and target
    holdings change as stock beta varies from 0 to 1.
    """
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
            bond_duration_benchmark=base_params.bond_duration_benchmark,
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

    plt.tight_layout()
    return fig


def create_lifecycle_figure(
    result: LifecycleResult,
    params: LifecycleParams,
    figsize: Tuple[int, int] = (20, 10),
    use_years: bool = True
) -> plt.Figure:
    """
    Create a single figure with lifecycle strategy charts.

    Layout: 2 rows x 4 columns
    (Portfolio shares and HC decomposition are shown in the beta comparison page)

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


def generate_lifecycle_pdf(
    output_path: str = 'lifecycle_strategy.pdf',
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    include_individual_pages: bool = True,
    include_beta_comparison: bool = True,
    use_years: bool = True
) -> str:
    """
    Generate a PDF report showing lifecycle investment strategy.

    Args:
        output_path: Path for output PDF file
        params: Lifecycle parameters (uses defaults if None)
        econ_params: Economic parameters (uses defaults if None)
        include_individual_pages: If True, also include each chart on its own page
        include_beta_comparison: If True, include beta comparison visualization
        use_years: If True, x-axis shows years from career start; if False, shows age

    Returns:
        Path to generated PDF file
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    with PdfPages(output_path) as pdf:
        # Beta comparison page (if enabled)
        if include_beta_comparison:
            fig = create_beta_comparison_figure(
                beta_values=[0.0, 0.5, 1.0],
                base_params=params,
                econ_params=econ_params,
                figsize=(16, 10),
                use_years=use_years
            )
            fig.suptitle('Effect of Stock Beta on Portfolio Allocation',
                        fontsize=14, fontweight='bold', y=1.02)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        if include_individual_pages:
            # Beta-separated sections: all charts for each beta value
            beta_values = [0.0, 0.5, 1.0]

            for beta in beta_values:
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
                    bond_duration_benchmark=params.bond_duration_benchmark,
                    target_stock_allocation=params.target_stock_allocation,
                    target_bond_allocation=params.target_bond_allocation,
                    risk_free_rate=params.risk_free_rate,
                    equity_premium=params.equity_premium,
                    initial_wealth=params.initial_wealth,
                )
                beta_result = compute_lifecycle_median_path(beta_params, econ_params)

                # Page with all charts for this beta
                fig = create_lifecycle_figure(beta_result, beta_params, figsize=(20, 10), use_years=use_years)
                fig.suptitle(f'Lifecycle Investment Strategy - Beta = {beta}',
                            fontsize=14, fontweight='bold', y=1.02)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        # Summary page with parameters
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')

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

Consumption Model:
  - Total Consumption = Subsistence + Rate x Net Worth
  - Net Worth = Human Capital + Financial Wealth - PV(Future Expenses)
  - Consumption Rate = Median Return + 1pp

Human Capital Allocation:
  - Stock Beta: {params.stock_beta_human_capital:.2f}
  - Bond Duration Benchmark: {params.bond_duration_benchmark:.1f} years

Target Total Wealth Allocation:
  - Stocks: {params.target_stock_allocation*100:.0f}%
  - Bonds: {params.target_bond_allocation*100:.0f}%
  - Cash: {(1-params.target_stock_allocation-params.target_bond_allocation)*100:.0f}%

Economic Parameters:
  - Risk-Free Rate: {econ_params.r_bar*100:.1f}%
  - Equity Risk Premium: {econ_params.mu_excess*100:.1f}%
  - Rate Persistence (phi): {econ_params.phi:.2f}

Key Insights:
-------------
1. Consumption = Subsistence + (Median Return + 1pp) of Net Worth.

2. Net Worth accounts for human capital and future expense
   liabilities, not just financial wealth.

3. As net worth grows, variable consumption increases,
   allowing higher spending while preserving subsistence.

4. The "Consumption / FW" chart shows what share of
   financial wealth is spent each year.
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
    initial_earnings: float = 100,
    stock_beta_hc: float = 0.1,
    bond_duration: float = 7.0,
    target_stocks: float = 0.60,
    target_bonds: float = 0.30,
    consumption_share: float = 0.05,
    use_years: bool = True,
    verbose: bool = True
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
        bond_duration: Benchmark bond duration for HC allocation (years)
        target_stocks: Target stock allocation in total wealth
        target_bonds: Target bond allocation in total wealth
        consumption_share: Share of net worth consumed above subsistence
        use_years: If True, x-axis shows years from start; if False, shows age
        verbose: If True, print progress and statistics
    """
    if verbose:
        print("Computing lifecycle investment strategy...")

    params = LifecycleParams(
        start_age=start_age,
        retirement_age=retirement_age,
        end_age=end_age,
        initial_earnings=initial_earnings,
        stock_beta_human_capital=stock_beta_hc,
        bond_duration_benchmark=bond_duration,
        target_stock_allocation=target_stocks,
        target_bond_allocation=target_bonds,
        consumption_share=consumption_share,
    )
    econ_params = EconomicParams()

    output = generate_lifecycle_pdf(
        output_path=output_path,
        params=params,
        econ_params=econ_params,
        use_years=use_years,
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
    parser.add_argument('--initial-earnings', type=float, default=100,
                       help='Initial earnings in $000s (default: 100)')
    parser.add_argument('--stock-beta', type=float, default=0.1,
                       help='Stock beta of human capital (default: 0.1)')
    parser.add_argument('--bond-duration', type=float, default=20.0,
                       help='Benchmark bond duration for HC allocation in years (default: 20.0)')
    parser.add_argument('--target-stocks', type=float, default=0.60,
                       help='Target stock allocation (default: 0.60)')
    parser.add_argument('--target-bonds', type=float, default=0.30,
                       help='Target bond allocation (default: 0.30)')
    parser.add_argument('--consumption-share', type=float, default=0.05,
                       help='Share of net worth consumed above subsistence (default: 0.05)')
    parser.add_argument('--use-age', action='store_true',
                       help='Use age instead of years from start on x-axis')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress output messages')

    args = parser.parse_args()

    main(
        output_path=args.output,
        start_age=args.start_age,
        retirement_age=args.retirement_age,
        end_age=args.end_age,
        initial_earnings=args.initial_earnings,
        stock_beta_hc=args.stock_beta,
        bond_duration=args.bond_duration,
        target_stocks=args.target_stocks,
        target_bonds=args.target_bonds,
        consumption_share=args.consumption_share,
        use_years=not args.use_age,
        verbose=not args.quiet,
    )
