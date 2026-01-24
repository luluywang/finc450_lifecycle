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
from typing import Tuple, Optional, List
from retirement_simulation import (
    EconomicParams,
    effective_duration,
    zero_coupon_price,
    compute_full_merton_allocation_constrained,
    generate_correlated_shocks,
    simulate_interest_rates,
    simulate_stock_returns,
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
    consumption_boost: float = 0.01  # Boost above median return for consumption rate

    # Asset allocation parameters
    stock_beta_human_capital: float = 0.1    # Beta of human capital to stocks

    # Mean-variance optimization parameters
    # If gamma > 0, target allocations are derived from MV optimization
    # If gamma = 0, use the fixed target allocations below
    gamma: float = 2.0               # Risk aversion coefficient for MV optimization
    target_stock_allocation: float = 0.60    # Target stock allocation (used if gamma=0)
    target_bond_allocation: float = 0.30     # Target bond allocation (used if gamma=0)

    # Economic parameters (consistent with EconomicParams for DGP)
    risk_free_rate: float = 0.02     # Long-run real risk-free rate
    equity_premium: float = 0.04     # Equity risk premium

    # Initial financial wealth (can be negative for student loans)
    initial_wealth: float = 1        # Starting financial wealth ($1k, negative allowed)


@dataclass
class MonteCarloParams:
    """Parameters for Monte Carlo simulation."""
    n_simulations: int = 1000    # Number of simulation paths
    random_seed: int = 42        # Random seed for reproducibility


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation of lifecycle paths."""
    ages: np.ndarray                      # Shape: (total_years,)

    # Wealth paths: (n_sims, total_years)
    financial_wealth_paths: np.ndarray
    total_wealth_paths: np.ndarray
    human_capital_paths: np.ndarray

    # Consumption paths: (n_sims, total_years)
    total_consumption_paths: np.ndarray
    subsistence_consumption_paths: np.ndarray
    variable_consumption_paths: np.ndarray

    # Portfolio allocation paths: (n_sims, total_years)
    stock_weight_paths: np.ndarray
    bond_weight_paths: np.ndarray
    cash_weight_paths: np.ndarray

    # Return paths: (n_sims, total_years)
    stock_return_paths: np.ndarray
    interest_rate_paths: np.ndarray

    # Outcome statistics
    default_flags: np.ndarray             # Shape: (n_sims,) - True if defaulted
    default_ages: np.ndarray              # Shape: (n_sims,) - Age at default (NaN if no default)
    final_wealth: np.ndarray              # Shape: (n_sims,) - Terminal financial wealth
    total_lifetime_consumption: np.ndarray  # Shape: (n_sims,)

    # Median path for reference
    median_result: 'LifecycleResult'

    # Target allocations from MV optimization
    target_stock: float
    target_bond: float
    target_cash: float


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

    # Expense liability decomposition (mirrors HC decomposition)
    exp_bond_component: np.ndarray     # Bond-like portion of expense liability
    exp_cash_component: np.ndarray     # Cash-like portion of expense liability

    # Consumption model
    net_worth: np.ndarray              # HC + FW - PV(future expenses)
    subsistence_consumption: np.ndarray # Baseline/floor consumption
    variable_consumption: np.ndarray    # Share of net worth consumed
    total_consumption: np.ndarray       # Subsistence + variable
    consumption_share_of_fw: np.ndarray # Total consumption / financial wealth


@dataclass
class RuleOfThumbResult:
    """Results from rule-of-thumb strategy simulation."""
    ages: np.ndarray
    financial_wealth: np.ndarray
    earnings: np.ndarray
    total_consumption: np.ndarray
    retirement_consumption_fixed: float  # 4% of retirement wealth
    stock_weight: np.ndarray
    bond_weight: np.ndarray
    cash_weight: np.ndarray
    defaulted: bool
    default_age: Optional[int]
    savings_rate: float  # 0.20
    withdrawal_rate: float  # 0.04


@dataclass
class StrategyComparisonResult:
    """Results from comparing optimal vs rule-of-thumb strategies."""
    n_simulations: int
    ages: np.ndarray
    # Optimal strategy paths
    optimal_wealth_paths: np.ndarray
    optimal_consumption_paths: np.ndarray
    optimal_default_flags: np.ndarray
    optimal_default_ages: np.ndarray
    # Rule-of-thumb paths
    rot_wealth_paths: np.ndarray
    rot_consumption_paths: np.ndarray
    rot_default_flags: np.ndarray
    rot_default_ages: np.ndarray
    # Sample allocation path (from first simulation)
    rot_stock_weight_sample: np.ndarray
    rot_bond_weight_sample: np.ndarray
    rot_cash_weight_sample: np.ndarray
    # Percentile statistics
    percentiles: List[int]
    optimal_wealth_percentiles: np.ndarray
    rot_wealth_percentiles: np.ndarray
    optimal_consumption_percentiles: np.ndarray
    rot_consumption_percentiles: np.ndarray
    # Summary stats
    optimal_default_rate: float
    rot_default_rate: float
    optimal_median_final_wealth: float
    rot_median_final_wealth: float
    # PV consumption stats (at time 0)
    optimal_pv_consumption: np.ndarray  # Shape: (n_sims,) - PV consumption for each sim
    rot_pv_consumption: np.ndarray      # Shape: (n_sims,)
    optimal_pv_consumption_percentiles: np.ndarray  # Shape: (n_percentiles,)
    rot_pv_consumption_percentiles: np.ndarray
    # Market condition paths (for visualization)
    stock_return_paths: np.ndarray      # Shape: (n_sims, total_years)
    interest_rate_paths: np.ndarray     # Shape: (n_sims, total_years)


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


def compute_mv_optimal_allocation(
    mu_stock: float,
    mu_bond: float,
    sigma_s: float,
    sigma_r: float,
    rho: float,
    duration: float,
    gamma: float
) -> Tuple[float, float, float]:
    """
    Compute optimal portfolio allocation using full Merton solution with VCV matrix.

    This implements the multi-asset Merton solution:
        w* = (1/gamma) * Sigma^(-1) * mu

    Asset return models:
    - Stock: R_s = r + mu_stock + sigma_s * eps_s
    - Bond:  R_b = r + mu_bond - D * sigma_r * eps_r
      (negative sign because rising rates hurt bond prices)

    The VCV matrix is derived from:
    - sigma_s: Stock return volatility
    - sigma_r: Interest rate shock volatility
    - rho: Correlation between rate shocks and stock shocks
    - duration: Effective duration of bond portfolio

    This gives:
    - Var(R_s) = sigma_s^2
    - Var(R_b) = (D * sigma_r)^2
    - Cov(R_s, R_b) = -D * sigma_s * sigma_r * rho

    Args:
        mu_stock: Stock excess return (equity risk premium)
        mu_bond: Bond excess return (risk premium over short rate)
        sigma_s: Standard deviation of stock returns
        sigma_r: Interest rate shock volatility
        rho: Correlation between rate and stock shocks
        duration: Effective duration of bond portfolio
        gamma: Risk aversion coefficient (higher = more conservative)

    Returns:
        Tuple of (stock_weight, bond_weight, cash_weight) summing to 1.0
    """
    return compute_full_merton_allocation_constrained(
        mu_stock=mu_stock,
        mu_bond=mu_bond,
        sigma_s=sigma_s,
        sigma_r=sigma_r,
        rho=rho,
        duration=duration,
        gamma=gamma
    )


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


def compute_pv_consumption(consumption: np.ndarray, rate: float) -> float:
    """
    Compute Present Value of consumption at time 0.

    This discounts all future consumption back to the starting age,
    providing a single metric for lifetime consumption in present value terms.

    Args:
        consumption: Array of consumption values over time
        rate: Discount rate (typically the risk-free rate)

    Returns:
        Present value of total lifetime consumption at time 0
    """
    pv = 0.0
    for t, c in enumerate(consumption):
        pv += c / (1 + rate) ** t
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

    # Compute target allocations: either from MV optimization or use fixed targets
    if params.gamma > 0:
        # Mean-variance optimal allocation using full VCV matrix
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
        # Use fixed target allocations
        target_stock = params.target_stock_allocation
        target_bond = params.target_bond_allocation
        target_cash = 1.0 - target_stock - target_bond

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
        if econ_params.bond_duration > 0 and non_stock_hc[i] > 0:
            # Bond fraction based on duration ratio (capped at 1.0)
            bond_fraction = min(1.0, duration_earnings[i] / econ_params.bond_duration)
            hc_bond_component[i] = non_stock_hc[i] * bond_fraction
            hc_cash_component[i] = non_stock_hc[i] * (1.0 - bond_fraction)
        else:
            # If no benchmark duration or no HC, treat as cash
            hc_cash_component[i] = non_stock_hc[i]

    # Decompose expense liability into bond/cash components (similar to HC decomposition)
    # Based on duration of expense stream - longer duration = more bond-like
    exp_bond_component = np.zeros(total_years)
    exp_cash_component = np.zeros(total_years)

    for i in range(total_years):
        if econ_params.bond_duration > 0 and pv_expenses[i] > 0:
            # Bond fraction based on expense duration ratio (capped at 1.0)
            bond_fraction = min(1.0, duration_expenses[i] / econ_params.bond_duration)
            exp_bond_component[i] = pv_expenses[i] * bond_fraction
            exp_cash_component[i] = pv_expenses[i] * (1.0 - bond_fraction)
        else:
            # If no benchmark duration, treat as cash liability
            exp_cash_component[i] = pv_expenses[i]

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
        target_stock * expected_stock_return +
        target_bond * expected_bond_return +
        target_cash * expected_cash_return
    )

    # Consumption rate = median return + boost (default 1 percentage point)
    consumption_rate = avg_return + params.consumption_boost

    # Simulate wealth accumulation with consumption model
    # Consumption = subsistence + share × net_worth
    # Net worth = HC + FW - PV(future expenses)
    # During working years: cap total consumption at earnings (no borrowing against HC)
    # During retirement: cap consumption at financial wealth (can't consume more than you have)
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
        else:
            # Retirement: cap consumption at financial wealth
            fw = financial_wealth[i]
            if subsistence_consumption[i] > fw:
                # Bankruptcy: can't even meet subsistence, consume whatever remains
                total_consumption[i] = fw
                subsistence_consumption[i] = fw
                variable_consumption[i] = 0
            elif total_consumption[i] > fw:
                # Can meet subsistence but not variable consumption
                total_consumption[i] = fw
                variable_consumption[i] = fw - subsistence_consumption[i]

        # Actual savings = earnings - total consumption
        savings[i] = earnings[i] - total_consumption[i]

        # Accumulate wealth for next period (allow negative for student loans)
        if i < total_years - 1:
            financial_wealth[i+1] = financial_wealth[i] * (1 + avg_return) + savings[i]
            # Note: negative wealth is allowed to model student loan debt

    # Total wealth = financial wealth + human capital
    total_wealth = financial_wealth + human_capital

    # Target total portfolio allocation (from MV optimization or fixed targets)
    target_total_stocks = target_stock * total_wealth
    target_total_bonds = target_bond * total_wealth
    target_total_cash = target_cash * total_wealth

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
                equity = target_stock
                fixed_income = target_bond + target_cash

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
                target_fi = target_bond + target_cash
                if target_fi > 0:
                    w_b = fixed_income * (target_bond / target_fi)
                    w_c = fixed_income * (target_cash / target_fi)
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
        exp_bond_component=exp_bond_component,
        exp_cash_component=exp_cash_component,
        net_worth=net_worth,
        subsistence_consumption=subsistence_consumption,
        variable_consumption=variable_consumption,
        total_consumption=total_consumption,
        consumption_share_of_fw=consumption_share_of_fw,
    )


def compute_lifecycle_fixed_consumption(
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    withdrawal_rate: float = 0.04
) -> LifecycleResult:
    """
    Compute lifecycle path using a FIXED consumption rule (4% rule style).

    Unlike the optimal variable consumption model, this uses:
    - During working years: consume subsistence expenses only (save the rest)
    - During retirement: consume a fixed percentage of retirement wealth each year

    This is the classic "4% rule" approach that can lead to default if returns are poor.

    Args:
        params: Lifecycle parameters
        econ_params: Economic parameters
        withdrawal_rate: Fixed withdrawal rate in retirement (default 4%)

    Returns:
        LifecycleResult with consumption and wealth trajectories
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    r = econ_params.r_bar
    phi = econ_params.phi

    # Compute target allocations from MV optimization
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

    total_years = params.end_age - params.start_age
    ages = np.arange(params.start_age, params.end_age)
    working_years = params.retirement_age - params.start_age

    # Initialize arrays
    earnings = np.zeros(total_years)
    expenses = np.zeros(total_years)

    # Fill earnings and expenses
    earnings_profile = compute_earnings_profile(params)
    working_exp, retirement_exp = compute_expense_profile(params)
    earnings[:working_years] = earnings_profile
    expenses[:working_years] = working_exp
    expenses[working_years:] = retirement_exp

    # Forward-looking present values (same as optimal)
    pv_earnings = np.zeros(total_years)
    pv_expenses = np.zeros(total_years)
    duration_earnings = np.zeros(total_years)
    duration_expenses = np.zeros(total_years)

    for i in range(total_years):
        if i < working_years:
            remaining_earnings = earnings[i:working_years]
        else:
            remaining_earnings = np.array([])
        remaining_expenses = expenses[i:]

        pv_earnings[i] = compute_present_value(remaining_earnings, r, phi, r)
        pv_expenses[i] = compute_present_value(remaining_expenses, r, phi, r)
        duration_earnings[i] = compute_duration(remaining_earnings, r, phi, r)
        duration_expenses[i] = compute_duration(remaining_expenses, r, phi, r)

    # Human capital
    human_capital = pv_earnings.copy()

    # HC decomposition
    hc_stock_component = human_capital * params.stock_beta_human_capital
    non_stock_hc = human_capital * (1.0 - params.stock_beta_human_capital)
    hc_bond_component = np.zeros(total_years)
    hc_cash_component = np.zeros(total_years)

    for i in range(total_years):
        if econ_params.bond_duration > 0 and non_stock_hc[i] > 0:
            bond_fraction = min(1.0, duration_earnings[i] / econ_params.bond_duration)
            hc_bond_component[i] = non_stock_hc[i] * bond_fraction
            hc_cash_component[i] = non_stock_hc[i] * (1.0 - bond_fraction)
        else:
            hc_cash_component[i] = non_stock_hc[i]

    # Decompose expense liability into bond/cash components
    exp_bond_component = np.zeros(total_years)
    exp_cash_component = np.zeros(total_years)

    for i in range(total_years):
        if econ_params.bond_duration > 0 and pv_expenses[i] > 0:
            bond_fraction = min(1.0, duration_expenses[i] / econ_params.bond_duration)
            exp_bond_component[i] = pv_expenses[i] * bond_fraction
            exp_cash_component[i] = pv_expenses[i] * (1.0 - bond_fraction)
        else:
            exp_cash_component[i] = pv_expenses[i]

    # Financial wealth accumulation with FIXED consumption rule
    financial_wealth = np.zeros(total_years)
    financial_wealth[0] = params.initial_wealth

    net_worth = np.zeros(total_years)
    subsistence_consumption = np.zeros(total_years)
    variable_consumption = np.zeros(total_years)
    total_consumption = np.zeros(total_years)
    savings = np.zeros(total_years)

    # Expected portfolio return
    expected_stock_return = r + econ_params.mu_excess
    avg_return = (target_stock * expected_stock_return +
                  target_bond * r + target_cash * r)

    # Fixed retirement consumption = withdrawal_rate × wealth at retirement
    # We'll compute this once we know retirement wealth
    fixed_retirement_consumption = None
    defaulted = False
    default_year = None

    for i in range(total_years):
        net_worth[i] = human_capital[i] + financial_wealth[i] - pv_expenses[i]

        if i < working_years:
            # Working years: consume only subsistence, save the rest
            subsistence_consumption[i] = expenses[i]
            variable_consumption[i] = 0
            total_consumption[i] = subsistence_consumption[i]
            savings[i] = earnings[i] - total_consumption[i]
        else:
            # Retirement: use fixed consumption rule
            if fixed_retirement_consumption is None:
                # First year of retirement: set fixed consumption
                fixed_retirement_consumption = withdrawal_rate * financial_wealth[i]

            if defaulted:
                # Already defaulted - no more consumption possible
                total_consumption[i] = 0
                subsistence_consumption[i] = 0
                variable_consumption[i] = 0
                savings[i] = 0
            elif financial_wealth[i] <= 0:
                # Default!
                defaulted = True
                default_year = i
                total_consumption[i] = 0
                subsistence_consumption[i] = 0
                variable_consumption[i] = 0
                savings[i] = 0
            elif financial_wealth[i] < fixed_retirement_consumption:
                # Can't meet fixed consumption - consume what's left
                total_consumption[i] = financial_wealth[i]
                subsistence_consumption[i] = min(expenses[i], financial_wealth[i])
                variable_consumption[i] = max(0, total_consumption[i] - subsistence_consumption[i])
                savings[i] = -total_consumption[i]  # Negative savings (drawdown)
            else:
                # Normal case: consume fixed amount
                total_consumption[i] = fixed_retirement_consumption
                subsistence_consumption[i] = min(expenses[i], total_consumption[i])
                variable_consumption[i] = max(0, total_consumption[i] - subsistence_consumption[i])
                savings[i] = -total_consumption[i]  # Negative savings (drawdown)

        # Accumulate wealth for next period
        if i < total_years - 1:
            financial_wealth[i+1] = financial_wealth[i] * (1 + avg_return) + savings[i]
            if financial_wealth[i+1] < 0:
                financial_wealth[i+1] = 0

    # Total wealth
    total_wealth = financial_wealth + human_capital

    # Target financial holdings and weights (same logic as optimal)
    target_fin_stocks = target_stock * total_wealth - hc_stock_component
    target_fin_bonds = target_bond * total_wealth - hc_bond_component
    target_fin_cash = target_cash * total_wealth - hc_cash_component

    stock_weight_no_short = np.zeros(total_years)
    bond_weight_no_short = np.zeros(total_years)
    cash_weight_no_short = np.zeros(total_years)

    for i in range(total_years):
        fw = financial_wealth[i]
        if fw > 1e-6:
            w_stock = target_fin_stocks[i] / fw
            w_bond = target_fin_bonds[i] / fw
            w_cash = target_fin_cash[i] / fw

            equity = max(0, w_stock)
            fixed_income = max(0, w_bond + w_cash)
            total_agg = equity + fixed_income
            if total_agg > 0:
                equity /= total_agg
                fixed_income /= total_agg
            else:
                equity = target_stock
                fixed_income = target_bond + target_cash

            if w_bond > 0 and w_cash > 0:
                fi_total = w_bond + w_cash
                bond_weight_no_short[i] = fixed_income * (w_bond / fi_total)
                cash_weight_no_short[i] = fixed_income * (w_cash / fi_total)
            elif w_bond > 0:
                bond_weight_no_short[i] = fixed_income
            elif w_cash > 0:
                cash_weight_no_short[i] = fixed_income
            else:
                target_fi = target_bond + target_cash
                if target_fi > 0:
                    bond_weight_no_short[i] = fixed_income * (target_bond / target_fi)
                    cash_weight_no_short[i] = fixed_income * (target_cash / target_fi)
                else:
                    bond_weight_no_short[i] = fixed_income / 2
                    cash_weight_no_short[i] = fixed_income / 2
            stock_weight_no_short[i] = equity

    # Total holdings
    total_stocks = stock_weight_no_short * financial_wealth + hc_stock_component
    total_bonds = bond_weight_no_short * financial_wealth + hc_bond_component
    total_cash = cash_weight_no_short * financial_wealth + hc_cash_component

    # Consumption share of FW
    consumption_share_of_fw = np.zeros(total_years)
    for i in range(total_years):
        if financial_wealth[i] > 1e-6:
            consumption_share_of_fw[i] = total_consumption[i] / financial_wealth[i]
        else:
            consumption_share_of_fw[i] = np.nan

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
        exp_bond_component=exp_bond_component,
        exp_cash_component=exp_cash_component,
        net_worth=net_worth,
        subsistence_consumption=subsistence_consumption,
        variable_consumption=variable_consumption,
        total_consumption=total_consumption,
        consumption_share_of_fw=consumption_share_of_fw,
    )


def compute_rule_of_thumb_strategy(
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    savings_rate: float = 0.20,
    withdrawal_rate: float = 0.04,
    stock_returns: np.ndarray = None,
    interest_rates: np.ndarray = None,
) -> RuleOfThumbResult:
    """
    Compute lifecycle path using the classic "rule of thumb" financial advisor strategy.

    This implements traditional heuristics:
    - During working years: Save 20% of income
    - Allocation: (100 - age)% in stocks, rest split 50/50 between bonds and cash
    - At retirement: Freeze allocation at retirement age
    - Retirement withdrawal: 4% of initial retirement wealth (fixed, not adjusted)

    This is NOT optimal but represents common retail advice.

    Args:
        params: Lifecycle parameters
        econ_params: Economic parameters
        savings_rate: Fraction of income to save during working years (default 20%)
        withdrawal_rate: Fixed withdrawal rate in retirement (default 4%)
        stock_returns: Optional array of stock returns for each year (uses expected if None)
        interest_rates: Optional array of interest rates for each year (uses r_bar if None)

    Returns:
        RuleOfThumbResult with wealth and consumption trajectories
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    total_years = params.end_age - params.start_age
    working_years = params.retirement_age - params.start_age
    ages = np.arange(params.start_age, params.end_age)

    # Get earnings profile
    earnings_profile = compute_earnings_profile(params)
    earnings = np.zeros(total_years)
    earnings[:working_years] = earnings_profile

    # Initialize arrays
    financial_wealth = np.zeros(total_years)
    financial_wealth[0] = params.initial_wealth
    total_consumption = np.zeros(total_years)
    stock_weight = np.zeros(total_years)
    bond_weight = np.zeros(total_years)
    cash_weight = np.zeros(total_years)

    # Use expected returns if not provided
    r = econ_params.r_bar
    if stock_returns is None:
        stock_returns = np.full(total_years, r + econ_params.mu_excess)
    if interest_rates is None:
        interest_rates = np.full(total_years, r)

    defaulted = False
    default_age = None
    fixed_retirement_consumption = None
    retirement_stock_weight = None
    retirement_bond_weight = None
    retirement_cash_weight = None

    for t in range(total_years):
        age = params.start_age + t
        fw = financial_wealth[t]

        # Compute allocation: (100 - age)% stocks, rest split 50/50 bonds/cash
        if t < working_years:
            # Working years: update allocation based on age
            stock_pct = max(0.0, min(1.0, (100 - age) / 100.0))
            fixed_income_pct = 1.0 - stock_pct
            bond_pct = fixed_income_pct * 0.5
            cash_pct = fixed_income_pct * 0.5
        else:
            # Retirement: freeze allocation at retirement age
            if retirement_stock_weight is None:
                retirement_age = params.retirement_age
                retirement_stock_weight = max(0.0, min(1.0, (100 - retirement_age) / 100.0))
                retirement_fixed_income = 1.0 - retirement_stock_weight
                retirement_bond_weight = retirement_fixed_income * 0.5
                retirement_cash_weight = retirement_fixed_income * 0.5
            stock_pct = retirement_stock_weight
            bond_pct = retirement_bond_weight
            cash_pct = retirement_cash_weight

        stock_weight[t] = stock_pct
        bond_weight[t] = bond_pct
        cash_weight[t] = cash_pct

        # Compute consumption
        if t < working_years:
            # Working years: consume (1 - savings_rate) of earnings
            total_consumption[t] = earnings[t] * (1.0 - savings_rate)
        else:
            # Retirement: fixed consumption from 4% rule
            if fixed_retirement_consumption is None:
                # First year of retirement: set fixed consumption
                fixed_retirement_consumption = withdrawal_rate * fw

            if defaulted:
                total_consumption[t] = 0
            elif fw <= 0:
                # Default!
                defaulted = True
                default_age = age
                total_consumption[t] = 0
            elif fw < fixed_retirement_consumption:
                # Can't meet fixed consumption - consume what's left
                total_consumption[t] = fw
            else:
                total_consumption[t] = fixed_retirement_consumption

        # Evolve wealth to next period
        if t < total_years - 1 and not defaulted:
            if t < working_years:
                savings = earnings[t] - total_consumption[t]
            else:
                savings = -total_consumption[t]

            # Portfolio return
            stock_ret = stock_returns[t]
            bond_ret = interest_rates[t] + econ_params.mu_bond
            cash_ret = interest_rates[t]

            portfolio_return = (
                stock_weight[t] * stock_ret +
                bond_weight[t] * bond_ret +
                cash_weight[t] * cash_ret
            )

            financial_wealth[t + 1] = fw * (1 + portfolio_return) + savings
            if financial_wealth[t + 1] < 0:
                financial_wealth[t + 1] = 0

    return RuleOfThumbResult(
        ages=ages,
        financial_wealth=financial_wealth,
        earnings=earnings,
        total_consumption=total_consumption,
        retirement_consumption_fixed=fixed_retirement_consumption if fixed_retirement_consumption else 0,
        stock_weight=stock_weight,
        bond_weight=bond_weight,
        cash_weight=cash_weight,
        defaulted=defaulted,
        default_age=default_age,
        savings_rate=savings_rate,
        withdrawal_rate=withdrawal_rate,
    )


# =============================================================================
# Monte Carlo Simulation
# =============================================================================

def run_lifecycle_monte_carlo(
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    mc_params: MonteCarloParams = None,
    initial_rate: float = None,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation of lifecycle investment strategy.

    Simulates many paths with stochastic stock returns and interest rates,
    tracking wealth accumulation, consumption, and portfolio allocation.

    Args:
        params: Lifecycle parameters
        econ_params: Economic parameters
        mc_params: Monte Carlo simulation parameters
        initial_rate: Initial interest rate (defaults to r_bar)

    Returns:
        MonteCarloResult with paths and statistics
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()
    if mc_params is None:
        mc_params = MonteCarloParams()
    if initial_rate is None:
        initial_rate = econ_params.r_bar

    # Get median path for reference
    median_result = compute_lifecycle_median_path(params, econ_params)

    # Compute target allocations from MV optimization
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

    # Simulation setup
    n_sims = mc_params.n_simulations
    total_years = params.end_age - params.start_age
    working_years = params.retirement_age - params.start_age
    rng = np.random.default_rng(mc_params.random_seed)

    # Generate correlated shocks for entire simulation
    rate_shocks, stock_shocks = generate_correlated_shocks(
        total_years, n_sims, econ_params.rho, rng
    )

    # Simulate interest rate paths
    rate_paths = simulate_interest_rates(
        initial_rate, total_years, n_sims, econ_params, rate_shocks
    )

    # Simulate stock return paths
    stock_return_paths = simulate_stock_returns(rate_paths, econ_params, stock_shocks)

    # Get earnings and expenses from median path (deterministic)
    earnings = median_result.earnings.copy()
    expenses = median_result.expenses.copy()

    # Initialize output arrays
    financial_wealth_paths = np.zeros((n_sims, total_years))
    total_wealth_paths = np.zeros((n_sims, total_years))
    human_capital_paths = np.zeros((n_sims, total_years))
    total_consumption_paths = np.zeros((n_sims, total_years))
    subsistence_consumption_paths = np.zeros((n_sims, total_years))
    variable_consumption_paths = np.zeros((n_sims, total_years))
    stock_weight_paths = np.zeros((n_sims, total_years))
    bond_weight_paths = np.zeros((n_sims, total_years))
    cash_weight_paths = np.zeros((n_sims, total_years))
    default_flags = np.zeros(n_sims, dtype=bool)
    default_ages = np.full(n_sims, np.nan)

    # Set initial conditions
    financial_wealth_paths[:, 0] = params.initial_wealth
    human_capital_paths[:, :] = median_result.human_capital[np.newaxis, :]

    # Consumption rate = median return + boost
    r = econ_params.r_bar
    expected_stock_return = r + econ_params.mu_excess
    avg_median_return = (
        target_stock * expected_stock_return +
        target_bond * r +
        target_cash * r
    )
    consumption_rate = avg_median_return + params.consumption_boost

    # Simulate each path
    for sim in range(n_sims):
        defaulted = False

        for t in range(total_years):
            fw = financial_wealth_paths[sim, t]
            hc = human_capital_paths[sim, t]
            pv_exp = median_result.pv_expenses[t]

            # Compute net worth
            net_worth = hc + fw - pv_exp

            # Compute consumption
            subsistence = expenses[t]
            variable = max(0, consumption_rate * net_worth)
            total_cons = subsistence + variable

            # Apply constraints
            if t < working_years:
                # Working: cap at earnings
                if total_cons > earnings[t]:
                    total_cons = earnings[t]
                    variable = max(0, earnings[t] - subsistence)
            else:
                # Retirement: cap at financial wealth
                if defaulted:
                    total_cons = 0
                    subsistence = 0
                    variable = 0
                elif fw <= 0:
                    defaulted = True
                    default_flags[sim] = True
                    default_ages[sim] = params.start_age + t
                    total_cons = 0
                    subsistence = 0
                    variable = 0
                elif subsistence > fw:
                    total_cons = fw
                    subsistence = fw
                    variable = 0
                elif total_cons > fw:
                    total_cons = fw
                    variable = fw - subsistence

            # Store consumption
            total_consumption_paths[sim, t] = total_cons
            subsistence_consumption_paths[sim, t] = subsistence
            variable_consumption_paths[sim, t] = variable

            # Compute portfolio weights
            total_wealth = fw + hc
            hc_stock = median_result.hc_stock_component[t]
            hc_bond = median_result.hc_bond_component[t]
            hc_cash = median_result.hc_cash_component[t]

            target_fin_stock = target_stock * total_wealth - hc_stock
            target_fin_bond = target_bond * total_wealth - hc_bond
            target_fin_cash = target_cash * total_wealth - hc_cash

            if fw > 1e-6:
                w_stock = target_fin_stock / fw
                w_bond = target_fin_bond / fw
                w_cash = target_fin_cash / fw

                # Apply no-short constraint at aggregate level
                equity = max(0, w_stock)
                fixed_income = max(0, w_bond + w_cash)
                total_agg = equity + fixed_income
                if total_agg > 0:
                    equity /= total_agg
                    fixed_income /= total_agg
                else:
                    equity = target_stock
                    fixed_income = target_bond + target_cash

                # Split FI between bonds and cash
                if w_bond > 0 and w_cash > 0:
                    fi_total = w_bond + w_cash
                    w_b = fixed_income * (w_bond / fi_total)
                    w_c = fixed_income * (w_cash / fi_total)
                elif w_bond > 0:
                    w_b = fixed_income
                    w_c = 0
                elif w_cash > 0:
                    w_b = 0
                    w_c = fixed_income
                else:
                    target_fi = target_bond + target_cash
                    if target_fi > 0:
                        w_b = fixed_income * (target_bond / target_fi)
                        w_c = fixed_income * (target_cash / target_fi)
                    else:
                        w_b = fixed_income / 2
                        w_c = fixed_income / 2

                stock_weight_paths[sim, t] = equity
                bond_weight_paths[sim, t] = w_b
                cash_weight_paths[sim, t] = w_c
            else:
                stock_weight_paths[sim, t] = target_stock
                bond_weight_paths[sim, t] = target_bond
                cash_weight_paths[sim, t] = target_cash

            # Update total wealth
            total_wealth_paths[sim, t] = fw + hc

            # Evolve wealth to next period
            if t < total_years - 1 and not defaulted:
                savings = earnings[t] - total_cons

                # Portfolio return
                w_s = stock_weight_paths[sim, t]
                w_b = bond_weight_paths[sim, t]
                w_c = cash_weight_paths[sim, t]

                # Stock return from simulation
                stock_ret = stock_return_paths[sim, t]

                # Bond return: r + mu_bond - D * sigma_r * eps
                # (already reflected in the rate path impact)
                bond_ret = rate_paths[sim, t] + econ_params.mu_bond

                # Cash return = current rate
                cash_ret = rate_paths[sim, t]

                portfolio_return = w_s * stock_ret + w_b * bond_ret + w_c * cash_ret

                financial_wealth_paths[sim, t + 1] = fw * (1 + portfolio_return) + savings

    # Compute statistics
    final_wealth = financial_wealth_paths[:, -1]
    total_lifetime_consumption = np.sum(total_consumption_paths, axis=1)

    return MonteCarloResult(
        ages=median_result.ages,
        financial_wealth_paths=financial_wealth_paths,
        total_wealth_paths=total_wealth_paths,
        human_capital_paths=human_capital_paths,
        total_consumption_paths=total_consumption_paths,
        subsistence_consumption_paths=subsistence_consumption_paths,
        variable_consumption_paths=variable_consumption_paths,
        stock_weight_paths=stock_weight_paths,
        bond_weight_paths=bond_weight_paths,
        cash_weight_paths=cash_weight_paths,
        stock_return_paths=stock_return_paths,
        interest_rate_paths=rate_paths,
        default_flags=default_flags,
        default_ages=default_ages,
        final_wealth=final_wealth,
        total_lifetime_consumption=total_lifetime_consumption,
        median_result=median_result,
        target_stock=target_stock,
        target_bond=target_bond,
        target_cash=target_cash,
    )


def run_strategy_comparison(
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    n_simulations: int = 50,
    random_seed: int = 42,
    bad_returns_early: bool = False,
    percentiles: List[int] = None,
) -> StrategyComparisonResult:
    """
    Run a comparison between optimal and rule-of-thumb strategies.

    Both strategies are run with identical random seeds (same market conditions)
    for a fair comparison. Computes percentile statistics across simulations.

    Args:
        params: Lifecycle parameters
        econ_params: Economic parameters
        n_simulations: Number of simulation paths (default 50)
        random_seed: Random seed for reproducibility
        bad_returns_early: If True, simulate bad returns in early retirement
        percentiles: List of percentiles to compute (default [5, 25, 50, 75, 95])

    Returns:
        StrategyComparisonResult with paths and statistics for both strategies
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    total_years = params.end_age - params.start_age
    working_years = params.retirement_age - params.start_age
    ages = np.arange(params.start_age, params.end_age)

    # Compute optimal target allocations from MV optimization
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

    # Generate random shocks once - same for both strategies
    rng = np.random.default_rng(random_seed)
    rate_shocks, stock_shocks = generate_correlated_shocks(
        total_years, n_simulations, econ_params.rho, rng
    )

    # Simulate interest rate paths
    initial_rate = econ_params.r_bar
    rate_paths = simulate_interest_rates(
        initial_rate, total_years, n_simulations, econ_params, rate_shocks
    )

    # Simulate stock return paths
    stock_return_paths = simulate_stock_returns(rate_paths, econ_params, stock_shocks)

    # Apply bad returns early if requested (stress test)
    if bad_returns_early:
        # Bad returns: -20% for first 5 years of retirement
        for sim in range(n_simulations):
            for t in range(working_years, min(working_years + 5, total_years)):
                stock_return_paths[sim, t] = -0.20

    # Get median path for optimal strategy reference
    median_result = compute_lifecycle_median_path(params, econ_params)
    earnings = median_result.earnings.copy()
    expenses = median_result.expenses.copy()

    # Initialize output arrays
    optimal_wealth_paths = np.zeros((n_simulations, total_years))
    optimal_consumption_paths = np.zeros((n_simulations, total_years))
    optimal_default_flags = np.zeros(n_simulations, dtype=bool)
    optimal_default_ages = np.full(n_simulations, np.nan)

    rot_wealth_paths = np.zeros((n_simulations, total_years))
    rot_consumption_paths = np.zeros((n_simulations, total_years))
    rot_default_flags = np.zeros(n_simulations, dtype=bool)
    rot_default_ages = np.full(n_simulations, np.nan)

    # Sample allocation paths (from first simulation)
    rot_stock_weight_sample = np.zeros(total_years)
    rot_bond_weight_sample = np.zeros(total_years)
    rot_cash_weight_sample = np.zeros(total_years)

    # Consumption rate for optimal strategy
    r = econ_params.r_bar
    avg_median_return = (
        target_stock * (r + econ_params.mu_excess) +
        target_bond * r +
        target_cash * r
    )
    consumption_rate = avg_median_return + params.consumption_boost

    # Run simulations
    for sim in range(n_simulations):
        # ---- OPTIMAL STRATEGY ----
        optimal_wealth_paths[sim, 0] = params.initial_wealth
        opt_defaulted = False

        for t in range(total_years):
            fw = optimal_wealth_paths[sim, t]
            hc = median_result.human_capital[t]
            pv_exp = median_result.pv_expenses[t]
            net_worth = hc + fw - pv_exp

            # Consumption
            subsistence = expenses[t]
            variable = max(0, consumption_rate * net_worth)
            total_cons = subsistence + variable

            if t < working_years:
                if total_cons > earnings[t]:
                    total_cons = earnings[t]
            else:
                if opt_defaulted:
                    total_cons = 0
                elif fw <= 0:
                    opt_defaulted = True
                    optimal_default_flags[sim] = True
                    optimal_default_ages[sim] = params.start_age + t
                    total_cons = 0
                elif total_cons > fw:
                    total_cons = fw

            optimal_consumption_paths[sim, t] = total_cons

            # Evolve wealth
            if t < total_years - 1 and not opt_defaulted:
                savings = earnings[t] - total_cons

                # Portfolio weights (simplified - use target allocations)
                w_s = target_stock
                w_b = target_bond
                w_c = target_cash

                stock_ret = stock_return_paths[sim, t]
                bond_ret = rate_paths[sim, t] + econ_params.mu_bond
                cash_ret = rate_paths[sim, t]

                portfolio_return = w_s * stock_ret + w_b * bond_ret + w_c * cash_ret
                optimal_wealth_paths[sim, t + 1] = fw * (1 + portfolio_return) + savings

        # ---- RULE OF THUMB STRATEGY ----
        rot_result = compute_rule_of_thumb_strategy(
            params=params,
            econ_params=econ_params,
            savings_rate=0.20,
            withdrawal_rate=0.04,
            stock_returns=stock_return_paths[sim, :],
            interest_rates=rate_paths[sim, :],
        )

        rot_wealth_paths[sim, :] = rot_result.financial_wealth
        rot_consumption_paths[sim, :] = rot_result.total_consumption
        rot_default_flags[sim] = rot_result.defaulted
        if rot_result.default_age is not None:
            rot_default_ages[sim] = rot_result.default_age

        # Store sample allocation from first simulation
        if sim == 0:
            rot_stock_weight_sample = rot_result.stock_weight.copy()
            rot_bond_weight_sample = rot_result.bond_weight.copy()
            rot_cash_weight_sample = rot_result.cash_weight.copy()

    # Compute percentile statistics
    n_percentiles = len(percentiles)
    optimal_wealth_percentiles = np.zeros((n_percentiles, total_years))
    rot_wealth_percentiles = np.zeros((n_percentiles, total_years))
    optimal_consumption_percentiles = np.zeros((n_percentiles, total_years))
    rot_consumption_percentiles = np.zeros((n_percentiles, total_years))

    for i, p in enumerate(percentiles):
        optimal_wealth_percentiles[i, :] = np.percentile(optimal_wealth_paths, p, axis=0)
        rot_wealth_percentiles[i, :] = np.percentile(rot_wealth_paths, p, axis=0)
        optimal_consumption_percentiles[i, :] = np.percentile(optimal_consumption_paths, p, axis=0)
        rot_consumption_percentiles[i, :] = np.percentile(rot_consumption_paths, p, axis=0)

    # Compute summary statistics
    optimal_default_rate = np.mean(optimal_default_flags)
    rot_default_rate = np.mean(rot_default_flags)
    optimal_median_final_wealth = np.median(optimal_wealth_paths[:, -1])
    rot_median_final_wealth = np.median(rot_wealth_paths[:, -1])

    # Compute PV consumption for each simulation
    r = econ_params.r_bar
    optimal_pv_consumption = np.array([
        compute_pv_consumption(optimal_consumption_paths[sim, :], r)
        for sim in range(n_simulations)
    ])
    rot_pv_consumption = np.array([
        compute_pv_consumption(rot_consumption_paths[sim, :], r)
        for sim in range(n_simulations)
    ])

    # Compute PV consumption percentiles
    optimal_pv_consumption_percentiles = np.array([
        np.percentile(optimal_pv_consumption, p) for p in percentiles
    ])
    rot_pv_consumption_percentiles = np.array([
        np.percentile(rot_pv_consumption, p) for p in percentiles
    ])

    return StrategyComparisonResult(
        n_simulations=n_simulations,
        ages=ages,
        optimal_wealth_paths=optimal_wealth_paths,
        optimal_consumption_paths=optimal_consumption_paths,
        optimal_default_flags=optimal_default_flags,
        optimal_default_ages=optimal_default_ages,
        rot_wealth_paths=rot_wealth_paths,
        rot_consumption_paths=rot_consumption_paths,
        rot_default_flags=rot_default_flags,
        rot_default_ages=rot_default_ages,
        rot_stock_weight_sample=rot_stock_weight_sample,
        rot_bond_weight_sample=rot_bond_weight_sample,
        rot_cash_weight_sample=rot_cash_weight_sample,
        percentiles=percentiles,
        optimal_wealth_percentiles=optimal_wealth_percentiles,
        rot_wealth_percentiles=rot_wealth_percentiles,
        optimal_consumption_percentiles=optimal_consumption_percentiles,
        rot_consumption_percentiles=rot_consumption_percentiles,
        optimal_default_rate=optimal_default_rate,
        rot_default_rate=rot_default_rate,
        optimal_median_final_wealth=optimal_median_final_wealth,
        rot_median_final_wealth=rot_median_final_wealth,
        optimal_pv_consumption=optimal_pv_consumption,
        rot_pv_consumption=rot_pv_consumption,
        optimal_pv_consumption_percentiles=optimal_pv_consumption_percentiles,
        rot_pv_consumption_percentiles=rot_pv_consumption_percentiles,
        stock_return_paths=stock_return_paths,
        interest_rate_paths=rate_paths,
    )


# =============================================================================
# Teaching Scenario Analysis
# =============================================================================

@dataclass
class ScenarioResult:
    """Results from a teaching scenario simulation."""
    name: str
    description: str
    ages: np.ndarray
    financial_wealth: np.ndarray
    total_consumption: np.ndarray
    stock_weight: np.ndarray
    stock_returns: np.ndarray
    cumulative_consumption: np.ndarray


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


def create_strategy_comparison_figure(
    comparison_result: StrategyComparisonResult,
    params: LifecycleParams = None,
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

    Args:
        comparison_result: Results from run_strategy_comparison()
        params: Lifecycle parameters
        figsize: Figure size
        use_years: If True, x-axis shows years from career start
        title_suffix: Optional suffix for title (e.g., "(Bad Returns Early)")

    Returns:
        matplotlib Figure object
    """
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

    # Colors
    color_optimal = '#2ecc71'  # Green
    color_rot = '#3498db'      # Blue
    alpha_fan = 0.3

    # ---- (0,0): Default Risk Bar Chart ----
    ax = axes[0, 0]
    strategies = ['Optimal\n(Variable Consumption)', 'Rule of Thumb\n(100-Age, 4% Rule)']
    default_rates = [
        comparison_result.optimal_default_rate * 100,
        comparison_result.rot_default_rate * 100
    ]
    colors = [color_optimal, color_rot]
    bars = ax.bar(strategies, default_rates, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar, rate in zip(bars, default_rates):
        height = bar.get_height()
        ax.annotate(f'{rate:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Default Rate (%)')
    ax.set_title('Default Risk Comparison')
    ax.set_ylim(0, max(100, max(default_rates) * 1.2))

    # ---- (0,1): Consumption Percentile Fan Charts ----
    ax = axes[0, 1]

    # Find percentile indices for fan chart
    p_idx = {p: i for i, p in enumerate(comparison_result.percentiles)}

    # Optimal strategy fan
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

    # Rule-of-thumb fan
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

    # ---- (0,2): Wealth Percentile Fan Charts ----
    ax = axes[0, 2]

    # Optimal strategy fan
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

    # Rule-of-thumb fan
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

    # ---- (1,0): Rule-of-Thumb Allocation Glide Path ----
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

    # ---- (1,1): Summary Statistics Table ----
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
  - Savings Rate: 20% of income
  - Stock Allocation: (100 - age)%
  - Fixed Income: 50/50 bonds/cash
  - Retirement: 4% fixed withdrawal

Optimal Strategy:
  - Variable consumption based on net worth
  - MV-optimal allocation
  - Adapts to market conditions
"""

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ---- (1,2): Default Age Distribution ----
    ax = axes[1, 2]

    # Get default ages (remove NaN)
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


# =============================================================================
# Monte Carlo Visualization Functions
# =============================================================================

def create_monte_carlo_fan_chart(
    mc_result: MonteCarloResult,
    params: LifecycleParams = None,
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
    if params is None:
        params = LifecycleParams()

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

    # Percentile bands
    percentiles = [10, 25, 50, 75, 90]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#ff7f0e', '#d62728']
    alphas = [0.2, 0.3, 1.0, 0.3, 0.2]

    # Panel 1: Financial Wealth Fan Chart
    ax = axes[0, 0]
    fw_percentiles = np.percentile(mc_result.financial_wealth_paths, percentiles, axis=0)

    # Draw sample paths
    for i in range(min(n_sample_paths, mc_result.financial_wealth_paths.shape[0])):
        ax.plot(x, mc_result.financial_wealth_paths[i, :], color='blue', alpha=0.1, linewidth=0.5)

    # Draw percentile bands
    ax.fill_between(x, fw_percentiles[0], fw_percentiles[4], alpha=0.2, color='blue', label='10-90th pctl')
    ax.fill_between(x, fw_percentiles[1], fw_percentiles[3], alpha=0.3, color='blue', label='25-75th pctl')
    ax.plot(x, fw_percentiles[2], color='darkblue', linewidth=2, label='Median')
    ax.plot(x, mc_result.median_result.financial_wealth, color='green', linewidth=2,
            linestyle='--', label='Deterministic')

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth: Monte Carlo Fan Chart')
    ax.legend(loc='upper left', fontsize=8)

    # Panel 2: Consumption Fan Chart
    ax = axes[0, 1]
    cons_percentiles = np.percentile(mc_result.total_consumption_paths, percentiles, axis=0)

    for i in range(min(n_sample_paths, mc_result.total_consumption_paths.shape[0])):
        ax.plot(x, mc_result.total_consumption_paths[i, :], color='orange', alpha=0.1, linewidth=0.5)

    ax.fill_between(x, cons_percentiles[0], cons_percentiles[4], alpha=0.2, color='orange', label='10-90th pctl')
    ax.fill_between(x, cons_percentiles[1], cons_percentiles[3], alpha=0.3, color='orange', label='25-75th pctl')
    ax.plot(x, cons_percentiles[2], color='darkorange', linewidth=2, label='Median')
    ax.plot(x, mc_result.median_result.total_consumption, color='green', linewidth=2,
            linestyle='--', label='Deterministic')

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Consumption: Monte Carlo Fan Chart')
    ax.legend(loc='upper right', fontsize=8)

    # Panel 3: Final Wealth Distribution
    ax = axes[1, 0]
    final_wealth = mc_result.final_wealth
    non_default = final_wealth[~mc_result.default_flags]

    ax.hist(non_default, bins=50, alpha=0.7, color='blue', edgecolor='white', density=True)
    ax.axvline(x=np.median(non_default), color='red', linestyle='--', linewidth=2,
               label=f'Median: ${np.median(non_default):,.0f}k')
    ax.axvline(x=np.mean(non_default), color='green', linestyle='--', linewidth=2,
               label=f'Mean: ${np.mean(non_default):,.0f}k')

    ax.set_xlabel('Final Wealth ($ 000s)')
    ax.set_ylabel('Density')
    ax.set_title(f'Final Wealth Distribution (Non-Defaulted, n={len(non_default)})')
    ax.legend(loc='upper right', fontsize=9)

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
    mc_result: MonteCarloResult,
    params: LifecycleParams = None,
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
    if params is None:
        params = LifecycleParams()

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
                    alpha=0.3, color='green', label='10-90th pctl')
    ax.plot(x_ret, ret_percentiles[1]*100, color='darkgreen', linewidth=2, label='Median')
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
                    alpha=0.3, color='red', label='10-90th pctl')
    ax.plot(x, weight_percentiles[1]*100, color='darkred', linewidth=2, label='Median')
    ax.plot(x, mc_result.median_result.stock_weight_no_short*100, color='green',
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

    # Panel 5: Cumulative Consumption
    ax = axes[1, 1]
    cum_cons = np.cumsum(mc_result.total_consumption_paths, axis=1)
    cum_percentiles = np.percentile(cum_cons, percentiles, axis=0)
    ax.fill_between(x, cum_percentiles[0], cum_percentiles[2],
                    alpha=0.3, color='orange', label='10-90th pctl')
    ax.plot(x, cum_percentiles[1], color='darkorange', linewidth=2, label='Median')
    cum_det = np.cumsum(mc_result.median_result.total_consumption)
    ax.plot(x, cum_det, color='green', linewidth=2, linestyle='--', label='Deterministic')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Cumulative Lifetime Consumption')
    ax.legend(loc='upper left', fontsize=8)

    # Panel 6: Default Analysis
    ax = axes[1, 2]
    default_ages = mc_result.default_ages[~np.isnan(mc_result.default_ages)]
    if len(default_ages) > 0:
        ax.hist(default_ages, bins=20, alpha=0.7, color='red', edgecolor='white')
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


def create_teaching_scenarios_figure(
    scenarios: List[ScenarioResult],
    params: LifecycleParams = None,
    figsize: Tuple[int, int] = (18, 14),
    use_years: bool = True,
) -> plt.Figure:
    """
    Create visualization comparing teaching scenarios.

    Shows how different return sequences affect wealth and consumption outcomes.

    Args:
        scenarios: List of ScenarioResult objects
        params: Lifecycle parameters
        figsize: Figure size
        use_years: If True, use years from start on x-axis

    Returns:
        matplotlib Figure
    """
    if params is None:
        params = LifecycleParams()

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

    # Color palette for scenarios
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
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.3)
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
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    figsize: Tuple[int, int] = (16, 10),
    use_years: bool = True,
) -> plt.Figure:
    """
    Create visualization specifically illustrating sequence of returns risk.

    Compares two scenarios with identical average returns but different sequencing:
    1. Good returns early, bad returns late (favorable for accumulators)
    2. Bad returns early, good returns late (unfavorable - sequence risk)

    Args:
        params: Lifecycle parameters
        econ_params: Economic parameters
        figsize: Figure size
        use_years: If True, use years from start on x-axis

    Returns:
        matplotlib Figure
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    total_years = params.end_age - params.start_age
    working_years = params.retirement_age - params.start_age
    retirement_years = total_years - working_years

    # Create two return sequences with same average but different ordering
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

    colors = ['#2ecc71', '#e74c3c', '#3498db']

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
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.3)
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


def create_gamma_comparison_figure(
    gamma_values: list = [1.0, 2.0, 4.0, 8.0],
    base_params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    figsize: Tuple[int, int] = (16, 12),
    use_years: bool = True
) -> plt.Figure:
    """
    Create a figure comparing key metrics across different risk aversion (gamma) values.

    Shows how portfolio allocation changes with different risk tolerances.
    """
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
                linewidth=2, label=f'γ = {gamma}')
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
                linewidth=2, label=f'γ = {gamma}')
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
                linewidth=2, label=f'γ = {gamma}')
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
                linewidth=2, label=f'γ = {gamma}')
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
                linewidth=2, label=f'γ = {gamma}')
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
                linewidth=2, label=f'γ = {gamma}')
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
    base_params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    figsize: Tuple[int, int] = (16, 12),
    use_years: bool = True
) -> plt.Figure:
    """
    Create a figure comparing scenarios with different initial wealth levels.

    Useful for comparing student loan debt (-50k) to various savings levels.
    """
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
    base_params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    figsize: Tuple[int, int] = (16, 12),
    use_years: bool = True
) -> plt.Figure:
    """
    Create a figure comparing scenarios with different consumption boost levels.

    The boost is added to median return to get consumption rate.
    """
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
    base_params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    figsize: Tuple[int, int] = (16, 12),
    use_years: bool = True
) -> plt.Figure:
    """
    Create a figure comparing scenarios with different equity risk premiums.
    """
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
    base_params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    figsize: Tuple[int, int] = (16, 12),
    use_years: bool = True
) -> plt.Figure:
    """
    Create a figure comparing scenarios with different initial earnings levels.
    """
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
    base_params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    figsize: Tuple[int, int] = (16, 12),
    use_years: bool = True
) -> plt.Figure:
    """
    Create a figure comparing scenarios with different stock volatility levels.
    """
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
    # Earnings Profile
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, result.earnings, color=COLORS['earnings'], linewidth=2)
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5, label='Retirement')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Earnings Profile ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # Expense Profile
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x, result.expenses, color=COLORS['expenses'], linewidth=2)
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Expense Profile ($k)', fontweight='bold')

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

    # HC Decomposition (stacked area)
    ax = fig.add_subplot(gs[2, 1])
    ax.fill_between(x, 0, result.hc_cash_component, alpha=0.7, color=COLORS['cash'], label='HC Cash')
    ax.fill_between(x, result.hc_cash_component,
                   result.hc_cash_component + result.hc_bond_component,
                   alpha=0.7, color=COLORS['bond'], label='HC Bond')
    ax.fill_between(x, result.hc_cash_component + result.hc_bond_component,
                   result.hc_cash_component + result.hc_bond_component + result.hc_stock_component,
                   alpha=0.7, color=COLORS['stock'], label='HC Stock')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Human Capital Decomposition ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # Expense Liability Decomposition (stacked area)
    ax = fig.add_subplot(gs[3, 0])
    ax.fill_between(x, 0, result.exp_cash_component, alpha=0.7, color=COLORS['cash'], label='Expense Cash')
    ax.fill_between(x, result.exp_cash_component,
                   result.exp_cash_component + result.exp_bond_component,
                   alpha=0.7, color=COLORS['bond'], label='Expense Bond')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Expense Liability Decomposition ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # Net HC minus Expenses (stacked area)
    ax = fig.add_subplot(gs[3, 1])
    # Net = HC - Expenses for each component
    net_stock = result.hc_stock_component  # Expenses have no stock component
    net_bond = result.hc_bond_component - result.exp_bond_component
    net_cash = result.hc_cash_component - result.exp_cash_component

    ax.fill_between(x, 0, net_cash, alpha=0.7, color=COLORS['cash'], label='Net Cash')
    ax.fill_between(x, net_cash, net_cash + net_bond, alpha=0.7, color=COLORS['bond'], label='Net Bond')
    ax.fill_between(x, net_cash + net_bond, net_cash + net_bond + net_stock,
                   alpha=0.7, color=COLORS['stock'], label='Net Stock')
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
    use_years: bool = True
) -> str:
    """
    Generate a PDF report showing lifecycle investment strategy.

    NEW STRUCTURE (matching TSX visualizer):
    - Page 1: BASE CASE (Deterministic Median Path) - 4 sections, 10 charts
    - Page 2: MONTE CARLO (50 Runs) - 6 chart panels with percentile bands
    - Pages 3a-c: TEACHING SCENARIOS - Normal, Sequence Risk, Rate Shock

    Args:
        output_path: Path for output PDF file
        params: Lifecycle parameters (uses defaults if None)
        econ_params: Economic parameters (uses defaults if None)
        include_legacy_pages: If True, include old comparison pages (beta, gamma, etc.)
        use_years: If True, x-axis shows years from career start; if False, shows age

    Returns:
        Path to generated PDF file
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    with PdfPages(output_path) as pdf:
        # ====================================================================
        # PAGE 1: BASE CASE (Deterministic Median Path)
        # ====================================================================
        print("Generating Page 1: Base Case...")
        result = compute_lifecycle_median_path(params, econ_params)

        fig = create_base_case_page(
            result=result,
            params=params,
            econ_params=econ_params,
            figsize=(20, 24),
            use_years=use_years
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ====================================================================
        # PAGE 2: MONTE CARLO (50 Runs)
        # ====================================================================
        print("Generating Page 2: Monte Carlo...")
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

        # ====================================================================
        # PAGE 3a: TEACHING SCENARIO - Normal Market (Optimal vs Rule of Thumb)
        # ====================================================================
        print("Generating Page 3a: Normal Market Scenario...")
        fig = create_scenario_page(
            scenario_type='normal',
            params=params,
            econ_params=econ_params,
            figsize=(20, 18),
            use_years=use_years,
            n_simulations=50,
            random_seed=42
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ====================================================================
        # PAGE 3b: TEACHING SCENARIO - Sequence Risk (Bad Early Returns)
        # ====================================================================
        print("Generating Page 3b: Sequence Risk Scenario...")
        fig = create_scenario_page(
            scenario_type='sequenceRisk',
            params=params,
            econ_params=econ_params,
            figsize=(20, 18),
            use_years=use_years,
            n_simulations=50,
            random_seed=42
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ====================================================================
        # PAGE 3c: TEACHING SCENARIO - Interest Rate Shock
        # ====================================================================
        print("Generating Page 3c: Rate Shock Scenario...")
        fig = create_scenario_page(
            scenario_type='rateShock',
            params=params,
            econ_params=econ_params,
            figsize=(20, 18),
            use_years=use_years,
            n_simulations=50,
            random_seed=42,
            rate_shock_age=params.retirement_age,
            rate_shock_magnitude=-0.02
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ====================================================================
        # LEGACY PAGES (optional - for parameter sensitivity analysis)
        # ====================================================================
        if include_legacy_pages:
            print("Generating legacy comparison pages...")

            # Beta comparison page
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

            # Old strategy comparison figures
            comparison_result = run_strategy_comparison(
                params=params,
                econ_params=econ_params,
                n_simulations=50,
                random_seed=42,
                bad_returns_early=False,
            )
            fig = create_strategy_comparison_figure(
                comparison_result=comparison_result,
                params=params,
                figsize=(18, 12),
                use_years=use_years,
                title_suffix=" (Normal Market Conditions)"
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Keep legacy individual pages support (now disabled by default)
        if False:  # Previously: include_individual_pages
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
                    gamma=params.gamma,
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
  - Bond Excess Return (mu_b): {econ_params.mu_bond*100:.2f}%
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
    initial_earnings: float = 100,
    stock_beta_hc: float = 0.1,
    bond_duration: float = 20.0,
    gamma: float = 2.0,
    mu_excess: float = 0.04,
    mu_bond: float = 0.005,
    sigma_s: float = 0.18,
    sigma_r: float = 0.012,
    rho: float = -0.2,
    r_bar: float = 0.02,
    consumption_share: float = 0.05,
    consumption_boost: float = 0.01,
    initial_wealth: float = 1.0,
    include_scenarios: bool = True,
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
        bond_duration: Bond duration for MV optimization (years)
        gamma: Risk aversion coefficient for MV optimization (0 = use fixed targets)
        mu_excess: Equity risk premium (stock excess return)
        mu_bond: Bond risk premium (excess return over short rate)
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
    """
    if verbose:
        print("Computing lifecycle investment strategy...")

    # Create economic parameters with consistent DGP
    econ_params = EconomicParams(
        r_bar=r_bar,
        mu_excess=mu_excess,
        mu_bond=mu_bond,
        sigma_s=sigma_s,
        sigma_r=sigma_r,
        rho=rho,
        bond_duration=bond_duration,
    )

    # Compute MV optimal allocation for display
    if gamma > 0:
        opt_stock, opt_bond, opt_cash = compute_mv_optimal_allocation(
            mu_stock=mu_excess,
            mu_bond=mu_bond,
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
                       help='Bond duration for MV optimization in years (default: 7.0)')
    parser.add_argument('--gamma', type=float, default=2.0,
                       help='Risk aversion for MV optimization (default: 2.0, 0=use fixed targets)')
    parser.add_argument('--mu-excess', type=float, default=0.04,
                       help='Equity risk premium (default: 0.04 = 4%%)')
    parser.add_argument('--mu-bond', type=float, default=0.005,
                       help='Bond risk premium over short rate (default: 0.005 = 0.5%%)')
    parser.add_argument('--sigma', type=float, default=0.18,
                       help='Stock return volatility (default: 0.18 = 18%%)')
    parser.add_argument('--sigma-r', type=float, default=0.012,
                       help='Interest rate shock volatility (default: 0.012 = 1.2%%)')
    parser.add_argument('--rho', type=float, default=-0.2,
                       help='Correlation between rate and stock shocks (default: -0.2)')
    parser.add_argument('--r-bar', type=float, default=0.02,
                       help='Long-run real risk-free rate (default: 0.02 = 2%%)')
    parser.add_argument('--consumption-share', type=float, default=0.05,
                       help='Share of net worth consumed above subsistence (default: 0.05)')
    parser.add_argument('--consumption-boost', type=float, default=0.01,
                       help='Boost above median return for consumption rate (default: 0.01 = 1%%)')
    parser.add_argument('--initial-wealth', type=float, default=1,
                       help='Initial financial wealth in $000s (default: 1, can be negative for student loans)')
    parser.add_argument('--use-age', action='store_true',
                       help='Use age instead of years from start on x-axis')
    parser.add_argument('--no-scenarios', action='store_true',
                       help='Skip legacy comparison pages (gamma, wealth, etc.) - main 5 pages always included')
    parser.add_argument('--include-legacy', action='store_true',
                       help='Include legacy parameter sensitivity pages (beta, gamma, volatility comparisons)')
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
        gamma=args.gamma,
        mu_excess=args.mu_excess,
        mu_bond=args.mu_bond,
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
    )
