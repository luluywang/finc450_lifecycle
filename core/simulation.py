"""
Simulation engines for lifecycle investment strategy.

This module contains the Monte Carlo simulation and strategy comparison functions
that form the core computational engine for lifecycle analysis.
"""

import numpy as np
from typing import Tuple, List, Optional

from .params import (
    LifecycleParams,
    EconomicParams,
    MonteCarloParams,
    LifecycleResult,
    MonteCarloResult,
    RuleOfThumbResult,
    StrategyComparisonResult,
    MedianPathComparisonResult,
)
from .economics import (
    compute_present_value,
    compute_pv_consumption,
    compute_duration,
    compute_mv_optimal_allocation,
    generate_correlated_shocks,
    simulate_interest_rates,
    simulate_stock_returns,
)


# =============================================================================
# Earnings and Expense Profiles
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


# =============================================================================
# Lifecycle Median Path Computation
# =============================================================================

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
    expenses[:working_years] = working_exp
    expenses[working_years:] = retirement_exp

    # Base savings (earnings - subsistence expenses)
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

    # Human capital = PV(earnings)
    human_capital = pv_earnings.copy()

    # Decompose human capital into stock/bond/cash components
    hc_stock_component = human_capital * params.stock_beta_human_capital
    non_stock_hc = human_capital * (1.0 - params.stock_beta_human_capital)

    hc_bond_component = np.zeros(total_years)
    hc_cash_component = np.zeros(total_years)

    for i in range(total_years):
        if econ_params.bond_duration > 0 and non_stock_hc[i] > 0:
            bond_fraction = duration_earnings[i] / econ_params.bond_duration
            hc_bond_component[i] = non_stock_hc[i] * bond_fraction
            hc_cash_component[i] = non_stock_hc[i] * (1.0 - bond_fraction)
        else:
            hc_cash_component[i] = non_stock_hc[i]

    # Decompose expense liability into bond/cash components
    exp_bond_component = np.zeros(total_years)
    exp_cash_component = np.zeros(total_years)

    for i in range(total_years):
        if econ_params.bond_duration > 0 and pv_expenses[i] > 0:
            bond_fraction = duration_expenses[i] / econ_params.bond_duration
            exp_bond_component[i] = pv_expenses[i] * bond_fraction
            exp_cash_component[i] = pv_expenses[i] * (1.0 - bond_fraction)
        else:
            exp_cash_component[i] = pv_expenses[i]

    # Financial wealth accumulation along median path with consumption model
    financial_wealth = np.zeros(total_years)
    financial_wealth[0] = params.initial_wealth

    # Consumption model arrays
    net_worth = np.zeros(total_years)
    subsistence_consumption = expenses.copy()
    variable_consumption = np.zeros(total_years)
    total_consumption = np.zeros(total_years)
    savings = np.zeros(total_years)

    # Expected return on financial portfolio
    expected_stock_return = r + econ_params.mu_excess
    avg_return = (
        target_stock * expected_stock_return +
        target_bond * r +
        target_cash * r
    )

    consumption_rate = avg_return + params.consumption_boost

    for i in range(total_years):
        net_worth[i] = human_capital[i] + financial_wealth[i] - pv_expenses[i]
        variable_consumption[i] = max(0, consumption_rate * net_worth[i])
        total_consumption[i] = subsistence_consumption[i] + variable_consumption[i]

        if earnings[i] > 0:
            if total_consumption[i] > earnings[i]:
                total_consumption[i] = earnings[i]
                variable_consumption[i] = max(0, earnings[i] - subsistence_consumption[i])
        else:
            fw = financial_wealth[i]
            if subsistence_consumption[i] > fw:
                total_consumption[i] = fw
                subsistence_consumption[i] = fw
                variable_consumption[i] = 0
            elif total_consumption[i] > fw:
                total_consumption[i] = fw
                variable_consumption[i] = fw - subsistence_consumption[i]

        savings[i] = earnings[i] - total_consumption[i]

        if i < total_years - 1:
            financial_wealth[i+1] = financial_wealth[i] * (1 + avg_return) + savings[i]

    total_wealth = financial_wealth + human_capital

    target_total_stocks = target_stock * total_wealth
    target_total_bonds = target_bond * total_wealth
    target_total_cash = target_cash * total_wealth

    target_fin_stocks = target_total_stocks - hc_stock_component
    target_fin_bonds = target_total_bonds - hc_bond_component + exp_bond_component
    target_fin_cash = target_total_cash - hc_cash_component + exp_cash_component

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
                equity = equity / total_agg
                fixed_income = fixed_income / total_agg
            else:
                equity = target_stock
                fixed_income = target_bond + target_cash

            if w_bond > 0 and w_cash > 0:
                fi_total = w_bond + w_cash
                w_b = fixed_income * (w_bond / fi_total)
                w_c = fixed_income * (w_cash / fi_total)
            elif w_bond > 0:
                w_b = fixed_income
                w_c = 0.0
            elif w_cash > 0:
                w_b = 0.0
                w_c = fixed_income
            else:
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

    total_stocks = stock_weight_no_short * financial_wealth + hc_stock_component
    total_bonds = bond_weight_no_short * financial_wealth + hc_bond_component
    total_cash = cash_weight_no_short * financial_wealth + hc_cash_component

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


# =============================================================================
# Rule of Thumb Strategy
# =============================================================================

def compute_rule_of_thumb_strategy(
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    savings_rate: float = 0.15,
    withdrawal_rate: float = 0.04,
    target_duration: float = 6.0,
    stock_returns: np.ndarray = None,
    interest_rates: np.ndarray = None,
) -> RuleOfThumbResult:
    """
    Compute lifecycle path using the classic "rule of thumb" financial advisor strategy.

    This implements traditional heuristics:
    - During working years: Save 15% of income
    - Allocation: (100 - age)% in stocks, rest split between bonds/cash to achieve target duration
    - At retirement: Freeze allocation at retirement age
    - Retirement withdrawal: 4% of initial retirement wealth (fixed, not adjusted)

    This is NOT optimal but represents common retail advice.
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

    # Compute expenses (subsistence)
    working_exp, retirement_exp = compute_expense_profile(params)
    expenses = np.zeros(total_years)
    expenses[:working_years] = working_exp
    expenses[working_years:] = retirement_exp

    # Initialize arrays
    financial_wealth = np.zeros(total_years)
    financial_wealth[0] = params.initial_wealth
    total_consumption = np.zeros(total_years)
    subsistence_consumption = np.zeros(total_years)
    variable_consumption = np.zeros(total_years)
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

        # Compute allocation: (100 - age)% stocks
        long_bond_duration = econ_params.bond_duration
        bond_weight_in_fi = min(1.0, target_duration / long_bond_duration) if long_bond_duration > 0 else 0.0

        if t < working_years:
            stock_pct = max(0.0, min(1.0, (100 - age) / 100.0))
            fixed_income_pct = 1.0 - stock_pct
            bond_pct = fixed_income_pct * bond_weight_in_fi
            cash_pct = fixed_income_pct * (1.0 - bond_weight_in_fi)
        else:
            if retirement_stock_weight is None:
                retirement_age = params.retirement_age
                retirement_stock_weight = max(0.0, min(1.0, (100 - retirement_age) / 100.0))
                retirement_fixed_income = 1.0 - retirement_stock_weight
                retirement_bond_weight = retirement_fixed_income * bond_weight_in_fi
                retirement_cash_weight = retirement_fixed_income * (1.0 - bond_weight_in_fi)
            stock_pct = retirement_stock_weight
            bond_pct = retirement_bond_weight
            cash_pct = retirement_cash_weight

        stock_weight[t] = stock_pct
        bond_weight[t] = bond_pct
        cash_weight[t] = cash_pct

        # Compute consumption with subsistence floor
        subsistence = expenses[t]
        if t < working_years:
            baseline_consumption = earnings[t] * (1.0 - savings_rate)

            if baseline_consumption >= subsistence:
                total_consumption[t] = baseline_consumption
                subsistence_consumption[t] = subsistence
                variable_consumption[t] = baseline_consumption - subsistence
            else:
                available = earnings[t] + fw
                if available >= subsistence:
                    total_consumption[t] = subsistence
                    subsistence_consumption[t] = subsistence
                    variable_consumption[t] = 0
                else:
                    total_consumption[t] = max(0, available)
                    subsistence_consumption[t] = total_consumption[t]
                    variable_consumption[t] = 0
        else:
            if fixed_retirement_consumption is None:
                fixed_retirement_consumption = withdrawal_rate * fw

            if defaulted:
                total_consumption[t] = 0
                subsistence_consumption[t] = 0
                variable_consumption[t] = 0
            elif fw <= 0:
                defaulted = True
                default_age = age
                total_consumption[t] = 0
                subsistence_consumption[t] = 0
                variable_consumption[t] = 0
            else:
                target_consumption = max(fixed_retirement_consumption, subsistence)
                if fw < target_consumption:
                    total_consumption[t] = fw
                    subsistence_consumption[t] = min(fw, subsistence)
                    variable_consumption[t] = max(0, fw - subsistence)
                else:
                    total_consumption[t] = target_consumption
                    subsistence_consumption[t] = subsistence
                    variable_consumption[t] = target_consumption - subsistence

        # Evolve wealth to next period
        if t < total_years - 1 and not defaulted:
            if t < working_years:
                sav = earnings[t] - total_consumption[t]
            else:
                sav = -total_consumption[t]

            stock_ret = stock_returns[t]
            bond_ret = interest_rates[t] + econ_params.mu_bond
            cash_ret = interest_rates[t]

            portfolio_return = (
                stock_weight[t] * stock_ret +
                bond_weight[t] * bond_ret +
                cash_weight[t] * cash_ret
            )

            financial_wealth[t + 1] = fw * (1 + portfolio_return) + sav
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
        target_duration=target_duration,
        subsistence_consumption=subsistence_consumption,
        variable_consumption=variable_consumption,
    )


# =============================================================================
# Strategy Comparison Functions
# =============================================================================

def compute_median_path_comparison(
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    rot_savings_rate: float = 0.15,
    rot_target_duration: float = 6.0,
    rot_withdrawal_rate: float = 0.04,
) -> MedianPathComparisonResult:
    """
    Compare LDI strategy vs Rule-of-Thumb on deterministic median paths.

    Both strategies use expected returns (no stochastic simulation).
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    # Compute LDI median path
    ldi_result = compute_lifecycle_median_path(params, econ_params)

    # Compute Rule-of-Thumb median path (using expected returns)
    rot_result = compute_rule_of_thumb_strategy(
        params=params,
        econ_params=econ_params,
        savings_rate=rot_savings_rate,
        withdrawal_rate=rot_withdrawal_rate,
        target_duration=rot_target_duration,
        stock_returns=None,
        interest_rates=None,
    )

    # Compute PV consumption for both strategies
    r = econ_params.r_bar
    ldi_pv = compute_pv_consumption(ldi_result.total_consumption, r)
    rot_pv = compute_pv_consumption(rot_result.total_consumption, r)

    return MedianPathComparisonResult(
        ages=ldi_result.ages,
        ldi_financial_wealth=ldi_result.financial_wealth,
        ldi_total_consumption=ldi_result.total_consumption,
        ldi_stock_weight=ldi_result.stock_weight_no_short,
        ldi_bond_weight=ldi_result.bond_weight_no_short,
        ldi_cash_weight=ldi_result.cash_weight_no_short,
        ldi_human_capital=ldi_result.human_capital,
        ldi_net_worth=ldi_result.net_worth,
        rot_financial_wealth=rot_result.financial_wealth,
        rot_total_consumption=rot_result.total_consumption,
        rot_stock_weight=rot_result.stock_weight,
        rot_bond_weight=rot_result.bond_weight,
        rot_cash_weight=rot_result.cash_weight,
        earnings=ldi_result.earnings,
        ldi_pv_consumption=ldi_pv,
        rot_pv_consumption=rot_pv,
        rot_savings_rate=rot_savings_rate,
        rot_target_duration=rot_target_duration,
        rot_withdrawal_rate=rot_withdrawal_rate,
    )


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

    # Generate correlated shocks
    rate_shocks, stock_shocks = generate_correlated_shocks(
        total_years, n_sims, econ_params.rho, rng
    )

    # Simulate interest rate paths
    rate_paths = simulate_interest_rates(
        initial_rate, total_years, n_sims, econ_params, rate_shocks
    )

    # Simulate stock return paths
    stock_return_paths = simulate_stock_returns(rate_paths, econ_params, stock_shocks)

    # Get earnings and expenses from median path
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

    # Consumption rate
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

            # Dynamic revaluation using current simulated rate
            current_rate = rate_paths[sim, t]

            # PV of remaining expenses
            remaining_expenses = expenses[t:]
            pv_exp = compute_present_value(remaining_expenses, current_rate,
                                           econ_params.phi, econ_params.r_bar)

            # Human capital = PV of remaining earnings
            if t < working_years:
                remaining_earnings = earnings[t:working_years]
                hc = compute_present_value(remaining_earnings, current_rate,
                                           econ_params.phi, econ_params.r_bar)
            else:
                hc = 0.0

            human_capital_paths[sim, t] = hc
            net_worth = hc + fw - pv_exp

            # Compute consumption
            subsistence = expenses[t]
            variable = max(0, consumption_rate * net_worth)
            total_cons = subsistence + variable

            # Apply constraints
            if t < working_years:
                if total_cons > earnings[t]:
                    total_cons = earnings[t]
                    variable = max(0, earnings[t] - subsistence)
            else:
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

            total_wealth_paths[sim, t] = fw + hc

            # Evolve wealth to next period
            if t < total_years - 1 and not defaulted:
                savings = earnings[t] - total_cons

                w_s = stock_weight_paths[sim, t]
                w_b = bond_weight_paths[sim, t]
                w_c = cash_weight_paths[sim, t]

                stock_ret = stock_return_paths[sim, t]
                bond_ret = rate_paths[sim, t] + econ_params.mu_bond
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
    rot_savings_rate: float = 0.15,
    rot_target_duration: float = 6.0,
    rot_withdrawal_rate: float = 0.04,
) -> StrategyComparisonResult:
    """
    Run a comparison between optimal and rule-of-thumb strategies.

    Both strategies are run with identical random seeds (same market conditions)
    for a fair comparison. Computes percentile statistics across simulations.
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

    # Compute optimal target allocations
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

    # Apply bad returns early if requested
    if bad_returns_early:
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

    rot_stock_weight_sample = np.zeros(total_years)
    rot_bond_weight_sample = np.zeros(total_years)
    rot_cash_weight_sample = np.zeros(total_years)

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

            current_rate = rate_paths[sim, t]

            remaining_expenses = expenses[t:]
            pv_exp = compute_present_value(remaining_expenses, current_rate,
                                           econ_params.phi, econ_params.r_bar)

            if t < working_years:
                remaining_earnings = earnings[t:working_years]
                hc = compute_present_value(remaining_earnings, current_rate,
                                           econ_params.phi, econ_params.r_bar)
            else:
                hc = 0.0

            net_worth = hc + fw - pv_exp

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

            if t < total_years - 1 and not opt_defaulted:
                savings = earnings[t] - total_cons

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
            savings_rate=rot_savings_rate,
            withdrawal_rate=rot_withdrawal_rate,
            target_duration=rot_target_duration,
            stock_returns=stock_return_paths[sim, :],
            interest_rates=rate_paths[sim, :],
        )

        rot_wealth_paths[sim, :] = rot_result.financial_wealth
        rot_consumption_paths[sim, :] = rot_result.total_consumption
        rot_default_flags[sim] = rot_result.defaulted
        if rot_result.default_age is not None:
            rot_default_ages[sim] = rot_result.default_age

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
    optimal_pv_consumption = np.array([
        compute_pv_consumption(optimal_consumption_paths[sim, :], r)
        for sim in range(n_simulations)
    ])
    rot_pv_consumption = np.array([
        compute_pv_consumption(rot_consumption_paths[sim, :], r)
        for sim in range(n_simulations)
    ])

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
# Fixed Consumption Strategy (4% Rule)
# =============================================================================

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

    # HC decomposition (unconstrained - bond fraction can exceed 1.0)
    hc_stock_component = human_capital * params.stock_beta_human_capital
    non_stock_hc = human_capital * (1.0 - params.stock_beta_human_capital)
    hc_bond_component = np.zeros(total_years)
    hc_cash_component = np.zeros(total_years)

    for i in range(total_years):
        if econ_params.bond_duration > 0 and non_stock_hc[i] > 0:
            bond_fraction = duration_earnings[i] / econ_params.bond_duration
            hc_bond_component[i] = non_stock_hc[i] * bond_fraction
            hc_cash_component[i] = non_stock_hc[i] * (1.0 - bond_fraction)
        else:
            hc_cash_component[i] = non_stock_hc[i]

    # Decompose expense liability into bond/cash components (unconstrained)
    exp_bond_component = np.zeros(total_years)
    exp_cash_component = np.zeros(total_years)

    for i in range(total_years):
        if econ_params.bond_duration > 0 and pv_expenses[i] > 0:
            bond_fraction = duration_expenses[i] / econ_params.bond_duration
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
    # Implicit = HC (asset) - Expenses (liability)
    # HC is bond-like asset → subtract; Expenses is bond-like liability → add
    target_fin_stocks = target_stock * total_wealth - hc_stock_component
    target_fin_bonds = target_bond * total_wealth - hc_bond_component + exp_bond_component
    target_fin_cash = target_cash * total_wealth - hc_cash_component + exp_cash_component

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
