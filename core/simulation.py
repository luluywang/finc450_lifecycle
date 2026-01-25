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
    SimulationState,
    StrategyActions,
    StrategyProtocol,
)
from .economics import (
    compute_present_value,
    compute_pv_consumption,
    compute_duration,
    compute_mv_optimal_allocation,
    generate_correlated_shocks,
    simulate_interest_rates,
    simulate_stock_returns,
    compute_duration_approx_returns,
)


# =============================================================================
# DRY Helper Functions for PV and Decomposition (Single Source of Truth)
# =============================================================================

def compute_static_pvs(
    earnings: np.ndarray,
    expenses: np.ndarray,
    working_years: int,
    total_years: int,
    r: float,
    phi: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PV values and durations for earnings and expenses at constant rate.

    KEY INSIGHT: "Static" = "Dynamic with current_rate = r_bar"

    This function uses the same compute_present_value() as dynamic calculations,
    just with current_rate fixed at r_bar. This unifies the codebase:
    - Static/median path: call with current_rate = r_bar
    - Dynamic/Monte Carlo: call with current_rate from simulated rate path

    Args:
        earnings: Array of earnings over total_years
        expenses: Array of expenses over total_years
        working_years: Number of working years
        total_years: Total years in simulation
        r: Risk-free rate (r_bar) - used as current_rate for PV calculation
        phi: Mean reversion parameter

    Returns:
        Tuple of (pv_earnings, pv_expenses, duration_earnings, duration_expenses)
        Each array has shape (total_years,)
    """
    pv_earnings = np.zeros(total_years)
    pv_expenses = np.zeros(total_years)
    duration_earnings = np.zeros(total_years)
    duration_expenses = np.zeros(total_years)

    for t in range(total_years):
        if t < working_years:
            remaining_earnings = earnings[t:working_years]
        else:
            remaining_earnings = np.array([])
        remaining_expenses = expenses[t:]

        pv_earnings[t] = compute_present_value(remaining_earnings, r, phi, r)
        pv_expenses[t] = compute_present_value(remaining_expenses, r, phi, r)
        duration_earnings[t] = compute_duration(remaining_earnings, r, phi, r)
        duration_expenses[t] = compute_duration(remaining_expenses, r, phi, r)

    return pv_earnings, pv_expenses, duration_earnings, duration_expenses


def decompose_hc_to_components(
    pv_earnings: np.ndarray,
    duration_earnings: np.ndarray,
    stock_beta: float,
    bond_duration: float,
    total_years: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose human capital into stock, bond, and cash components.

    This is the single source of truth for HC decomposition, replacing
    duplicate loops in simulate_paths() and compute_lifecycle_fixed_consumption().

    Args:
        pv_earnings: PV of future earnings at each time
        duration_earnings: Duration of future earnings at each time
        stock_beta: Beta of human capital to stocks
        bond_duration: Duration of bonds
        total_years: Total years in simulation

    Returns:
        Tuple of (hc_stock_component, hc_bond_component, hc_cash_component)
    """
    hc_stock_component = pv_earnings * stock_beta
    non_stock_hc = pv_earnings * (1.0 - stock_beta)
    hc_bond_component = np.zeros(total_years)
    hc_cash_component = np.zeros(total_years)

    for t in range(total_years):
        if bond_duration > 0 and non_stock_hc[t] > 0:
            bond_fraction = duration_earnings[t] / bond_duration
            hc_bond_component[t] = non_stock_hc[t] * bond_fraction
            hc_cash_component[t] = non_stock_hc[t] * (1.0 - bond_fraction)
        else:
            hc_cash_component[t] = non_stock_hc[t]

    return hc_stock_component, hc_bond_component, hc_cash_component


def decompose_expenses_to_components(
    pv_expenses: np.ndarray,
    duration_expenses: np.ndarray,
    bond_duration: float,
    total_years: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose expense liability into bond and cash components.

    This is the single source of truth for expense decomposition, replacing
    duplicate loops in simulate_paths() and compute_lifecycle_fixed_consumption().

    Args:
        pv_expenses: PV of future expenses at each time
        duration_expenses: Duration of future expenses at each time
        bond_duration: Duration of bonds
        total_years: Total years in simulation

    Returns:
        Tuple of (exp_bond_component, exp_cash_component)
    """
    exp_bond_component = np.zeros(total_years)
    exp_cash_component = np.zeros(total_years)

    for t in range(total_years):
        if bond_duration > 0 and pv_expenses[t] > 0:
            bond_fraction = duration_expenses[t] / bond_duration
            exp_bond_component[t] = pv_expenses[t] * bond_fraction
            exp_cash_component[t] = pv_expenses[t] * (1.0 - bond_fraction)
        else:
            exp_cash_component[t] = pv_expenses[t]

    return exp_bond_component, exp_cash_component


# =============================================================================
# Helper Functions for Unified Simulation Engine
# =============================================================================

def compute_target_allocations(
    params: LifecycleParams,
    econ_params: EconomicParams
) -> Tuple[float, float, float]:
    """
    Compute target portfolio allocations (stock, bond, cash).

    Uses MV optimization if gamma > 0, otherwise returns fixed targets.
    """
    if params.gamma > 0:
        return compute_mv_optimal_allocation(
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
        return target_stock, target_bond, target_cash


def normalize_portfolio_weights(
    target_fin_stock: float,
    target_fin_bond: float,
    target_fin_cash: float,
    fw: float,
    target_stock: float,
    target_bond: float,
    target_cash: float,
    allow_leverage: bool = False,
) -> Tuple[float, float, float]:
    """
    Normalize portfolio weights with optional no-short constraint.

    Takes target financial holdings and normalizes to valid weights.

    When allow_leverage=False (default):
        - Clips negative weights to 0
        - Normalizes to sum to 1.0
        - Preserves relative proportions where possible

    When allow_leverage=True:
        - Returns raw weights (can be negative or >1)
        - Allows shorting and leveraged positions
        - Properly sizes the LDI hedge regardless of financial wealth

    Returns (stock_weight, bond_weight, cash_weight).
    """
    if fw <= 1e-6:
        return target_stock, target_bond, target_cash

    w_stock = target_fin_stock / fw
    w_bond = target_fin_bond / fw
    w_cash = target_fin_cash / fw

    # If leverage is allowed, return raw (unconstrained) weights
    if allow_leverage:
        return w_stock, w_bond, w_cash

    # Apply no-short constraint
    equity = max(0, w_stock)
    fixed_income = max(0, w_bond + w_cash)

    total_agg = equity + fixed_income
    if total_agg > 0:
        equity /= total_agg
        fixed_income /= total_agg
    else:
        equity = target_stock
        fixed_income = target_bond + target_cash

    # Split fixed income between bonds and cash
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

    return equity, w_b, w_c


def apply_consumption_constraints(
    subsistence: float,
    variable: float,
    earnings: float,
    fw: float,
    is_working: bool,
    defaulted: bool,
) -> Tuple[float, float, float, bool]:
    """
    Apply consumption constraints based on lifecycle stage.

    Returns (total_consumption, subsistence, variable, defaulted).
    """
    total_cons = subsistence + variable

    if is_working:
        # During working years: can't consume more than earnings
        if total_cons > earnings:
            total_cons = earnings
            variable = max(0, earnings - subsistence)
        return total_cons, subsistence, variable, defaulted

    # Retirement phase
    if defaulted:
        return 0.0, 0.0, 0.0, True

    if fw <= 0:
        return 0.0, 0.0, 0.0, True  # Default!

    if subsistence > fw:
        return fw, fw, 0.0, defaulted

    if total_cons > fw:
        return fw, subsistence, fw - subsistence, defaulted

    return total_cons, subsistence, variable, defaulted


def compute_dynamic_pv(
    cashflows: np.ndarray,
    current_rate: float,
    phi: float,
    r_bar: float,
) -> float:
    """
    Compute present value using dynamic rate revaluation.

    This is a thin wrapper around compute_present_value for clarity.
    """
    return compute_present_value(cashflows, current_rate, phi, r_bar)


# =============================================================================
# Generic Strategy Simulation Engine
# =============================================================================

def simulate_with_strategy(
    strategy: StrategyProtocol,
    params: LifecycleParams,
    econ_params: EconomicParams,
    rate_shocks: np.ndarray,
    stock_shocks: np.ndarray,
    initial_rate: float = None,
) -> dict:
    """
    Generic simulation engine that runs ANY strategy.

    This is the single source of truth for simulation logic. Strategies are
    simple functions that map state to actions, allowing easy comparison.

    Key insight: Static calculations (median path) are just dynamic calculations
    with zero shocks. This unifies the codebase.

    Args:
        strategy: Strategy implementing StrategyProtocol (maps state -> actions)
        params: Lifecycle parameters
        econ_params: Economic parameters
        rate_shocks: Shape (n_sims, n_periods) - interest rate epsilon shocks
        stock_shocks: Shape (n_sims, n_periods) - stock return epsilon shocks
        initial_rate: Starting interest rate (defaults to r_bar)

    Returns:
        Dictionary containing all simulation outputs:
        - financial_wealth_paths: (n_sims, n_periods)
        - human_capital_paths: (n_sims, n_periods)
        - total_wealth_paths: (n_sims, n_periods)
        - consumption paths (total, subsistence, variable)
        - weight paths (stock, bond, cash)
        - rate_paths, stock_return_paths
        - default_flags, default_ages
    """
    if initial_rate is None:
        initial_rate = econ_params.r_bar

    n_sims, n_periods = rate_shocks.shape
    total_years = params.end_age - params.start_age
    working_years = params.retirement_age - params.start_age

    assert n_periods == total_years, f"Shock periods {n_periods} != total years {total_years}"

    # Compute target allocations (from MV optimization or fixed)
    target_stock, target_bond, target_cash = compute_target_allocations(params, econ_params)

    # Simulate market paths from shocks
    rate_paths = simulate_interest_rates(
        initial_rate, total_years, n_sims, econ_params, rate_shocks
    )
    stock_return_paths = simulate_stock_returns(rate_paths, econ_params, stock_shocks)

    # Compute bond returns using duration approximation
    # bond_return ≈ yield + spread - duration × Δr
    if econ_params.bond_duration > 0:
        bond_return_paths = compute_duration_approx_returns(
            rate_paths, econ_params.bond_duration, econ_params
        )
        bond_return_paths = bond_return_paths + econ_params.mu_bond
    else:
        bond_return_paths = rate_paths[:, :-1] + econ_params.mu_bond

    # Get earnings and expenses profiles
    earnings_profile = compute_earnings_profile(params)
    working_exp, retirement_exp = compute_expense_profile(params)

    earnings = np.zeros(total_years)
    expenses = np.zeros(total_years)
    earnings[:working_years] = earnings_profile
    expenses[:working_years] = working_exp
    expenses[working_years:] = retirement_exp

    r_bar = econ_params.r_bar
    phi = econ_params.phi

    # Initialize output arrays
    financial_wealth_paths = np.zeros((n_sims, total_years))
    human_capital_paths = np.zeros((n_sims, total_years))
    total_wealth_paths = np.zeros((n_sims, total_years))
    pv_expenses_paths = np.zeros((n_sims, total_years))
    net_worth_paths = np.zeros((n_sims, total_years))

    total_consumption_paths = np.zeros((n_sims, total_years))
    subsistence_consumption_paths = np.zeros((n_sims, total_years))
    variable_consumption_paths = np.zeros((n_sims, total_years))
    savings_paths = np.zeros((n_sims, total_years))

    stock_weight_paths = np.zeros((n_sims, total_years))
    bond_weight_paths = np.zeros((n_sims, total_years))
    cash_weight_paths = np.zeros((n_sims, total_years))

    default_flags = np.zeros(n_sims, dtype=bool)
    default_ages = np.full(n_sims, np.nan)

    # Set initial wealth
    financial_wealth_paths[:, 0] = params.initial_wealth

    # Simulate each path
    for sim in range(n_sims):
        defaulted = False

        # Reset strategy state if it has a reset method (for RoT retirement values)
        if hasattr(strategy, 'reset'):
            strategy.reset()

        for t in range(total_years):
            fw = financial_wealth_paths[sim, t]
            is_working = t < working_years
            current_rate = rate_paths[sim, t]
            age = params.start_age + t

            # Compute PV values at current rate (dynamic revaluation)
            remaining_expenses = expenses[t:]
            pv_exp = compute_present_value(remaining_expenses, current_rate, phi, r_bar)
            duration_exp = compute_duration(remaining_expenses, current_rate, phi, r_bar)

            if is_working:
                remaining_earnings = earnings[t:working_years]
                hc = compute_present_value(remaining_earnings, current_rate, phi, r_bar)
                duration_hc = compute_duration(remaining_earnings, current_rate, phi, r_bar)
            else:
                hc = 0.0
                duration_hc = 0.0

            # Compute HC decomposition at current rate
            if is_working and hc > 0:
                hc_stock = hc * params.stock_beta_human_capital
                non_stock_hc = hc * (1.0 - params.stock_beta_human_capital)
                if econ_params.bond_duration > 0:
                    hc_bond_frac = duration_hc / econ_params.bond_duration
                    hc_bond = non_stock_hc * hc_bond_frac
                    hc_cash = non_stock_hc * (1.0 - hc_bond_frac)
                else:
                    hc_bond = 0.0
                    hc_cash = non_stock_hc
            else:
                hc_stock = 0.0
                hc_bond = 0.0
                hc_cash = 0.0

            # Compute expense decomposition at current rate
            if econ_params.bond_duration > 0 and pv_exp > 0:
                exp_bond_frac = duration_exp / econ_params.bond_duration
                exp_bond = pv_exp * exp_bond_frac
                exp_cash = pv_exp * (1.0 - exp_bond_frac)
            else:
                exp_bond = 0.0
                exp_cash = pv_exp

            # Store values
            human_capital_paths[sim, t] = hc
            pv_expenses_paths[sim, t] = pv_exp
            total_wealth = fw + hc
            total_wealth_paths[sim, t] = total_wealth
            net_worth = hc + fw - pv_exp
            net_worth_paths[sim, t] = net_worth

            # Build state for strategy
            state = SimulationState(
                t=t,
                age=age,
                is_working=is_working,
                financial_wealth=fw,
                human_capital=hc,
                pv_expenses=pv_exp,
                net_worth=net_worth,
                total_wealth=total_wealth,
                earnings=earnings[t],
                expenses=expenses[t],
                current_rate=current_rate,
                hc_stock_component=hc_stock,
                hc_bond_component=hc_bond,
                hc_cash_component=hc_cash,
                exp_bond_component=exp_bond,
                exp_cash_component=exp_cash,
                duration_hc=duration_hc,
                duration_expenses=duration_exp,
                target_stock=target_stock,
                target_bond=target_bond,
                target_cash=target_cash,
                params=params,
                econ_params=econ_params,
            )

            # Get actions from strategy
            if defaulted:
                actions = StrategyActions(
                    total_consumption=0.0,
                    subsistence_consumption=0.0,
                    variable_consumption=0.0,
                    stock_weight=target_stock,
                    bond_weight=target_bond,
                    cash_weight=target_cash,
                )
            else:
                actions = strategy(state)

            # Check for default
            if not is_working and fw <= 0 and not defaulted:
                defaulted = True
                default_flags[sim] = True
                default_ages[sim] = age

            # Store results
            total_consumption_paths[sim, t] = actions.total_consumption
            subsistence_consumption_paths[sim, t] = actions.subsistence_consumption
            variable_consumption_paths[sim, t] = actions.variable_consumption
            savings_paths[sim, t] = earnings[t] - actions.total_consumption

            stock_weight_paths[sim, t] = actions.stock_weight
            bond_weight_paths[sim, t] = actions.bond_weight
            cash_weight_paths[sim, t] = actions.cash_weight

            # Evolve wealth to next period
            if t < total_years - 1 and not defaulted:
                stock_ret = stock_return_paths[sim, t]
                bond_ret = bond_return_paths[sim, t]
                cash_ret = rate_paths[sim, t]

                portfolio_return = (
                    actions.stock_weight * stock_ret +
                    actions.bond_weight * bond_ret +
                    actions.cash_weight * cash_ret
                )
                savings = earnings[t] - actions.total_consumption
                financial_wealth_paths[sim, t + 1] = fw * (1 + portfolio_return) + savings

    return {
        # Core paths
        'financial_wealth_paths': financial_wealth_paths,
        'human_capital_paths': human_capital_paths,
        'total_wealth_paths': total_wealth_paths,
        'pv_expenses_paths': pv_expenses_paths,
        'net_worth_paths': net_worth_paths,
        # Consumption
        'total_consumption_paths': total_consumption_paths,
        'subsistence_consumption_paths': subsistence_consumption_paths,
        'variable_consumption_paths': variable_consumption_paths,
        'savings_paths': savings_paths,
        # Weights
        'stock_weight_paths': stock_weight_paths,
        'bond_weight_paths': bond_weight_paths,
        'cash_weight_paths': cash_weight_paths,
        # Market paths
        'rate_paths': rate_paths,
        'stock_return_paths': stock_return_paths,
        'bond_return_paths': bond_return_paths,
        # Default tracking
        'default_flags': default_flags,
        'default_ages': default_ages,
        # Reference arrays
        'earnings': earnings,
        'expenses': expenses,
        # Targets
        'target_stock': target_stock,
        'target_bond': target_bond,
        'target_cash': target_cash,
    }


# =============================================================================
# Unified Simulation Engine (Legacy - kept for backward compatibility)
# =============================================================================

def simulate_paths(
    params: LifecycleParams,
    econ_params: EconomicParams,
    rate_shocks: np.ndarray,
    stock_shocks: np.ndarray,
    initial_rate: float = None,
    use_dynamic_revaluation: bool = True,
) -> dict:
    """
    Unified simulation engine that takes epsilon shocks as input.

    This is the single source of truth for simulation logic:
    - Median path: pass zeros for shocks (n_sims=1)
    - Monte Carlo: pass random shocks from generate_correlated_shocks()

    Args:
        params: Lifecycle parameters
        econ_params: Economic parameters
        rate_shocks: Shape (n_sims, n_periods) - interest rate epsilon shocks
        stock_shocks: Shape (n_sims, n_periods) - stock return epsilon shocks
        initial_rate: Starting interest rate (defaults to r_bar)
        use_dynamic_revaluation: If True, revalue PV(expenses) and HC using
            current simulated rate. If False, use r_bar throughout.

    Returns:
        Dictionary containing all simulation outputs:
        - financial_wealth_paths: (n_sims, n_periods)
        - human_capital_paths: (n_sims, n_periods)
        - total_wealth_paths: (n_sims, n_periods)
        - consumption paths (total, subsistence, variable)
        - weight paths (stock, bond, cash)
        - rate_paths, stock_return_paths
        - default_flags, default_ages
        - And reference arrays (earnings, expenses, etc.)
    """
    if initial_rate is None:
        initial_rate = econ_params.r_bar

    n_sims, n_periods = rate_shocks.shape
    total_years = params.end_age - params.start_age
    working_years = params.retirement_age - params.start_age

    assert n_periods == total_years, f"Shock periods {n_periods} != total years {total_years}"

    # Compute target allocations
    target_stock, target_bond, target_cash = compute_target_allocations(params, econ_params)

    # Simulate interest rate and stock return paths from shocks
    rate_paths = simulate_interest_rates(
        initial_rate, total_years, n_sims, econ_params, rate_shocks
    )
    stock_return_paths = simulate_stock_returns(rate_paths, econ_params, stock_shocks)

    # Compute bond returns using duration approximation
    # bond_return ≈ yield + spread - duration × Δr
    if econ_params.bond_duration > 0:
        bond_return_paths = compute_duration_approx_returns(
            rate_paths, econ_params.bond_duration, econ_params
        )
        # Add bond spread to the duration-based returns
        bond_return_paths = bond_return_paths + econ_params.mu_bond
    else:
        # If no duration, bond returns are just yield + spread
        bond_return_paths = rate_paths[:, :-1] + econ_params.mu_bond

    # Get earnings and expenses profiles
    earnings_profile = compute_earnings_profile(params)
    working_exp, retirement_exp = compute_expense_profile(params)

    earnings = np.zeros(total_years)
    expenses = np.zeros(total_years)
    earnings[:working_years] = earnings_profile
    expenses[:working_years] = working_exp
    expenses[working_years:] = retirement_exp

    # Compute static PV values at r_bar (used for HC decomposition)
    r = econ_params.r_bar
    phi = econ_params.phi

    # Use DRY helper functions for PV and decomposition calculations
    pv_earnings_static, pv_expenses_static, duration_earnings, duration_expenses = compute_static_pvs(
        earnings, expenses, working_years, total_years, r, phi
    )

    # Compute HC decomposition (static, based on r_bar)
    hc_stock_component, hc_bond_component, hc_cash_component = decompose_hc_to_components(
        pv_earnings_static, duration_earnings, params.stock_beta_human_capital,
        econ_params.bond_duration, total_years
    )

    # Expense decomposition
    exp_bond_component, exp_cash_component = decompose_expenses_to_components(
        pv_expenses_static, duration_expenses, econ_params.bond_duration, total_years
    )

    # Consumption rate based on expected returns
    expected_stock_return = r + econ_params.mu_excess
    avg_return = (
        target_stock * expected_stock_return +
        target_bond * r +
        target_cash * r
    )
    consumption_rate = avg_return + params.consumption_boost

    # Initialize output arrays
    financial_wealth_paths = np.zeros((n_sims, total_years))
    human_capital_paths = np.zeros((n_sims, total_years))
    total_wealth_paths = np.zeros((n_sims, total_years))
    pv_expenses_paths = np.zeros((n_sims, total_years))
    net_worth_paths = np.zeros((n_sims, total_years))

    total_consumption_paths = np.zeros((n_sims, total_years))
    subsistence_consumption_paths = np.zeros((n_sims, total_years))
    variable_consumption_paths = np.zeros((n_sims, total_years))
    savings_paths = np.zeros((n_sims, total_years))

    stock_weight_paths = np.zeros((n_sims, total_years))
    bond_weight_paths = np.zeros((n_sims, total_years))
    cash_weight_paths = np.zeros((n_sims, total_years))

    target_fin_stocks_paths = np.zeros((n_sims, total_years))
    target_fin_bonds_paths = np.zeros((n_sims, total_years))
    target_fin_cash_paths = np.zeros((n_sims, total_years))

    default_flags = np.zeros(n_sims, dtype=bool)
    default_ages = np.full(n_sims, np.nan)

    # Set initial wealth
    financial_wealth_paths[:, 0] = params.initial_wealth

    # Simulate each path
    for sim in range(n_sims):
        defaulted = False

        for t in range(total_years):
            fw = financial_wealth_paths[sim, t]
            is_working = t < working_years

            # Compute PV values (dynamic or static)
            if use_dynamic_revaluation:
                current_rate = rate_paths[sim, t]
                remaining_expenses = expenses[t:]
                pv_exp = compute_dynamic_pv(remaining_expenses, current_rate, phi, r)

                if is_working:
                    remaining_earnings = earnings[t:working_years]
                    hc = compute_dynamic_pv(remaining_earnings, current_rate, phi, r)
                else:
                    hc = 0.0
            else:
                current_rate = r
                pv_exp = pv_expenses_static[t]
                hc = pv_earnings_static[t]

            human_capital_paths[sim, t] = hc
            pv_expenses_paths[sim, t] = pv_exp
            total_wealth = fw + hc
            total_wealth_paths[sim, t] = total_wealth
            net_worth = hc + fw - pv_exp
            net_worth_paths[sim, t] = net_worth

            # Compute consumption
            subsistence = expenses[t]
            variable = max(0, consumption_rate * net_worth)

            total_cons, subsistence, variable, defaulted = apply_consumption_constraints(
                subsistence, variable, earnings[t], fw, is_working, defaulted
            )

            if defaulted and not default_flags[sim]:
                default_flags[sim] = True
                default_ages[sim] = params.start_age + t

            total_consumption_paths[sim, t] = total_cons
            subsistence_consumption_paths[sim, t] = subsistence
            variable_consumption_paths[sim, t] = variable
            savings_paths[sim, t] = earnings[t] - total_cons

            # Compute portfolio weights
            # Target financial holdings = target total - HC component + expense component
            # When using dynamic revaluation, recalculate hedge components at current rate
            if use_dynamic_revaluation:
                # Recalculate expense hedge components at current rate
                remaining_expenses = expenses[t:]
                duration_exp_t = compute_duration(remaining_expenses, current_rate, phi, r)
                if econ_params.bond_duration > 0 and pv_exp > 0:
                    exp_bond_frac = duration_exp_t / econ_params.bond_duration
                    exp_bond_t = pv_exp * exp_bond_frac
                    exp_cash_t = pv_exp * (1.0 - exp_bond_frac)
                else:
                    exp_bond_t = 0.0
                    exp_cash_t = pv_exp

                # Recalculate HC hedge components at current rate (if working)
                if is_working and hc > 0:
                    remaining_earnings = earnings[t:working_years]
                    duration_hc_t = compute_duration(remaining_earnings, current_rate, phi, r)
                    hc_stock_t = hc * params.stock_beta_human_capital
                    non_stock_hc_t = hc * (1.0 - params.stock_beta_human_capital)
                    if econ_params.bond_duration > 0:
                        hc_bond_frac = duration_hc_t / econ_params.bond_duration
                        hc_bond_t = non_stock_hc_t * hc_bond_frac
                        hc_cash_t = non_stock_hc_t * (1.0 - hc_bond_frac)
                    else:
                        hc_bond_t = 0.0
                        hc_cash_t = non_stock_hc_t
                else:
                    hc_stock_t = 0.0
                    hc_bond_t = 0.0
                    hc_cash_t = 0.0
            else:
                # Use static (pre-computed at r_bar) hedge components
                exp_bond_t = exp_bond_component[t]
                exp_cash_t = exp_cash_component[t]
                hc_stock_t = hc_stock_component[t]
                hc_bond_t = hc_bond_component[t]
                hc_cash_t = hc_cash_component[t]

            target_fin_stock = target_stock * total_wealth - hc_stock_t
            target_fin_bond = target_bond * total_wealth - hc_bond_t + exp_bond_t
            target_fin_cash = target_cash * total_wealth - hc_cash_t + exp_cash_t

            target_fin_stocks_paths[sim, t] = target_fin_stock
            target_fin_bonds_paths[sim, t] = target_fin_bond
            target_fin_cash_paths[sim, t] = target_fin_cash

            w_s, w_b, w_c = normalize_portfolio_weights(
                target_fin_stock, target_fin_bond, target_fin_cash, fw,
                target_stock, target_bond, target_cash,
                allow_leverage=params.allow_leverage
            )

            stock_weight_paths[sim, t] = w_s
            bond_weight_paths[sim, t] = w_b
            cash_weight_paths[sim, t] = w_c

            # Evolve wealth to next period
            if t < total_years - 1 and not defaulted:
                stock_ret = stock_return_paths[sim, t]
                # Use duration-based bond returns (includes capital gains/losses from rate changes)
                bond_ret = bond_return_paths[sim, t]
                cash_ret = rate_paths[sim, t]

                portfolio_return = w_s * stock_ret + w_b * bond_ret + w_c * cash_ret
                savings = earnings[t] - total_cons
                financial_wealth_paths[sim, t + 1] = fw * (1 + portfolio_return) + savings

    # Compute total holdings
    total_stocks_paths = stock_weight_paths * financial_wealth_paths + hc_stock_component
    total_bonds_paths = bond_weight_paths * financial_wealth_paths + hc_bond_component
    total_cash_paths = cash_weight_paths * financial_wealth_paths + hc_cash_component

    return {
        # Core paths
        'financial_wealth_paths': financial_wealth_paths,
        'human_capital_paths': human_capital_paths,
        'total_wealth_paths': total_wealth_paths,
        'pv_expenses_paths': pv_expenses_paths,
        'net_worth_paths': net_worth_paths,
        # Consumption
        'total_consumption_paths': total_consumption_paths,
        'subsistence_consumption_paths': subsistence_consumption_paths,
        'variable_consumption_paths': variable_consumption_paths,
        'savings_paths': savings_paths,
        # Weights
        'stock_weight_paths': stock_weight_paths,
        'bond_weight_paths': bond_weight_paths,
        'cash_weight_paths': cash_weight_paths,
        # Target holdings
        'target_fin_stocks_paths': target_fin_stocks_paths,
        'target_fin_bonds_paths': target_fin_bonds_paths,
        'target_fin_cash_paths': target_fin_cash_paths,
        # Total holdings
        'total_stocks_paths': total_stocks_paths,
        'total_bonds_paths': total_bonds_paths,
        'total_cash_paths': total_cash_paths,
        # Market paths
        'rate_paths': rate_paths,
        'stock_return_paths': stock_return_paths,
        'bond_return_paths': bond_return_paths,
        # Default tracking
        'default_flags': default_flags,
        'default_ages': default_ages,
        # Reference arrays (static)
        'earnings': earnings,
        'expenses': expenses,
        'pv_earnings_static': pv_earnings_static,
        'pv_expenses_static': pv_expenses_static,
        'duration_earnings': duration_earnings,
        'duration_expenses': duration_expenses,
        'hc_stock_component': hc_stock_component,
        'hc_bond_component': hc_bond_component,
        'hc_cash_component': hc_cash_component,
        'exp_bond_component': exp_bond_component,
        'exp_cash_component': exp_cash_component,
        # Targets
        'target_stock': target_stock,
        'target_bond': target_bond,
        'target_cash': target_cash,
    }


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

    This is a thin wrapper around simulate_paths() with zero shocks,
    producing the expected-value path with no stochastic variation.

    This computes:
    - Earnings and expense profiles
    - Human capital (PV of future earnings)
    - Portfolio decomposition of human capital
    - Optimal financial portfolio to achieve target total allocation
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    n_periods = params.end_age - params.start_age
    ages = np.arange(params.start_age, params.end_age)

    # Zero shocks = deterministic expected-value path
    rate_shocks = np.zeros((1, n_periods))
    stock_shocks = np.zeros((1, n_periods))

    # Run simulation with zero shocks and no dynamic revaluation
    result = simulate_paths(
        params, econ_params, rate_shocks, stock_shocks,
        initial_rate=econ_params.r_bar,
        use_dynamic_revaluation=False
    )

    # Extract single path (sim=0) from result arrays
    financial_wealth = result['financial_wealth_paths'][0]
    total_wealth = result['total_wealth_paths'][0]
    human_capital = result['human_capital_paths'][0]
    net_worth = result['net_worth_paths'][0]

    # Consumption arrays
    total_consumption = result['total_consumption_paths'][0]
    subsistence_consumption = result['subsistence_consumption_paths'][0]
    variable_consumption = result['variable_consumption_paths'][0]
    savings = result['savings_paths'][0]

    # Weight arrays
    stock_weight_no_short = result['stock_weight_paths'][0]
    bond_weight_no_short = result['bond_weight_paths'][0]
    cash_weight_no_short = result['cash_weight_paths'][0]

    # Target holdings
    target_fin_stocks = result['target_fin_stocks_paths'][0]
    target_fin_bonds = result['target_fin_bonds_paths'][0]
    target_fin_cash = result['target_fin_cash_paths'][0]

    # Total holdings
    total_stocks = result['total_stocks_paths'][0]
    total_bonds = result['total_bonds_paths'][0]
    total_cash = result['total_cash_paths'][0]

    # Compute consumption share of FW
    consumption_share_of_fw = np.zeros(n_periods)
    for i in range(n_periods):
        if financial_wealth[i] > 1e-6:
            consumption_share_of_fw[i] = total_consumption[i] / financial_wealth[i]
        else:
            consumption_share_of_fw[i] = np.nan

    return LifecycleResult(
        ages=ages,
        earnings=result['earnings'],
        expenses=result['expenses'],
        savings=savings,
        pv_earnings=result['pv_earnings_static'],
        pv_expenses=result['pv_expenses_static'],
        human_capital=human_capital,
        hc_stock_component=result['hc_stock_component'],
        hc_bond_component=result['hc_bond_component'],
        hc_cash_component=result['hc_cash_component'],
        duration_earnings=result['duration_earnings'],
        duration_expenses=result['duration_expenses'],
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
        exp_bond_component=result['exp_bond_component'],
        exp_cash_component=result['exp_cash_component'],
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
    bond_returns: np.ndarray = None,
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
    if bond_returns is None:
        # Default to yield + spread (for median path without duration effects)
        bond_returns = interest_rates + econ_params.mu_bond

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
            # Use duration-based bond returns if provided
            bond_ret = bond_returns[t]
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

    This is a thin wrapper around simulate_paths() with random shocks,
    producing stochastic paths with dynamic revaluation.

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

    # Get median path for reference (returned in result)
    median_result = compute_lifecycle_median_path(params, econ_params)

    # Simulation setup
    n_sims = mc_params.n_simulations
    n_periods = params.end_age - params.start_age
    ages = np.arange(params.start_age, params.end_age)
    rng = np.random.default_rng(mc_params.random_seed)

    # Generate random shocks
    rate_shocks, stock_shocks = generate_correlated_shocks(
        n_periods, n_sims, econ_params.rho, rng
    )

    # Run simulation with random shocks and dynamic revaluation
    result = simulate_paths(
        params, econ_params, rate_shocks, stock_shocks,
        initial_rate=initial_rate,
        use_dynamic_revaluation=True
    )

    # Compute summary statistics
    final_wealth = result['financial_wealth_paths'][:, -1]
    total_lifetime_consumption = np.sum(result['total_consumption_paths'], axis=1)

    return MonteCarloResult(
        ages=ages,
        financial_wealth_paths=result['financial_wealth_paths'],
        total_wealth_paths=result['total_wealth_paths'],
        human_capital_paths=result['human_capital_paths'],
        total_consumption_paths=result['total_consumption_paths'],
        subsistence_consumption_paths=result['subsistence_consumption_paths'],
        variable_consumption_paths=result['variable_consumption_paths'],
        stock_weight_paths=result['stock_weight_paths'],
        bond_weight_paths=result['bond_weight_paths'],
        cash_weight_paths=result['cash_weight_paths'],
        stock_return_paths=result['stock_return_paths'],
        interest_rate_paths=result['rate_paths'],
        default_flags=result['default_flags'],
        default_ages=result['default_ages'],
        final_wealth=final_wealth,
        total_lifetime_consumption=total_lifetime_consumption,
        median_result=median_result,
        target_stock=result['target_stock'],
        target_bond=result['target_bond'],
        target_cash=result['target_cash'],
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

    # Compute bond returns using duration approximation
    if econ_params.bond_duration > 0:
        bond_return_paths = compute_duration_approx_returns(
            rate_paths, econ_params.bond_duration, econ_params
        )
        bond_return_paths = bond_return_paths + econ_params.mu_bond
    else:
        bond_return_paths = rate_paths[:, :-1] + econ_params.mu_bond

    # Apply bad returns early if requested
    if bad_returns_early:
        for sim in range(n_simulations):
            for t in range(working_years, min(working_years + 5, total_years)):
                stock_return_paths[sim, t] = -0.20

    # Get median path for optimal strategy reference
    median_result = compute_lifecycle_median_path(params, econ_params)
    earnings = median_result.earnings.copy()
    expenses = median_result.expenses.copy()

    # Pre-compute static HC and expense decompositions (at r_bar) using DRY helper
    r = econ_params.r_bar
    phi = econ_params.phi

    pv_earnings_static, pv_expenses_static, duration_earnings_static, duration_expenses_static = compute_static_pvs(
        earnings, expenses, working_years, total_years, r, phi
    )

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

                # Compute LDI hedge: dynamic HC and expense decomposition at current rate
                total_wealth = fw + hc

                # Compute duration of remaining expenses at current rate
                remaining_expenses = expenses[t:]
                duration_exp_t = compute_duration(remaining_expenses, current_rate, phi, r)
                if econ_params.bond_duration > 0 and pv_exp > 0:
                    exp_bond_frac = duration_exp_t / econ_params.bond_duration
                    exp_bond_t = pv_exp * exp_bond_frac
                    exp_cash_t = pv_exp * (1.0 - exp_bond_frac)
                else:
                    exp_bond_t = 0.0
                    exp_cash_t = pv_exp

                # Compute HC decomposition at current rate (if working)
                if t < working_years and hc > 0:
                    remaining_earnings = earnings[t:working_years]
                    duration_hc_t = compute_duration(remaining_earnings, current_rate, phi, r)
                    hc_stock_t = hc * params.stock_beta_human_capital
                    non_stock_hc_t = hc * (1.0 - params.stock_beta_human_capital)
                    if econ_params.bond_duration > 0:
                        hc_bond_frac = duration_hc_t / econ_params.bond_duration
                        hc_bond_t = non_stock_hc_t * hc_bond_frac
                        hc_cash_t = non_stock_hc_t * (1.0 - hc_bond_frac)
                    else:
                        hc_bond_t = 0.0
                        hc_cash_t = non_stock_hc_t
                else:
                    hc_stock_t = 0.0
                    hc_bond_t = 0.0
                    hc_cash_t = 0.0

                # Target financial holdings = target total - HC component + expense component
                target_fin_stock = target_stock * total_wealth - hc_stock_t
                target_fin_bond = target_bond * total_wealth - hc_bond_t + exp_bond_t
                target_fin_cash = target_cash * total_wealth - hc_cash_t + exp_cash_t

                # Normalize to portfolio weights (with no-short constraint)
                w_s, w_b, w_c = normalize_portfolio_weights(
                    target_fin_stock, target_fin_bond, target_fin_cash, fw,
                    target_stock, target_bond, target_cash,
                    allow_leverage=params.allow_leverage
                )

                stock_ret = stock_return_paths[sim, t]
                # Use duration-based bond returns (includes capital gains/losses from rate changes)
                bond_ret = bond_return_paths[sim, t]
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
            bond_returns=bond_return_paths[sim, :],
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

    # Forward-looking present values using DRY helper
    pv_earnings, pv_expenses, duration_earnings, duration_expenses = compute_static_pvs(
        earnings, expenses, working_years, total_years, r, phi
    )

    # Human capital
    human_capital = pv_earnings.copy()

    # HC decomposition using DRY helper (unconstrained - bond fraction can exceed 1.0)
    hc_stock_component, hc_bond_component, hc_cash_component = decompose_hc_to_components(
        human_capital, duration_earnings, params.stock_beta_human_capital,
        econ_params.bond_duration, total_years
    )

    # Decompose expense liability into bond/cash components using DRY helper
    exp_bond_component, exp_cash_component = decompose_expenses_to_components(
        pv_expenses, duration_expenses, econ_params.bond_duration, total_years
    )

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

    # Use normalize_portfolio_weights helper for consistent weight normalization
    for i in range(total_years):
        fw = financial_wealth[i]
        w_s, w_b, w_c = normalize_portfolio_weights(
            target_fin_stocks[i], target_fin_bonds[i], target_fin_cash[i], fw,
            target_stock, target_bond, target_cash,
            allow_leverage=False  # 4% rule uses no leverage
        )
        stock_weight_no_short[i] = w_s
        bond_weight_no_short[i] = w_b
        cash_weight_no_short[i] = w_c

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
