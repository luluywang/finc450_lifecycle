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
    SimulationState,
    StrategyActions,
    StrategyProtocol,
    SimulationResult,
    StrategyComparison,
)
from .strategies import (
    LDIStrategy,
    RuleOfThumbStrategy,
    FixedConsumptionStrategy,
)
from .economics import (
    compute_present_value,
    compute_duration,
    compute_full_merton_allocation,
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
    hc_spread: float = 0.0,
    max_duration: float = None,
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
        hc_spread: CAPM spread for human capital (beta * mu_excess), default 0

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

        pv_earnings[t] = compute_present_value(remaining_earnings, r + hc_spread, phi, r + hc_spread)
        pv_expenses[t] = compute_present_value(remaining_expenses, r, phi, r)
        duration_earnings[t] = compute_duration(remaining_earnings, r + hc_spread, phi, r + hc_spread, max_duration=max_duration)
        duration_expenses[t] = compute_duration(remaining_expenses, r, phi, r, max_duration=max_duration)

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
        return compute_full_merton_allocation(
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
    max_leverage: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Normalize portfolio weights with leverage cap.

    - Stocks >= 0, Bonds >= 0 (no shorting risky assets)
    - Stocks + Bonds <= max_leverage * FW (cap total long exposure)
    - Cash = FW - Stocks - Bonds (residual, can be negative = borrowing)

    Key values for max_leverage:
        1.0: No borrowing (cash >= 0), equivalent to old allow_leverage=False
        2.0: Can borrow up to 1x FW
        inf: Unconstrained, equivalent to old allow_leverage=True

    Returns (stock_weight, bond_weight, cash_weight).
    """
    if fw <= 1e-6:
        # Clip MV targets (may be negative with unconstrained optimization)
        ws = max(0.0, target_stock)
        wb = max(0.0, target_bond)
        total = ws + wb
        if total > max_leverage:
            scale = max_leverage / total
            ws *= scale
            wb *= scale
        wc = 1.0 - ws - wb
        if ws + wb > 0 or wc > 0:
            return ws, wb, wc
        return 0.0, 0.0, 1.0

    # Clip stocks and bonds at 0 (no shorting risky assets)
    fin_stock = max(0.0, target_fin_stock)
    fin_bond = max(0.0, target_fin_bond)

    # Cap total long exposure at max_leverage * FW
    total_long = fin_stock + fin_bond
    max_long = max_leverage * fw
    if total_long > max_long:
        scale = max_long / total_long
        fin_stock *= scale
        fin_bond *= scale

    # Cash is residual: can be negative (= borrowing)
    fin_cash = fw - fin_stock - fin_bond

    return fin_stock / fw, fin_bond / fw, fin_cash / fw



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
    description: str = "",
    use_geometric_returns: bool = False,
) -> SimulationResult:
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
        description: Optional description for the simulation result

    Returns:
        SimulationResult containing all simulation outputs with unified field names.
        Arrays are 2D [n_sims, n_periods] for Monte Carlo, squeezed to 1D for single sim.
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

    # Get BASE earnings and expenses profiles (deterministic)
    earnings_profile = compute_earnings_profile(params)
    working_exp, retirement_exp = compute_expense_profile(params)

    base_earnings = np.zeros(total_years)
    expenses = np.zeros(total_years)
    base_earnings[:working_years] = earnings_profile
    expenses[:working_years] = working_exp
    expenses[working_years:] = retirement_exp

    r_bar = econ_params.r_bar
    phi = econ_params.phi

    # CAPM spread for human capital: discount earnings at r + beta * mu_excess
    hc_spread = params.stock_beta_human_capital * econ_params.mu_excess

    # Initialize wage shock tracking (for risky human capital)
    # log_wage_level tracks cumulative permanent shocks to wage level
    log_wage_level = np.zeros((n_sims, total_years))
    actual_earnings_paths = np.zeros((n_sims, total_years))

    # Initialize output arrays
    financial_wealth_paths = np.zeros((n_sims, total_years))
    human_capital_paths = np.zeros((n_sims, total_years))
    pv_expenses_paths = np.zeros((n_sims, total_years))
    net_worth_paths = np.zeros((n_sims, total_years))

    total_consumption_paths = np.zeros((n_sims, total_years))
    subsistence_consumption_paths = np.zeros((n_sims, total_years))
    variable_consumption_paths = np.zeros((n_sims, total_years))
    savings_paths = np.zeros((n_sims, total_years))

    stock_weight_paths = np.zeros((n_sims, total_years))
    bond_weight_paths = np.zeros((n_sims, total_years))
    cash_weight_paths = np.zeros((n_sims, total_years))

    target_fin_stock_paths = np.zeros((n_sims, total_years))
    target_fin_bond_paths = np.zeros((n_sims, total_years))
    target_fin_cash_paths = np.zeros((n_sims, total_years))

    hc_stock_paths = np.zeros((n_sims, total_years))
    hc_bond_paths = np.zeros((n_sims, total_years))
    hc_cash_paths = np.zeros((n_sims, total_years))
    exp_bond_paths = np.zeros((n_sims, total_years))
    exp_cash_paths = np.zeros((n_sims, total_years))
    duration_hc_paths = np.zeros((n_sims, total_years))
    duration_exp_paths = np.zeros((n_sims, total_years))

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

            # Apply cumulative wage shock (permanent effect on wage level)
            # Wage shocks are correlated with stock returns via stock_beta_human_capital
            if t > 0 and params.stock_beta_human_capital != 0:
                # Previous period's stock shock affects this period's wage level
                # Beta scales stock returns (sigma_s * shock), not raw shock
                log_wage_level[sim, t] = (
                    log_wage_level[sim, t - 1] +
                    params.stock_beta_human_capital * econ_params.sigma_s * stock_shocks[sim, t - 1]
                )

            wage_multiplier = np.exp(log_wage_level[sim, t])
            current_earnings = base_earnings[t] * wage_multiplier if is_working else 0.0
            actual_earnings_paths[sim, t] = current_earnings

            # Compute PV values at current rate (dynamic revaluation)
            remaining_expenses = expenses[t:]
            pv_exp = compute_present_value(remaining_expenses, current_rate, phi, r_bar)
            duration_exp = compute_duration(remaining_expenses, current_rate, phi, r_bar, max_duration=econ_params.max_duration)

            if is_working:
                # Scale remaining earnings by current wage level (permanent shock)
                # Discount at CAPM rate: current_rate + hc_spread
                remaining_base = base_earnings[t:working_years]
                remaining_earnings = remaining_base * wage_multiplier
                hc = compute_present_value(remaining_earnings, current_rate + hc_spread, phi, r_bar + hc_spread)
                duration_hc = compute_duration(remaining_earnings, current_rate + hc_spread, phi, r_bar + hc_spread, max_duration=econ_params.max_duration)
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
            net_worth = hc + fw - pv_exp
            net_worth_paths[sim, t] = net_worth

            # Store decomposition and duration paths
            hc_stock_paths[sim, t] = hc_stock
            hc_bond_paths[sim, t] = hc_bond
            hc_cash_paths[sim, t] = hc_cash
            exp_bond_paths[sim, t] = exp_bond
            exp_cash_paths[sim, t] = exp_cash
            duration_hc_paths[sim, t] = duration_hc
            duration_exp_paths[sim, t] = duration_exp

            # Build state for strategy
            state = SimulationState(
                t=t,
                age=age,
                is_working=is_working,
                financial_wealth=fw,
                human_capital=hc,
                pv_expenses=pv_exp,
                net_worth=net_worth,
                earnings=current_earnings,
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
            savings_paths[sim, t] = current_earnings - actions.total_consumption

            stock_weight_paths[sim, t] = actions.stock_weight
            bond_weight_paths[sim, t] = actions.bond_weight
            cash_weight_paths[sim, t] = actions.cash_weight

            if actions.target_fin_stock is not None:
                target_fin_stock_paths[sim, t] = actions.target_fin_stock
                target_fin_bond_paths[sim, t] = actions.target_fin_bond
                target_fin_cash_paths[sim, t] = actions.target_fin_cash

            # Evolve wealth to next period
            if t < total_years - 1 and not defaulted:
                if use_geometric_returns:
                    # Use geometric (median) return: E[R_p] - 0.5*Var(R_p)
                    w_s_act = actions.stock_weight
                    w_b_act = actions.bond_weight
                    w_c_act = actions.cash_weight
                    expected_ret = (
                        w_s_act * (current_rate + econ_params.mu_excess) +
                        w_b_act * (current_rate + econ_params.mu_bond) +
                        w_c_act * current_rate
                    )
                    port_var = (
                        w_s_act**2 * econ_params.sigma_s**2 +
                        w_b_act**2 * (econ_params.bond_duration * econ_params.sigma_r)**2 +
                        2 * w_s_act * w_b_act * (-econ_params.bond_duration * econ_params.sigma_s * econ_params.sigma_r * econ_params.rho)
                    )
                    portfolio_return = expected_ret - 0.5 * port_var
                else:
                    stock_ret = stock_return_paths[sim, t]
                    bond_ret = bond_return_paths[sim, t]
                    cash_ret = rate_paths[sim, t]
                    portfolio_return = (
                        actions.stock_weight * stock_ret +
                        actions.bond_weight * bond_ret +
                        actions.cash_weight * cash_ret
                    )

                savings = current_earnings - actions.total_consumption
                financial_wealth_paths[sim, t + 1] = fw * (1 + portfolio_return) + savings

    # Compute ages array
    ages = np.arange(params.start_age, params.end_age)

    # Compute final wealth (last period)
    final_wealth = financial_wealth_paths[:, -1]

    # Get strategy name from class name
    strategy_name = type(strategy).__name__

    # Check if any target_fin values were set (LDI-type strategies)
    has_target_fin = np.any(target_fin_stock_paths != 0) or np.any(target_fin_bond_paths != 0)

    # For single simulation, squeeze arrays to 1D for cleaner API
    if n_sims == 1:
        return SimulationResult(
            strategy_name=strategy_name,
            ages=ages,
            financial_wealth=financial_wealth_paths.squeeze(0),
            consumption=total_consumption_paths.squeeze(0),
            subsistence_consumption=subsistence_consumption_paths.squeeze(0),
            variable_consumption=variable_consumption_paths.squeeze(0),
            stock_weight=stock_weight_paths.squeeze(0),
            bond_weight=bond_weight_paths.squeeze(0),
            cash_weight=cash_weight_paths.squeeze(0),
            interest_rates=rate_paths.squeeze(0),
            stock_returns=stock_return_paths.squeeze(0),
            earnings=actual_earnings_paths.squeeze(0),
            defaulted=default_flags[0],
            default_age=default_ages[0],
            final_wealth=final_wealth[0],
            description=description,
            target_fin_stock=target_fin_stock_paths.squeeze(0) if has_target_fin else None,
            target_fin_bond=target_fin_bond_paths.squeeze(0) if has_target_fin else None,
            target_fin_cash=target_fin_cash_paths.squeeze(0) if has_target_fin else None,
            human_capital=human_capital_paths.squeeze(0),
            pv_expenses=pv_expenses_paths.squeeze(0),
            net_worth=net_worth_paths.squeeze(0),
            savings=savings_paths.squeeze(0),
            hc_stock_component=hc_stock_paths.squeeze(0),
            hc_bond_component=hc_bond_paths.squeeze(0),
            hc_cash_component=hc_cash_paths.squeeze(0),
            exp_bond_component=exp_bond_paths.squeeze(0),
            exp_cash_component=exp_cash_paths.squeeze(0),
            duration_hc=duration_hc_paths.squeeze(0),
            duration_expenses=duration_exp_paths.squeeze(0),
        )

    return SimulationResult(
        strategy_name=strategy_name,
        ages=ages,
        financial_wealth=financial_wealth_paths,
        consumption=total_consumption_paths,
        subsistence_consumption=subsistence_consumption_paths,
        variable_consumption=variable_consumption_paths,
        stock_weight=stock_weight_paths,
        bond_weight=bond_weight_paths,
        cash_weight=cash_weight_paths,
        interest_rates=rate_paths,
        stock_returns=stock_return_paths,
        earnings=actual_earnings_paths,
        defaulted=default_flags,
        default_age=default_ages,
        final_wealth=final_wealth,
        description=description,
        target_fin_stock=target_fin_stock_paths if has_target_fin else None,
        target_fin_bond=target_fin_bond_paths if has_target_fin else None,
        target_fin_cash=target_fin_cash_paths if has_target_fin else None,
        human_capital=human_capital_paths,
        pv_expenses=pv_expenses_paths,
        net_worth=net_worth_paths,
        savings=savings_paths,
        hc_stock_component=hc_stock_paths,
        hc_bond_component=hc_bond_paths,
        hc_cash_component=hc_cash_paths,
        exp_bond_component=exp_bond_paths,
        exp_cash_component=exp_cash_paths,
        duration_hc=duration_hc_paths,
        duration_expenses=duration_exp_paths,
    )


# =============================================================================
# Deprecated Legacy API (delegates to simulate_with_strategy)
# =============================================================================

def simulate_paths(
    params: LifecycleParams,
    econ_params: EconomicParams,
    rate_shocks: np.ndarray,
    stock_shocks: np.ndarray,
    initial_rate: float = None,
    use_dynamic_revaluation: bool = True,
    use_geometric_returns: bool = False,
) -> dict:
    """
    DEPRECATED: Use simulate_with_strategy() instead.

    Thin wrapper that delegates to simulate_with_strategy(LDIStrategy)
    and converts the result to the legacy dict format.

    The use_dynamic_revaluation parameter is ignored — Engine A always
    uses the rate path. With zero shocks at r_bar, current_rate=r_bar
    at every step, so the result is identical.
    """
    import warnings
    warnings.warn(
        "simulate_paths() is deprecated. Use simulate_with_strategy() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    strategy = LDIStrategy(max_leverage=params.max_leverage)
    result = simulate_with_strategy(
        strategy, params, econ_params,
        rate_shocks, stock_shocks,
        initial_rate=initial_rate,
        use_geometric_returns=use_geometric_returns,
    )

    return _sim_result_to_legacy_dict(result, params, econ_params)


def _sim_result_to_legacy_dict(
    result: SimulationResult,
    params: LifecycleParams,
    econ_params: EconomicParams,
) -> dict:
    """Convert SimulationResult to the legacy dict format returned by simulate_paths()."""
    n_sims = result.n_sims
    n_periods = result.n_periods

    # Compute static PV values at r_bar (legacy API returns these)
    total_years = params.end_age - params.start_age
    working_years = params.retirement_age - params.start_age
    earnings_profile = compute_earnings_profile(params)
    working_exp, retirement_exp = compute_expense_profile(params)
    base_earnings = np.zeros(total_years)
    expenses = np.zeros(total_years)
    base_earnings[:working_years] = earnings_profile
    expenses[:working_years] = working_exp
    expenses[working_years:] = retirement_exp

    hc_spread = params.stock_beta_human_capital * econ_params.mu_excess
    pv_earnings_static, pv_expenses_static, duration_earnings, duration_expenses = compute_static_pvs(
        base_earnings, expenses, working_years, total_years,
        econ_params.r_bar, econ_params.phi, hc_spread=hc_spread,
        max_duration=econ_params.max_duration,
    )

    # Ensure 2D arrays for multi-sim compatibility
    def ensure_2d(arr):
        if arr is not None and arr.ndim == 1:
            return arr[np.newaxis, :]
        return arr

    fw_paths = ensure_2d(result.financial_wealth)
    hc_paths = ensure_2d(result.human_capital)
    pv_exp_paths = ensure_2d(result.pv_expenses)
    nw_paths = ensure_2d(result.net_worth)
    cons_paths = ensure_2d(result.consumption)
    sub_cons_paths = ensure_2d(result.subsistence_consumption)
    var_cons_paths = ensure_2d(result.variable_consumption)
    savings_paths = ensure_2d(result.savings)
    sw_paths = ensure_2d(result.stock_weight)
    bw_paths = ensure_2d(result.bond_weight)
    cw_paths = ensure_2d(result.cash_weight)
    tfs_paths = ensure_2d(result.target_fin_stock)
    tfb_paths = ensure_2d(result.target_fin_bond)
    tfc_paths = ensure_2d(result.target_fin_cash)
    hcs_paths = ensure_2d(result.hc_stock_component)
    hcb_paths = ensure_2d(result.hc_bond_component)
    hcc_paths = ensure_2d(result.hc_cash_component)
    expb_paths = ensure_2d(result.exp_bond_component)
    expc_paths = ensure_2d(result.exp_cash_component)
    rate_paths = ensure_2d(result.interest_rates)
    sr_paths = ensure_2d(result.stock_returns)
    earn_paths = ensure_2d(result.earnings)

    # Compute bond return paths for legacy API
    if econ_params.bond_duration > 0:
        bond_return_paths = compute_duration_approx_returns(
            rate_paths, econ_params.bond_duration, econ_params
        ) + econ_params.mu_bond
    else:
        bond_return_paths = rate_paths[:, :-1] + econ_params.mu_bond

    # Total holdings
    total_stocks_paths = sw_paths * fw_paths + hcs_paths
    total_bonds_paths = bw_paths * fw_paths + hcb_paths
    total_cash_paths = cw_paths * fw_paths + hcc_paths

    # Default tracking
    default_flags = np.atleast_1d(result.defaulted)
    default_ages = np.atleast_1d(result.default_age)

    target_stock, target_bond, target_cash = compute_target_allocations(params, econ_params)

    return {
        'financial_wealth_paths': fw_paths,
        'human_capital_paths': hc_paths,
        'pv_expenses_paths': pv_exp_paths,
        'net_worth_paths': nw_paths,
        'total_consumption_paths': cons_paths,
        'subsistence_consumption_paths': sub_cons_paths,
        'variable_consumption_paths': var_cons_paths,
        'savings_paths': savings_paths,
        'stock_weight_paths': sw_paths,
        'bond_weight_paths': bw_paths,
        'cash_weight_paths': cw_paths,
        'target_fin_stocks_paths': tfs_paths,
        'target_fin_bonds_paths': tfb_paths,
        'target_fin_cash_paths': tfc_paths,
        'total_stocks_paths': total_stocks_paths,
        'total_bonds_paths': total_bonds_paths,
        'total_cash_paths': total_cash_paths,
        'rate_paths': rate_paths,
        'stock_return_paths': sr_paths,
        'bond_return_paths': bond_return_paths,
        'default_flags': default_flags,
        'default_ages': default_ages,
        'base_earnings': base_earnings,
        'actual_earnings_paths': earn_paths,
        'expenses': expenses,
        'pv_earnings_static': pv_earnings_static,
        'pv_expenses_static': pv_expenses_static,
        'duration_earnings': duration_earnings,
        'duration_expenses': duration_expenses,
        'hc_stock_paths': hcs_paths,
        'hc_bond_paths': hcb_paths,
        'hc_cash_paths': hcc_paths,
        'exp_bond_paths': expb_paths,
        'exp_cash_paths': expc_paths,
        'hc_stock_component': hcs_paths[0],
        'hc_bond_component': hcb_paths[0],
        'hc_cash_component': hcc_paths[0],
        'exp_bond_component': expb_paths[0],
        'exp_cash_component': expc_paths[0],
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
# Conversion Helpers: SimulationResult -> LifecycleResult / MonteCarloResult
# =============================================================================

def _sim_result_to_lifecycle_result(
    result: SimulationResult,
    params: LifecycleParams,
    econ_params: EconomicParams,
) -> LifecycleResult:
    """
    Convert a 1D SimulationResult (zero-shock run) to LifecycleResult.

    Since zero shocks at r_bar produce current_rate=r_bar at every step,
    the dynamic values equal the old "static" values — no separate
    compute_static_pvs() call needed.
    """
    n_periods = len(result.ages)
    fw = result.financial_wealth

    # Compute expense profile for the result
    working_exp, retirement_exp = compute_expense_profile(params)
    total_years = params.end_age - params.start_age
    working_years = params.retirement_age - params.start_age
    expenses = np.zeros(total_years)
    expenses[:working_years] = working_exp
    expenses[working_years:] = retirement_exp

    # Total holdings
    total_stocks = result.stock_weight * fw + result.hc_stock_component
    total_bonds = result.bond_weight * fw + result.hc_bond_component
    total_cash = result.cash_weight * fw + result.hc_cash_component

    # Consumption share of FW
    consumption_share_of_fw = np.where(
        fw > 1e-6, result.consumption / fw, np.nan
    )

    return LifecycleResult(
        ages=result.ages,
        earnings=result.earnings,
        expenses=expenses,
        savings=result.savings,
        pv_earnings=result.human_capital,
        pv_expenses=result.pv_expenses,
        human_capital=result.human_capital,
        hc_stock_component=result.hc_stock_component,
        hc_bond_component=result.hc_bond_component,
        hc_cash_component=result.hc_cash_component,
        duration_earnings=result.duration_hc,
        duration_expenses=result.duration_expenses,
        financial_wealth=fw,
        target_fin_stocks=result.target_fin_stock,
        target_fin_bonds=result.target_fin_bond,
        target_fin_cash=result.target_fin_cash,
        stock_weight_no_short=result.stock_weight,
        bond_weight_no_short=result.bond_weight,
        cash_weight_no_short=result.cash_weight,
        total_stocks=total_stocks,
        total_bonds=total_bonds,
        total_cash=total_cash,
        exp_bond_component=result.exp_bond_component,
        exp_cash_component=result.exp_cash_component,
        net_worth=result.net_worth,
        subsistence_consumption=result.subsistence_consumption,
        variable_consumption=result.variable_consumption,
        total_consumption=result.consumption,
        consumption_share_of_fw=consumption_share_of_fw,
        interest_rates=result.interest_rates,
        stock_returns=result.stock_returns,
    )


def _sim_result_to_mc_result(
    result: SimulationResult,
    median_result: LifecycleResult,
    params: LifecycleParams,
    econ_params: EconomicParams,
) -> MonteCarloResult:
    """
    Convert a 2D SimulationResult (MC run) to MonteCarloResult.
    """
    target_stock, target_bond, target_cash = compute_target_allocations(params, econ_params)
    total_lifetime_consumption = np.sum(result.consumption, axis=1)

    return MonteCarloResult(
        ages=result.ages,
        financial_wealth_paths=result.financial_wealth,
        human_capital_paths=result.human_capital,
        total_consumption_paths=result.consumption,
        subsistence_consumption_paths=result.subsistence_consumption,
        variable_consumption_paths=result.variable_consumption,
        actual_earnings_paths=result.earnings,
        stock_weight_paths=result.stock_weight,
        bond_weight_paths=result.bond_weight,
        cash_weight_paths=result.cash_weight,
        stock_return_paths=result.stock_returns,
        interest_rate_paths=result.interest_rates,
        default_flags=result.defaulted,
        default_ages=result.default_age,
        final_wealth=result.final_wealth,
        total_lifetime_consumption=total_lifetime_consumption,
        median_result=median_result,
        target_stock=target_stock,
        target_bond=target_bond,
        target_cash=target_cash,
    )


# =============================================================================
# Lifecycle Median Path Computation
# =============================================================================

def compute_lifecycle_median_path(
    params: LifecycleParams = None,
    econ_params: EconomicParams = None
) -> LifecycleResult:
    """
    Compute lifecycle investment strategy along the median (deterministic) path.

    Uses simulate_with_strategy(LDIStrategy) with zero shocks.
    Zero shocks + initial_rate=r_bar naturally produces a constant-rate path,
    making use_dynamic_revaluation branches unnecessary.

    use_geometric_returns=True makes this a true median path
    (not expected-value trajectory).
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    n_periods = params.end_age - params.start_age

    strategy = LDIStrategy(max_leverage=params.max_leverage)
    result = simulate_with_strategy(
        strategy, params, econ_params,
        np.zeros((1, n_periods)), np.zeros((1, n_periods)),
        initial_rate=econ_params.r_bar,
        use_geometric_returns=True,
    )

    return _sim_result_to_lifecycle_result(result, params, econ_params)


# =============================================================================
# Strategy Comparison Functions
# =============================================================================

def compute_median_path_comparison(
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    rot_savings_rate: float = 0.15,
    rot_target_duration: float = 6.0,
    rot_withdrawal_rate: float = 0.04,
) -> StrategyComparison:
    """
    Compare LDI strategy vs Rule-of-Thumb on deterministic median paths.

    Both strategies use expected returns (zero shocks = deterministic path).

    Returns:
        StrategyComparison with result_a = LDI, result_b = RuleOfThumb
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    # Zero shocks for deterministic median paths
    n_periods = params.end_age - params.start_age
    zero_rate_shocks = np.zeros((1, n_periods))
    zero_stock_shocks = np.zeros((1, n_periods))

    # Strategy 1: LDI (Liability-Driven Investment)
    ldi_strategy = LDIStrategy()
    ldi_result = simulate_with_strategy(
        ldi_strategy, params, econ_params,
        zero_rate_shocks, zero_stock_shocks,
        description="LDI (Liability-Driven Investment)",
        use_geometric_returns=True,
    )

    # Strategy 2: Rule-of-Thumb (100-age rule)
    rot_strategy = RuleOfThumbStrategy(
        savings_rate=rot_savings_rate,
        withdrawal_rate=rot_withdrawal_rate,
        target_duration=rot_target_duration,
    )
    rot_result = simulate_with_strategy(
        rot_strategy, params, econ_params,
        zero_rate_shocks, zero_stock_shocks,
        description="Rule-of-Thumb (100-age rule)",
        use_geometric_returns=True,
    )

    return StrategyComparison(
        result_a=ldi_result,
        result_b=rot_result,
        strategy_a_params={'max_leverage': 1.0},
        strategy_b_params={
            'savings_rate': rot_savings_rate,
            'withdrawal_rate': rot_withdrawal_rate,
            'target_duration': rot_target_duration,
        },
    )


def run_lifecycle_monte_carlo(
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    mc_params: MonteCarloParams = None,
    initial_rate: float = None,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation of lifecycle investment strategy.

    Uses simulate_with_strategy(LDIStrategy) with random shocks.
    Engine A always uses the rate path — no use_dynamic_revaluation needed.
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()
    if mc_params is None:
        mc_params = MonteCarloParams()
    if initial_rate is None:
        initial_rate = econ_params.r_bar

    median_result = compute_lifecycle_median_path(params, econ_params)

    n_sims = mc_params.n_simulations
    n_periods = params.end_age - params.start_age
    rng = np.random.default_rng(mc_params.random_seed)

    rate_shocks, stock_shocks = generate_correlated_shocks(
        n_periods, n_sims, econ_params.rho, rng
    )

    strategy = LDIStrategy(max_leverage=params.max_leverage)
    result = simulate_with_strategy(
        strategy, params, econ_params,
        rate_shocks, stock_shocks,
        initial_rate=initial_rate,
    )

    return _sim_result_to_mc_result(result, median_result, params, econ_params)


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
) -> StrategyComparison:
    """
    Run a comparison between LDI and Rule-of-Thumb strategies.

    Both strategies are run with identical random shocks (same market conditions)
    for a fair comparison. Statistics are computed on demand via StrategyComparison methods.

    Args:
        params: Lifecycle parameters
        econ_params: Economic parameters
        n_simulations: Number of Monte Carlo simulations
        random_seed: Random seed for reproducibility
        bad_returns_early: DEPRECATED - use compare_teaching_scenarios instead
        percentiles: DEPRECATED - use comparison.wealth_percentiles() instead
        rot_savings_rate: Rule-of-thumb savings rate during working years
        rot_target_duration: Rule-of-thumb target bond duration
        rot_withdrawal_rate: Rule-of-thumb withdrawal rate in retirement

    Returns:
        StrategyComparison with result_a = LDI, result_b = RuleOfThumb
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    if bad_returns_early:
        import warnings
        warnings.warn(
            "bad_returns_early is deprecated. Use compare_teaching_scenarios.py instead.",
            DeprecationWarning
        )

    # Generate random shocks once - same for both strategies
    n_periods = params.end_age - params.start_age
    rng = np.random.default_rng(random_seed)
    rate_shocks, stock_shocks = generate_correlated_shocks(
        n_periods, n_simulations, econ_params.rho, rng
    )

    # Strategy 1: LDI (Liability-Driven Investment)
    ldi_strategy = LDIStrategy()
    ldi_result = simulate_with_strategy(
        ldi_strategy, params, econ_params,
        rate_shocks, stock_shocks,
        description="LDI (Optimal)"
    )

    # Strategy 2: Rule-of-Thumb
    rot_strategy = RuleOfThumbStrategy(
        savings_rate=rot_savings_rate,
        withdrawal_rate=rot_withdrawal_rate,
        target_duration=rot_target_duration,
    )
    rot_result = simulate_with_strategy(
        rot_strategy, params, econ_params,
        rate_shocks, stock_shocks,
        description="Rule-of-Thumb"
    )

    return StrategyComparison(
        result_a=ldi_result,
        result_b=rot_result,
        strategy_a_params={'max_leverage': 1.0},
        strategy_b_params={
            'savings_rate': rot_savings_rate,
            'withdrawal_rate': rot_withdrawal_rate,
            'target_duration': rot_target_duration,
        },
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

    Uses simulate_with_strategy(FixedConsumptionStrategy) with zero shocks.

    - During working years: consume subsistence expenses only (save the rest)
    - During retirement: consume a fixed percentage of retirement wealth each year
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    n_periods = params.end_age - params.start_age

    strategy = FixedConsumptionStrategy(
        withdrawal_rate=withdrawal_rate,
        max_leverage=params.max_leverage,
    )
    result = simulate_with_strategy(
        strategy, params, econ_params,
        np.zeros((1, n_periods)), np.zeros((1, n_periods)),
        initial_rate=econ_params.r_bar,
        use_geometric_returns=True,
    )

    return _sim_result_to_lifecycle_result(result, params, econ_params)
