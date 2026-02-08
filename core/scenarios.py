"""
Teaching scenario generation for lifecycle investment analysis.

This module provides functions to create custom scenarios with specified
return sequences, useful for demonstrating concepts like sequence risk,
market crashes, and volatility impacts.
"""

import numpy as np
from typing import List

from .params import (
    LifecycleParams,
    EconomicParams,
    ScenarioResult,
    SimulationResult,
)
from .economics import compute_full_merton_allocation, annuity_consumption_rate
from .simulation import compute_lifecycle_median_path


def create_teaching_scenario(
    name: str,
    description: str,
    returns_override: np.ndarray,
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
) -> ScenarioResult:
    """
    Create a teaching scenario with specified return sequence.

    This function simulates the LDI strategy but with custom stock returns
    instead of stochastic or expected returns. Useful for demonstrating
    sequence risk, market crashes, etc.

    Args:
        name: Scenario name (e.g., "Early Crash")
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
        target_stock, target_bond, target_cash = compute_full_merton_allocation(
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

    # Consumption rate (deterministic scenario uses r_bar since rates don't evolve)
    # Uses realized weights from median path, but for simplicity uses target weights
    # since this is a simplified teaching scenario function
    r = econ_params.r_bar
    expected_return = (
        target_stock * (r + econ_params.mu_excess) +
        target_bond * (r + econ_params.mu_bond) +
        target_cash * r
    )
    # Jensen's correction: median return = E[r] - 0.5 * Var(r_portfolio)
    sigma_b = econ_params.bond_duration * econ_params.sigma_r
    cov_sb = -econ_params.bond_duration * econ_params.sigma_s * econ_params.sigma_r * econ_params.rho
    portfolio_var = (
        target_stock**2 * econ_params.sigma_s**2 +
        target_bond**2 * sigma_b**2 +
        2 * target_stock * target_bond * cov_sb
    )
    ce_return = expected_return - 0.5 * portfolio_var + params.consumption_boost

    defaulted = False

    for t in range(total_years):
        fw = financial_wealth[t]
        hc = human_capital[t]
        net_worth = hc + fw - pv_expenses[t]

        # Compute consumption rate (time-varying if annuity mode)
        if params.annuity_consumption:
            remaining = total_years - t
            consumption_rate = annuity_consumption_rate(ce_return, remaining)
        else:
            consumption_rate = ce_return

        subsistence = expenses[t]
        variable = max(0, consumption_rate * net_worth)
        total_cons = subsistence + variable

        # Apply constraints: leave at least $1K investable
        available = max(0.0, fw + earnings[t] - 1.0)
        if defaulted:
            total_cons = 0
        elif total_cons > available:
            total_cons = available
            # Default if can't meet subsistence in retirement
            if t >= working_years and total_cons < subsistence:
                defaulted = True

        total_consumption[t] = total_cons

        # Compute portfolio weight using surplus optimization
        surplus = max(0, net_worth)
        target_fin_stock = target_stock * surplus - hc_stock[t]

        # Evolve wealth
        if t < total_years - 1 and not defaulted:
            savings = earnings[t] - total_cons
            investable = fw + savings

            # Normalize weights to investable base for exact hedge
            if investable > 1e-6:
                w_stock = target_fin_stock / investable
                w_stock = max(0, min(1, w_stock))  # Constrain to [0, 1]
            else:
                w_stock = target_stock

            stock_weight[t] = w_stock

            # Use overridden stock return
            stock_ret = returns_override[t]
            bond_ret = r + econ_params.mu_bond
            cash_ret = r

            w_b = target_bond / (target_bond + target_cash) * (1 - w_stock) if (target_bond + target_cash) > 0 else 0
            w_c = 1 - w_stock - w_b

            portfolio_return = w_stock * stock_ret + w_b * bond_ret + w_c * cash_ret
            financial_wealth[t + 1] = investable * (1 + portfolio_return)
        elif not defaulted:
            # Last period: still record the weight based on fw
            if fw > 1e-6:
                w_stock = target_fin_stock / fw
                w_stock = max(0, min(1, w_stock))
            else:
                w_stock = target_stock
            stock_weight[t] = w_stock

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
    1. Median returns (baseline) - expected returns every year
    2. Early crash (sequence risk - bad) - 30% crash at retirement start
    3. Late crash (sequence risk - less bad) - 30% crash 10 years into retirement
    4. Bull market - 15% returns every year
    5. High volatility - alternating +20% and -10% returns

    Args:
        params: Lifecycle parameters
        econ_params: Economic parameters

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


def create_scenario_from_simulation_result(
    result: 'SimulationResult',
    name: str = None,
    description: str = None,
) -> ScenarioResult:
    """
    Convert a SimulationResult to a ScenarioResult for backward compatibility.

    Args:
        result: SimulationResult to convert
        name: Override name (defaults to result.strategy_name)
        description: Override description (defaults to result.description)

    Returns:
        ScenarioResult with equivalent data
    """
    return ScenarioResult(
        name=name or result.strategy_name,
        description=description or result.description,
        ages=result.ages,
        financial_wealth=result.financial_wealth,
        total_consumption=result.consumption,
        stock_weight=result.stock_weight,
        stock_returns=result.stock_returns,
        cumulative_consumption=result.cumulative_consumption,
    )
