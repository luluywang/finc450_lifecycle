"""
Retirement Simulation: Monte Carlo Analysis of Life Cycle Investing Strategies

This module implements a Monte Carlo simulation comparing retirement spending strategies,
focusing on duration matching and variable consumption approaches.

Author: FINC 450 Life Cycle Investing

.. deprecated::
    This module is deprecated in favor of the `core` module.
    - For dataclasses, use `from core import LifecycleParams, EconomicParams, ...`
    - For simulation functions, use `from core import run_lifecycle_monte_carlo, ...`
    - For visualization, use `from visualization import ...`

    This module is maintained for backward compatibility but may be removed in a future release.
"""

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List

warnings.warn(
    "The retirement_simulation module is deprecated. "
    "Use `from core import ...` for dataclasses and simulation functions, "
    "and `from visualization import ...` for plotting.",
    DeprecationWarning,
    stacklevel=2
)

# =============================================================================
# Re-export from core for backward compatibility
# =============================================================================

from core import (
    # Dataclasses
    EconomicParams, BondParams, SimulationParams, RandomWalkParams,
    BondStrategy, ConsumptionRule, Strategy, STRATEGIES,
    SimulationResult, MedianPathResult,
    # Economic functions
    generate_correlated_shocks, simulate_interest_rates,
    simulate_interest_rates_random_walk, simulate_stock_returns,
    effective_duration, effective_duration_vectorized,
    zero_coupon_price, zero_coupon_price_vectorized, spot_rate,
    compute_zero_coupon_returns, liability_pv, liability_pv_vectorized,
    liability_duration, liability_duration_vectorized,
    compute_bond_weights, compute_portfolio_return,
    compute_full_merton_allocation, compute_full_merton_allocation_constrained,
    compute_funded_ratio,
)


# =============================================================================
# Consumption Rules (unique to this module)
# =============================================================================

def fixed_consumption(
    wealth: float,
    target: float,
    years_remaining: int
) -> float:
    """Fixed consumption: spend target or remaining wealth if insufficient."""
    return min(target, wealth)


def variable_consumption(
    wealth: float,
    target: float,
    years_remaining: int,
    cap_multiplier: float = 1.5
) -> float:
    """
    Variable consumption: alpha_t * W_t where alpha_t = 1 / (T - t + 5)
    Cap at 1.5 * target to prevent excessive spending.
    """
    if years_remaining <= 0:
        return wealth

    alpha = 1.0 / (years_remaining + 5)
    consumption = alpha * wealth

    # Cap at multiple of target
    return min(consumption, cap_multiplier * target)


# =============================================================================
# Simulation Engine (unique to this module)
# =============================================================================

def run_single_simulation(
    strategy: Strategy,
    rates: np.ndarray,
    stock_returns: np.ndarray,
    mm_returns: np.ndarray,
    lb_returns: np.ndarray,
    sim_params: SimulationParams,
    bond_params: BondParams,
    econ_params: EconomicParams
) -> SimulationResult:
    """
    Run simulation for a single strategy across all paths.

    Args:
        strategy: Strategy to simulate
        rates: Interest rate paths (n_sims, n_periods + 1)
        stock_returns: Stock returns (n_sims, n_periods)
        mm_returns: Money market returns (n_sims, n_periods)
        lb_returns: Long bond returns (n_sims, n_periods)
        sim_params: Simulation parameters
        bond_params: Bond parameters
        econ_params: Economic parameters (for consistent term structure)

    Returns:
        SimulationResult with all paths and metrics
    """
    n_sims = sim_params.n_simulations
    n_periods = sim_params.horizon

    # Initialize tracking arrays
    wealth_paths = np.zeros((n_sims, n_periods + 1))
    consumption_paths = np.zeros((n_sims, n_periods))
    wealth_paths[:, 0] = sim_params.initial_wealth

    defaulted = np.zeros(n_sims, dtype=bool)
    default_year = np.full(n_sims, -1)

    # Simulate year by year
    for t in range(n_periods):
        years_remaining = n_periods - t
        current_wealth = wealth_paths[:, t]
        current_rate = rates[:, t]

        # Compute liability EFFECTIVE duration for this period
        liab_dur = liability_duration_vectorized(
            sim_params.annual_consumption,
            current_rate,
            years_remaining,
            r_bar=econ_params.r_bar,
            phi=econ_params.phi
        )

        # Compute portfolio weights using effective durations
        mean_liab_dur = np.mean(liab_dur)
        w_mm, w_lb = compute_bond_weights(
            sim_params.stock_weight,
            mean_liab_dur,
            bond_params,
            strategy.bond_strategy,
            phi=econ_params.phi
        )
        w_s = sim_params.stock_weight

        # Compute portfolio returns
        port_returns = compute_portfolio_return(
            stock_returns[:, t],
            mm_returns[:, t],
            lb_returns[:, t],
            w_s, w_mm, w_lb
        )

        # Determine consumption
        if strategy.consumption_rule == ConsumptionRule.FIXED:
            consumption = np.minimum(
                sim_params.annual_consumption * np.ones(n_sims),
                current_wealth
            )
        else:  # Variable
            alpha = 1.0 / (years_remaining + 5)
            consumption = alpha * current_wealth
            consumption = np.minimum(consumption, 1.5 * sim_params.annual_consumption)

        consumption_paths[:, t] = consumption

        # Update wealth: grow then consume
        wealth_after_return = current_wealth * (1 + port_returns)
        new_wealth = wealth_after_return - consumption

        # Track defaults (wealth hits zero)
        newly_defaulted = (new_wealth <= 0) & ~defaulted
        default_year[newly_defaulted] = t + 1
        defaulted |= newly_defaulted

        # Floor wealth at zero
        wealth_paths[:, t + 1] = np.maximum(new_wealth, 0)

    return SimulationResult(
        strategy_name=str(strategy),
        wealth_paths=wealth_paths,
        consumption_paths=consumption_paths,
        defaulted=defaulted,
        default_year=default_year,
        total_consumption=consumption_paths.sum(axis=1),
        final_wealth=wealth_paths[:, -1]
    )


def run_monte_carlo(
    strategies: List[Strategy] = None,
    econ_params: EconomicParams = None,
    bond_params: BondParams = None,
    sim_params: SimulationParams = None,
    initial_rate: float = None
) -> Tuple[Dict[str, SimulationResult], np.ndarray, np.ndarray]:
    """
    Run full Monte Carlo simulation for all strategies.

    Returns:
        Tuple of:
        - Dictionary mapping strategy names to SimulationResult
        - Interest rate paths (n_sims, n_periods + 1)
        - Stock returns (n_sims, n_periods)
    """
    # Use defaults if not provided
    if strategies is None:
        strategies = STRATEGIES
    if econ_params is None:
        econ_params = EconomicParams()
    if bond_params is None:
        bond_params = BondParams()
    if sim_params is None:
        sim_params = SimulationParams()
    if initial_rate is None:
        initial_rate = econ_params.r_bar

    # Set random seed
    rng = np.random.default_rng(sim_params.random_seed)

    n_sims = sim_params.n_simulations
    n_periods = sim_params.horizon

    # Generate correlated shocks
    rate_shocks, stock_shocks = generate_correlated_shocks(
        n_periods, n_sims, econ_params.rho, rng
    )

    # Simulate interest rates
    rates = simulate_interest_rates(
        initial_rate, n_periods, n_sims, econ_params, rate_shocks
    )

    # Simulate stock returns
    stock_returns = simulate_stock_returns(rates, econ_params, stock_shocks)

    # Compute bond returns using consistent zero-coupon pricing
    mm_returns = compute_zero_coupon_returns(rates, bond_params.D_mm, econ_params)
    lb_returns = compute_zero_coupon_returns(rates, bond_params.D_lb, econ_params)

    # Run each strategy
    results = {}
    for strategy in strategies:
        result = run_single_simulation(
            strategy, rates, stock_returns, mm_returns, lb_returns,
            sim_params, bond_params, econ_params
        )
        results[str(strategy)] = result

    return results, rates, stock_returns


# =============================================================================
# Lifecycle Asset Allocation Model (Human Capital Framework)
# Unique to this module - uses different parameter structure
# =============================================================================

@dataclass
class LifecycleParams:
    """
    Parameters for lifecycle asset allocation with Human Capital.

    Note: This is a different structure from core.LifecycleParams, designed
    for the human capital framework in this module.
    """
    current_age: int = 25
    retirement_age: int = 65
    life_expectancy: int = 90
    current_wage: float = 250_000
    wage_growth: float = 0.00      # Real wage growth rate
    subsistence: float = 100_000   # Mandatory spending (inflation-adjusted)
    beta_labor: float = 0.0        # Labor income correlation with stocks
    gamma: float = 2.0             # Risk aversion coefficient
    initial_wealth: float = 10_000 # Starting financial wealth


def calculate_human_capital(
    age: int,
    wage: float,
    retirement_age: int,
    beta_labor: float,
    subsistence: float,
    wage_growth: float,
    rf: float,
    mkt_excess: float
) -> float:
    """
    Calculate the Present Value of future net labor income (Human Capital).

    Human Capital is discounted at a risk-adjusted rate:
    r_labor = rf + beta_labor * market_excess_return

    If beta_labor = 0, Human Capital behaves like a bond.
    If beta_labor = 1, Human Capital behaves like a stock.
    """
    if age >= retirement_age:
        return 0.0

    # Risk-adjusted discount rate for Human Capital
    r_labor = rf + beta_labor * mkt_excess

    # Sum discounted future net income
    pv_wages = 0.0
    years_to_work = retirement_age - age

    for t in range(1, years_to_work + 1):
        # Projected net income (wage - subsistence)
        net_income = (wage - subsistence) * ((1 + wage_growth) ** (t - 1))
        # Discount back to today
        pv_wages += net_income / ((1 + r_labor) ** t)

    return max(pv_wages, 0.0)


@dataclass
class LifecycleResult:
    """
    Results from lifecycle simulation.

    Note: This is a different structure from core.LifecycleResult, designed
    for the human capital framework in this module.
    """
    ages: np.ndarray                    # Age at each period
    human_capital: np.ndarray           # Human Capital over time
    financial_wealth: np.ndarray        # Financial wealth over time
    total_wealth: np.ndarray            # Total wealth (HC + FW)
    stock_allocation: np.ndarray        # Stock % in financial portfolio
    bond_allocation: np.ndarray         # Bond % in financial portfolio
    cash_allocation: np.ndarray         # Cash % in financial portfolio
    consumption: np.ndarray             # Consumption each period
    savings: np.ndarray                 # Savings each period
    target_stock: float                 # Target stock weight from MV optimization
    target_bond: float                  # Target bond weight from MV optimization
    target_cash: float                  # Target cash weight from MV optimization


def run_lifecycle_simulation(
    lifecycle_params: LifecycleParams,
    econ_params: EconomicParams,
    use_random_walk: bool = False,
    rw_params: RandomWalkParams = None
) -> LifecycleResult:
    """
    Run a deterministic lifecycle simulation showing the glide path.

    This implements the Total Wealth framework where:
    - Human Capital is valued as PV of future net labor income
    - Financial portfolio allocation adjusts to achieve target total wealth risk
    - Uses 3-asset allocation: stocks, bonds, and cash
    """
    lp = lifecycle_params
    ep = econ_params

    rf = ep.r_bar

    # Calculate target allocation using full Merton VCV solution
    target_stock, target_bond, target_cash = compute_full_merton_allocation_constrained(
        mu_stock=ep.mu_excess,
        mu_bond=ep.mu_bond,
        sigma_s=ep.sigma_s,
        sigma_r=ep.sigma_r,
        rho=ep.rho,
        duration=ep.bond_duration,
        gamma=lp.gamma
    )

    # Initialize arrays
    n_periods = lp.life_expectancy - lp.current_age + 1
    ages = np.arange(lp.current_age, lp.life_expectancy + 1)
    human_capital = np.zeros(n_periods)
    financial_wealth = np.zeros(n_periods)
    total_wealth = np.zeros(n_periods)
    stock_allocation = np.zeros(n_periods)
    bond_allocation = np.zeros(n_periods)
    cash_allocation = np.zeros(n_periods)
    consumption = np.zeros(n_periods)
    savings = np.zeros(n_periods)

    # Initial conditions
    fin_wealth = lp.initial_wealth

    # Duration benchmark for splitting HC fixed income between bonds and cash
    duration_benchmark = ep.bond_duration

    for i, age in enumerate(ages):
        # Calculate Human Capital
        hc = calculate_human_capital(
            age=age,
            wage=lp.current_wage if age < lp.retirement_age else 0,
            retirement_age=lp.retirement_age,
            beta_labor=lp.beta_labor,
            subsistence=lp.subsistence,
            wage_growth=lp.wage_growth,
            rf=rf,
            mkt_excess=ep.mu_excess
        )

        # Decompose human capital into stock/bond/cash components
        hc_stock = hc * lp.beta_labor
        hc_non_stock = hc * (1.0 - lp.beta_labor)

        # Duration of human capital (approximate: years to retirement / 2)
        years_to_retire = max(0, lp.retirement_age - age)
        hc_duration = years_to_retire / 2.0 if years_to_retire > 0 else 0.0

        # Split non-stock HC between bonds and cash based on duration
        if duration_benchmark > 0 and hc_non_stock > 0:
            bond_fraction = min(1.0, hc_duration / duration_benchmark)
            hc_bond = hc_non_stock * bond_fraction
            hc_cash = hc_non_stock * (1.0 - bond_fraction)
        else:
            hc_bond = 0.0
            hc_cash = hc_non_stock

        # Total wealth
        tw = hc + fin_wealth

        # Target total holdings
        target_total_stocks = target_stock * tw
        target_total_bonds = target_bond * tw
        target_total_cash = target_cash * tw

        # Target financial holdings = Total target - Human capital component
        target_fin_stocks = target_total_stocks - hc_stock
        target_fin_bonds = target_total_bonds - hc_bond
        target_fin_cash = target_total_cash - hc_cash

        # Compute financial portfolio weights with no-short constraint
        if fin_wealth > 1e-6:
            w_stock = target_fin_stocks / fin_wealth
            w_bond = target_fin_bonds / fin_wealth
            w_cash = target_fin_cash / fin_wealth

            # Aggregate into equity and fixed income
            equity = w_stock
            fixed_income = w_bond + w_cash

            # No-short at aggregate level
            equity = max(0, equity)
            fixed_income = max(0, fixed_income)

            # Normalize
            total_agg = equity + fixed_income
            if total_agg > 0:
                equity = equity / total_agg
                fixed_income = fixed_income / total_agg
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

            stock_pct = equity
            bond_pct = w_b
            cash_pct = w_c
        else:
            stock_pct = target_stock
            bond_pct = target_bond
            cash_pct = target_cash

        # Store data
        human_capital[i] = hc
        financial_wealth[i] = fin_wealth
        total_wealth[i] = tw
        stock_allocation[i] = stock_pct
        bond_allocation[i] = bond_pct
        cash_allocation[i] = cash_pct

        # Determine consumption and savings
        if age < lp.retirement_age:
            annual_consumption = lp.subsistence
            annual_savings = lp.current_wage - lp.subsistence
        else:
            years_remaining = lp.life_expectancy - age
            if years_remaining > 0:
                annual_consumption = fin_wealth / (years_remaining + 5)
            else:
                annual_consumption = fin_wealth
            annual_savings = -annual_consumption

        consumption[i] = annual_consumption
        savings[i] = annual_savings

        # Simulate next year's wealth (deterministic: use expected returns)
        if i < n_periods - 1:
            stock_return = rf + ep.mu_excess
            bond_return = rf + ep.mu_bond
            cash_return = rf
            port_return = (stock_pct * stock_return +
                          bond_pct * bond_return +
                          cash_pct * cash_return)

            if age < lp.retirement_age:
                fin_wealth = fin_wealth * (1 + port_return) + annual_savings
            else:
                fin_wealth = fin_wealth * (1 + port_return) - annual_consumption

            fin_wealth = max(fin_wealth, 0)

    return LifecycleResult(
        ages=ages,
        human_capital=human_capital,
        financial_wealth=financial_wealth,
        total_wealth=total_wealth,
        stock_allocation=stock_allocation,
        bond_allocation=bond_allocation,
        cash_allocation=cash_allocation,
        consumption=consumption,
        savings=savings,
        target_stock=target_stock,
        target_bond=target_bond,
        target_cash=target_cash
    )


# =============================================================================
# Median Path Simulation (unique to this module)
# =============================================================================

def run_median_path_simulation(
    strategy: Strategy,
    r0: float,
    sim_params: SimulationParams,
    bond_params: BondParams,
    econ_params: EconomicParams = None,
    rw_params: RandomWalkParams = None,
    use_random_walk: bool = False
) -> MedianPathResult:
    """
    Run a single deterministic simulation where median returns are realized each period.
    """
    if econ_params is None:
        econ_params = EconomicParams()
    if rw_params is None:
        rw_params = RandomWalkParams()

    n_periods = sim_params.horizon

    # Initialize arrays
    rates = np.zeros(n_periods + 1)
    wealth = np.zeros(n_periods + 1)
    consumption = np.zeros(n_periods)
    stock_weight = np.zeros(n_periods)
    mm_weight = np.zeros(n_periods)
    lb_weight = np.zeros(n_periods)
    liab_pv_path = np.zeros(n_periods + 1)
    funded_ratio = np.zeros(n_periods + 1)

    # Initial conditions
    rates[0] = r0
    wealth[0] = sim_params.initial_wealth

    # For random walk, we use phi=1 for bond pricing (no mean reversion)
    if use_random_walk:
        phi_for_pricing = 1.0
        r_bar_for_pricing = r0
    else:
        phi_for_pricing = econ_params.phi
        r_bar_for_pricing = econ_params.r_bar

    # Compute initial liability PV and funded ratio
    liab_pv_path[0] = liability_pv(
        sim_params.annual_consumption, rates[0], n_periods,
        r_bar=r_bar_for_pricing, phi=phi_for_pricing
    )
    funded_ratio[0] = wealth[0] / liab_pv_path[0] if liab_pv_path[0] > 0 else np.inf

    # Simulate year by year with median (zero) shocks
    for t in range(n_periods):
        years_remaining = n_periods - t
        current_rate = rates[t]
        current_wealth = wealth[t]

        # Update interest rate for next period (with zero shock = median)
        if use_random_walk:
            rates[t + 1] = current_rate + rw_params.drift
        else:
            rates[t + 1] = econ_params.r_bar + econ_params.phi * (current_rate - econ_params.r_bar)

        # Floor rates
        rates[t + 1] = max(rates[t + 1], econ_params.r_floor if not use_random_walk else rw_params.r_floor)

        # Compute liability effective duration
        liab_dur = liability_duration(
            sim_params.annual_consumption,
            current_rate,
            years_remaining,
            r_bar=r_bar_for_pricing,
            phi=phi_for_pricing
        )

        # Compute portfolio weights
        w_mm, w_lb = compute_bond_weights(
            sim_params.stock_weight,
            liab_dur,
            bond_params,
            strategy.bond_strategy,
            phi=phi_for_pricing
        )
        w_s = sim_params.stock_weight

        stock_weight[t] = w_s
        mm_weight[t] = w_mm
        lb_weight[t] = w_lb

        # Compute expected returns
        expected_stock_return = current_rate + econ_params.mu_excess
        expected_mm_return = current_rate

        if use_random_walk:
            expected_lb_return = current_rate
        else:
            expected_lb_return = spot_rate(current_rate, bond_params.D_lb, r_bar_for_pricing, phi_for_pricing)

        # Portfolio return
        port_return = (
            w_s * expected_stock_return +
            w_mm * expected_mm_return +
            w_lb * expected_lb_return
        )

        # Consumption
        if strategy.consumption_rule == ConsumptionRule.FIXED:
            cons = min(sim_params.annual_consumption, current_wealth)
        else:
            alpha = 1.0 / (years_remaining + 5)
            cons = min(alpha * current_wealth, 1.5 * sim_params.annual_consumption)

        consumption[t] = cons

        # Update wealth
        wealth_after_return = current_wealth * (1 + port_return)
        wealth[t + 1] = max(wealth_after_return - cons, 0)

        # Compute liability PV and funded ratio for next period
        if years_remaining > 1:
            liab_pv_path[t + 1] = liability_pv(
                sim_params.annual_consumption, rates[t + 1], years_remaining - 1,
                r_bar=r_bar_for_pricing, phi=phi_for_pricing
            )
            funded_ratio[t + 1] = wealth[t + 1] / liab_pv_path[t + 1] if liab_pv_path[t + 1] > 0 else np.inf
        else:
            liab_pv_path[t + 1] = 0
            funded_ratio[t + 1] = np.inf

    return MedianPathResult(
        strategy_name=str(strategy),
        years=np.arange(n_periods + 1),
        rates=rates,
        wealth=wealth,
        consumption=consumption,
        stock_weight=stock_weight,
        mm_weight=mm_weight,
        lb_weight=lb_weight,
        liability_pv=liab_pv_path,
        funded_ratio=funded_ratio
    )


def run_random_walk_monte_carlo(
    strategies: List[Strategy] = None,
    econ_params: EconomicParams = None,
    bond_params: BondParams = None,
    sim_params: SimulationParams = None,
    rw_params: RandomWalkParams = None,
    initial_rate: float = None
) -> Tuple[Dict[str, SimulationResult], np.ndarray, np.ndarray]:
    """
    Run Monte Carlo simulation using random walk interest rate model.
    """
    if strategies is None:
        strategies = STRATEGIES
    if econ_params is None:
        econ_params = EconomicParams()
    if bond_params is None:
        bond_params = BondParams()
    if sim_params is None:
        sim_params = SimulationParams()
    if rw_params is None:
        rw_params = RandomWalkParams()
    if initial_rate is None:
        initial_rate = econ_params.r_bar

    # Set random seed
    rng = np.random.default_rng(sim_params.random_seed)

    n_sims = sim_params.n_simulations
    n_periods = sim_params.horizon

    # Generate correlated shocks
    rate_shocks, stock_shocks = generate_correlated_shocks(
        n_periods, n_sims, econ_params.rho, rng
    )

    # Simulate interest rates using random walk
    rates = simulate_interest_rates_random_walk(
        initial_rate, n_periods, n_sims,
        rw_params.sigma_r, rw_params.drift, rate_shocks, rw_params.r_floor
    )

    # Simulate stock returns
    stock_returns = simulate_stock_returns(rates, econ_params, stock_shocks)

    # Create modified economic params with phi=1
    rw_econ_params = EconomicParams(
        r_bar=initial_rate,
        phi=1.0,
        sigma_r=rw_params.sigma_r,
        mu_excess=econ_params.mu_excess,
        bond_sharpe=econ_params.bond_sharpe,
        sigma_s=econ_params.sigma_s,
        rho=econ_params.rho,
        r_floor=rw_params.r_floor,
        bond_duration=econ_params.bond_duration
    )

    # Compute bond returns
    mm_returns = compute_zero_coupon_returns(rates, bond_params.D_mm, rw_econ_params)
    lb_returns = compute_zero_coupon_returns(rates, bond_params.D_lb, rw_econ_params)

    # Run each strategy
    results = {}
    for strategy in strategies:
        result = run_single_simulation(
            strategy, rates, stock_returns, mm_returns, lb_returns,
            sim_params, bond_params, rw_econ_params
        )
        results[str(strategy)] = result

    return results, rates, stock_returns


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_summary_stats(results: Dict[str, SimulationResult]) -> pd.DataFrame:
    """Compute summary statistics for each strategy."""
    stats = []

    for name, result in results.items():
        stats.append({
            'Strategy': name,
            'Default Rate (%)': 100 * result.defaulted.mean(),
            'Avg Total Consumption ($k)': result.total_consumption.mean() / 1000,
            'Consumption Volatility ($k)': result.consumption_paths.std(axis=1).mean() / 1000,
            'Avg Final Wealth ($k)': result.final_wealth.mean() / 1000,
            'Median Final Wealth ($k)': np.median(result.final_wealth) / 1000,
            '10th Pctl Final Wealth ($k)': np.percentile(result.final_wealth, 10) / 1000,
            '90th Pctl Final Wealth ($k)': np.percentile(result.final_wealth, 90) / 1000,
        })

    return pd.DataFrame(stats).set_index('Strategy')
