"""
Retirement Simulation: Monte Carlo Analysis of Life Cycle Investing Strategies

This module implements a Monte Carlo simulation comparing retirement spending strategies,
focusing on duration matching and variable consumption approaches.

Author: FINC 450 Life Cycle Investing
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List
from enum import Enum


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class EconomicParams:
    """Parameters for the economic environment (VAR structure)."""
    r_bar: float = 0.03        # Long-run mean real rate
    phi: float = 0.85          # Interest rate persistence
    sigma_r: float = 0.012     # Rate shock volatility
    mu_excess: float = 0.04    # Equity risk premium
    sigma_s: float = 0.18      # Stock return volatility
    rho: float = -0.2          # Correlation between rate and stock shocks
    r_floor: float = 0.001     # Minimum interest rate (0.1%)


@dataclass
class BondParams:
    """Parameters for bond instruments."""
    D_mm: float = 0.25         # Money market duration
    D_lb: float = 15.0         # Long bond duration


@dataclass
class SimulationParams:
    """Parameters for the simulation."""
    initial_wealth: float = 2_500_000
    annual_consumption: float = 100_000
    horizon: int = 30          # Years in retirement
    stock_weight: float = 0.40 # Fixed stock allocation
    n_simulations: int = 10_000
    random_seed: int = 42


class BondStrategy(Enum):
    """Bond allocation strategy."""
    MONEY_MARKET = "mm"        # All bonds in money market
    DURATION_MATCH = "dm"      # Duration-matched to liabilities


class ConsumptionRule(Enum):
    """Consumption rule."""
    FIXED = "fixed"            # Fixed annual consumption
    VARIABLE = "variable"      # Percentage of wealth


# =============================================================================
# Economic Environment
# =============================================================================

def generate_correlated_shocks(
    n_periods: int,
    n_sims: int,
    rho: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate correlated shocks for rates and stocks using Cholesky decomposition.

    Returns:
        Tuple of (rate_shocks, stock_shocks) each with shape (n_sims, n_periods)
    """
    # Correlation matrix
    corr_matrix = np.array([[1.0, rho], [rho, 1.0]])

    # Cholesky decomposition
    L = np.linalg.cholesky(corr_matrix)

    # Generate independent standard normal shocks
    z = rng.standard_normal((n_sims, n_periods, 2))

    # Apply correlation structure
    correlated = np.einsum('ijk,lk->ijl', z, L)

    return correlated[:, :, 0], correlated[:, :, 1]


def simulate_interest_rates(
    r0: float,
    n_periods: int,
    n_sims: int,
    params: EconomicParams,
    rate_shocks: np.ndarray
) -> np.ndarray:
    """
    Simulate interest rates following AR(1) process.

    r_{t+1} = r_bar + phi * (r_t - r_bar) + epsilon_r

    Returns:
        Array of shape (n_sims, n_periods + 1) with rate paths
    """
    rates = np.zeros((n_sims, n_periods + 1))
    rates[:, 0] = r0

    for t in range(n_periods):
        rates[:, t + 1] = (
            params.r_bar
            + params.phi * (rates[:, t] - params.r_bar)
            + params.sigma_r * rate_shocks[:, t]
        )
        # Floor rates at minimum
        rates[:, t + 1] = np.maximum(rates[:, t + 1], params.r_floor)

    return rates


def simulate_stock_returns(
    rates: np.ndarray,
    params: EconomicParams,
    stock_shocks: np.ndarray
) -> np.ndarray:
    """
    Simulate stock returns: R_stock = r_t + mu_excess + epsilon_s

    Returns:
        Array of shape (n_sims, n_periods) with stock returns
    """
    n_sims, n_periods_plus_1 = rates.shape
    n_periods = n_periods_plus_1 - 1

    # Stock return = risk-free rate + equity premium + shock
    stock_returns = (
        rates[:, :-1]  # r_t at start of period
        + params.mu_excess
        + params.sigma_s * stock_shocks
    )

    return stock_returns


# =============================================================================
# Bond Returns
# =============================================================================

def compute_bond_returns(
    rates: np.ndarray,
    duration: float
) -> np.ndarray:
    """
    Compute bond returns: R_bond = r_t - D * delta_r

    Args:
        rates: Interest rate paths of shape (n_sims, n_periods + 1)
        duration: Modified duration of the bond

    Returns:
        Array of shape (n_sims, n_periods) with bond returns
    """
    # Change in rates
    delta_r = np.diff(rates, axis=1)

    # Bond return = current rate - duration * rate change
    bond_returns = rates[:, :-1] - duration * delta_r

    return bond_returns


# =============================================================================
# Liability Calculations
# =============================================================================

def liability_pv(
    consumption: float,
    rate: float,
    years_remaining: int
) -> float:
    """
    Calculate present value of liability stream.

    PV = C * [1 - (1+r)^(-T)] / r
    """
    if years_remaining <= 0:
        return 0.0
    if rate < 1e-10:
        return consumption * years_remaining

    return consumption * (1 - (1 + rate) ** (-years_remaining)) / rate


def liability_pv_vectorized(
    consumption: float,
    rates: np.ndarray,
    years_remaining: int
) -> np.ndarray:
    """Vectorized version of liability_pv for arrays of rates."""
    if years_remaining <= 0:
        return np.zeros_like(rates)

    # Handle very small rates
    result = np.where(
        rates < 1e-10,
        consumption * years_remaining,
        consumption * (1 - (1 + rates) ** (-years_remaining)) / rates
    )
    return result


def liability_duration(
    consumption: float,
    rate: float,
    years_remaining: int
) -> float:
    """
    Calculate modified duration of liability stream.

    D = (1/PV) * sum_{t=1}^{T} [t * C / (1+r)^{t+1}]
    """
    if years_remaining <= 0:
        return 0.0

    pv = liability_pv(consumption, rate, years_remaining)
    if pv < 1e-10:
        return 0.0

    # Sum of weighted present values
    weighted_sum = 0.0
    for t in range(1, years_remaining + 1):
        weighted_sum += t * consumption / ((1 + rate) ** (t + 1))

    return weighted_sum / pv


def liability_duration_vectorized(
    consumption: float,
    rates: np.ndarray,
    years_remaining: int
) -> np.ndarray:
    """Vectorized version of liability_duration."""
    if years_remaining <= 0:
        return np.zeros_like(rates)

    pv = liability_pv_vectorized(consumption, rates, years_remaining)

    # Compute weighted sum for each rate
    weighted_sums = np.zeros_like(rates)
    for t in range(1, years_remaining + 1):
        weighted_sums += t * consumption / ((1 + rates) ** (t + 1))

    # Avoid division by zero
    result = np.where(pv > 1e-10, weighted_sums / pv, 0.0)
    return result


# =============================================================================
# Portfolio Allocation
# =============================================================================

def compute_bond_weights(
    stock_weight: float,
    liability_duration: float,
    bond_params: BondParams,
    strategy: BondStrategy
) -> Tuple[float, float]:
    """
    Compute money market and long bond weights.

    For duration matching:
        w_lb = min(D_liab / D_lb, 1) * (1 - w_s)
        w_mm = (1 - w_s) - w_lb

    Returns:
        Tuple of (w_mm, w_lb)
    """
    bond_allocation = 1 - stock_weight

    if strategy == BondStrategy.MONEY_MARKET:
        return bond_allocation, 0.0

    # Duration matching
    # We need: w_lb * D_lb = D_liab (for the bond portion)
    # But D_liab is for the full liability, so we match bond portfolio duration
    target_lb_weight = min(liability_duration / bond_params.D_lb, 1.0)
    w_lb = target_lb_weight * bond_allocation
    w_mm = bond_allocation - w_lb

    return w_mm, w_lb


def compute_portfolio_return(
    stock_returns: np.ndarray,
    mm_returns: np.ndarray,
    lb_returns: np.ndarray,
    w_s: float,
    w_mm: float,
    w_lb: float
) -> np.ndarray:
    """Compute weighted portfolio return."""
    return w_s * stock_returns + w_mm * mm_returns + w_lb * lb_returns


# =============================================================================
# Consumption Rules
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
# Strategy Definitions
# =============================================================================

@dataclass
class Strategy:
    """Definition of a retirement strategy."""
    name: str
    bond_strategy: BondStrategy
    consumption_rule: ConsumptionRule

    def __str__(self):
        bond_str = "MM" if self.bond_strategy == BondStrategy.MONEY_MARKET else "DurMatch"
        cons_str = "Fixed" if self.consumption_rule == ConsumptionRule.FIXED else "Variable"
        return f"{bond_str} + {cons_str}"


# Four strategies to compare
STRATEGIES = [
    Strategy("MM + Fixed", BondStrategy.MONEY_MARKET, ConsumptionRule.FIXED),
    Strategy("DurMatch + Fixed", BondStrategy.DURATION_MATCH, ConsumptionRule.FIXED),
    Strategy("MM + Variable", BondStrategy.MONEY_MARKET, ConsumptionRule.VARIABLE),
    Strategy("DurMatch + Variable", BondStrategy.DURATION_MATCH, ConsumptionRule.VARIABLE),
]


# =============================================================================
# Simulation Engine
# =============================================================================

@dataclass
class SimulationResult:
    """Results from a single strategy simulation."""
    strategy_name: str
    wealth_paths: np.ndarray          # (n_sims, n_periods + 1)
    consumption_paths: np.ndarray     # (n_sims, n_periods)
    defaulted: np.ndarray             # (n_sims,) boolean
    default_year: np.ndarray          # (n_sims,) year of default or -1
    total_consumption: np.ndarray     # (n_sims,)
    final_wealth: np.ndarray          # (n_sims,)


def run_single_simulation(
    strategy: Strategy,
    rates: np.ndarray,
    stock_returns: np.ndarray,
    mm_returns: np.ndarray,
    lb_returns: np.ndarray,
    sim_params: SimulationParams,
    bond_params: BondParams
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

        # Compute liability duration for this period
        liab_dur = liability_duration_vectorized(
            sim_params.annual_consumption,
            current_rate,
            years_remaining
        )

        # Compute portfolio weights (can vary by simulation due to rate-dependent duration)
        # For simplicity, use mean liability duration across sims
        mean_liab_dur = np.mean(liab_dur)
        w_mm, w_lb = compute_bond_weights(
            sim_params.stock_weight,
            mean_liab_dur,
            bond_params,
            strategy.bond_strategy
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
            # Cap at 1.5x target
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

    # Compute bond returns
    mm_returns = compute_bond_returns(rates, bond_params.D_mm)
    lb_returns = compute_bond_returns(rates, bond_params.D_lb)

    # Run each strategy
    results = {}
    for strategy in strategies:
        result = run_single_simulation(
            strategy, rates, stock_returns, mm_returns, lb_returns,
            sim_params, bond_params
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


def compute_funded_ratio(
    wealth: np.ndarray,
    rates: np.ndarray,
    consumption_target: float,
    years_remaining: int
) -> np.ndarray:
    """
    Compute funded ratio = Assets / PV(Liabilities)
    """
    liab_pv = liability_pv_vectorized(consumption_target, rates, years_remaining)
    # Avoid division by zero
    return np.where(liab_pv > 0, wealth / liab_pv, np.inf)
