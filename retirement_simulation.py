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
    """Parameters for bond instruments (maturity in years)."""
    D_mm: float = 0.25         # Money market maturity (≈ 3 months)
    D_lb: float = 15.0         # Long bond maturity (15-year zero)


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


def simulate_interest_rates_random_walk(
    r0: float,
    n_periods: int,
    n_sims: int,
    sigma_r: float,
    drift: float,
    rate_shocks: np.ndarray,
    r_floor: float = 0.001
) -> np.ndarray:
    """
    Simulate interest rates following a random walk process.

    r_{t+1} = r_t + drift + sigma_r * epsilon_r

    This is a benchmark model without mean reversion, useful for comparison.
    When drift = 0, this is a pure random walk (martingale).

    Args:
        r0: Initial interest rate
        n_periods: Number of periods to simulate
        n_sims: Number of simulation paths
        sigma_r: Volatility of rate shocks
        drift: Drift term (expected change per period)
        rate_shocks: Standard normal shocks of shape (n_sims, n_periods)
        r_floor: Minimum interest rate floor

    Returns:
        Array of shape (n_sims, n_periods + 1) with rate paths
    """
    rates = np.zeros((n_sims, n_periods + 1))
    rates[:, 0] = r0

    for t in range(n_periods):
        rates[:, t + 1] = rates[:, t] + drift + sigma_r * rate_shocks[:, t]
        # Floor rates at minimum
        rates[:, t + 1] = np.maximum(rates[:, t + 1], r_floor)

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
# Zero-Coupon Bond Pricing Under Mean Reversion
# =============================================================================

def effective_duration(tau: float, phi: float) -> float:
    """
    Effective duration of a zero-coupon bond under mean-reverting rates.

    For the AR(1) process r_{t+1} = r̄ + φ(r_t - r̄) + ε, the sensitivity
    of a τ-year zero-coupon bond price to the current short rate is:

        B(τ) = (1 - φ^τ) / (1 - φ)

    This is LESS than τ because mean reversion anchors long-term rates.

    As τ → ∞, B(τ) → 1/(1-φ). With φ=0.85, max duration ≈ 6.67 years.

    Args:
        tau: Time to maturity in years
        phi: Mean reversion parameter (persistence)

    Returns:
        Effective duration (sensitivity to short rate changes)
    """
    if tau <= 0:
        return 0.0
    if abs(phi - 1.0) < 1e-10:
        return tau  # No mean reversion case
    return (1 - phi**tau) / (1 - phi)


def effective_duration_vectorized(tau: np.ndarray, phi: float) -> np.ndarray:
    """Vectorized version of effective_duration."""
    if abs(phi - 1.0) < 1e-10:
        return tau.copy()
    return (1 - phi**tau) / (1 - phi)


def zero_coupon_price(r: float, tau: float, r_bar: float, phi: float) -> float:
    """
    Price of a zero-coupon bond under the discrete-time Vasicek model.

    Under the expectations hypothesis, the τ-period spot rate is the average
    of expected future short rates. Given:
        E_t[r_{t+k}] = r̄ + φ^k(r_t - r̄)

    The price is:
        P(τ) = exp(-τ·r̄ - B(τ)·(r - r̄))

    where B(τ) = (1 - φ^τ)/(1 - φ) is the effective duration.

    Args:
        r: Current short rate
        tau: Time to maturity
        r_bar: Long-run mean rate
        phi: Mean reversion parameter

    Returns:
        Bond price (between 0 and 1)
    """
    if tau <= 0:
        return 1.0
    B = effective_duration(tau, phi)
    return np.exp(-tau * r_bar - B * (r - r_bar))


def zero_coupon_price_vectorized(
    r: np.ndarray,
    tau: float,
    r_bar: float,
    phi: float
) -> np.ndarray:
    """Vectorized version of zero_coupon_price for arrays of rates."""
    if tau <= 0:
        return np.ones_like(r)
    B = effective_duration(tau, phi)
    return np.exp(-tau * r_bar - B * (r - r_bar))


def spot_rate(r: float, tau: float, r_bar: float, phi: float) -> float:
    """
    Spot rate (yield) for a τ-year zero-coupon bond.

    y(τ) = r̄ + (r - r̄) · B(τ)/τ

    As τ → ∞, y → r̄ (long rates converge to long-run mean)
    As τ → 0, y → r (short rates equal current rate)
    """
    if tau <= 0:
        return r
    B = effective_duration(tau, phi)
    return r_bar + (r - r_bar) * B / tau


def compute_zero_coupon_returns(
    rates: np.ndarray,
    tau: float,
    econ_params: EconomicParams
) -> np.ndarray:
    """
    Compute returns on a zero-coupon bond strategy (constant maturity).

    At each period:
    - Buy a τ-year zero at time t
    - Sell a (τ-1)-year zero at time t+1
    - Return = P(t+1, τ-1) / P(t, τ) - 1

    For a rolling strategy with constant maturity τ, we assume rebalancing
    to maintain maturity τ each period.

    Args:
        rates: Interest rate paths of shape (n_sims, n_periods + 1)
        tau: Target maturity of the zero-coupon bond
        econ_params: Economic parameters (r_bar, phi)

    Returns:
        Array of shape (n_sims, n_periods) with bond returns
    """
    r_bar = econ_params.r_bar
    phi = econ_params.phi

    n_sims, n_periods_plus_1 = rates.shape
    n_periods = n_periods_plus_1 - 1

    # Price at start of each period (maturity = tau)
    P_start = zero_coupon_price_vectorized(rates[:, :-1], tau, r_bar, phi)

    # Price at end of each period (maturity = tau - 1, rate = r_{t+1})
    P_end = zero_coupon_price_vectorized(rates[:, 1:], tau - 1, r_bar, phi)

    # Return = P_end / P_start - 1
    returns = P_end / P_start - 1

    return returns


# =============================================================================
# Liability Calculations (Consistent with Mean-Reverting Term Structure)
# =============================================================================

def liability_pv(
    consumption: float,
    rate: float,
    years_remaining: int,
    r_bar: float = None,
    phi: float = None
) -> float:
    """
    Calculate present value of liability stream.

    If r_bar and phi are provided, uses the mean-reverting term structure:
        PV = Σ C × P(t) where P(t) = exp(-t·r̄ - B(t)·(r - r̄))

    Otherwise falls back to flat discount rate:
        PV = C × [1 - (1+r)^(-T)] / r
    """
    if years_remaining <= 0:
        return 0.0

    # Use mean-reverting term structure if parameters provided
    if r_bar is not None and phi is not None:
        pv = 0.0
        for t in range(1, years_remaining + 1):
            pv += consumption * zero_coupon_price(rate, t, r_bar, phi)
        return pv

    # Fallback to flat rate discounting
    if rate < 1e-10:
        return consumption * years_remaining
    return consumption * (1 - (1 + rate) ** (-years_remaining)) / rate


def liability_pv_vectorized(
    consumption: float,
    rates: np.ndarray,
    years_remaining: int,
    r_bar: float = None,
    phi: float = None
) -> np.ndarray:
    """Vectorized version of liability_pv for arrays of rates."""
    if years_remaining <= 0:
        return np.zeros_like(rates)

    # Use mean-reverting term structure if parameters provided
    if r_bar is not None and phi is not None:
        pv = np.zeros_like(rates)
        for t in range(1, years_remaining + 1):
            pv += consumption * zero_coupon_price_vectorized(rates, t, r_bar, phi)
        return pv

    # Fallback to flat rate discounting
    result = np.where(
        rates < 1e-10,
        consumption * years_remaining,
        consumption * (1 - (1 + rates) ** (-years_remaining)) / rates
    )
    return result


def liability_duration(
    consumption: float,
    rate: float,
    years_remaining: int,
    r_bar: float = None,
    phi: float = None
) -> float:
    """
    Calculate duration of liability stream.

    If r_bar and phi provided, returns EFFECTIVE duration (sensitivity to
    short rate) under mean reversion:
        D_eff = (1/PV) × Σ C × P(t) × B(t)

    where B(t) = (1 - φ^t)/(1 - φ) is the effective duration of a t-year zero.

    Otherwise returns traditional modified duration.
    """
    if years_remaining <= 0:
        return 0.0

    # Use effective duration under mean reversion
    if r_bar is not None and phi is not None:
        pv = liability_pv(consumption, rate, years_remaining, r_bar, phi)
        if pv < 1e-10:
            return 0.0

        weighted_sum = 0.0
        for t in range(1, years_remaining + 1):
            P_t = zero_coupon_price(rate, t, r_bar, phi)
            B_t = effective_duration(t, phi)
            weighted_sum += consumption * P_t * B_t

        return weighted_sum / pv

    # Fallback to traditional modified duration
    pv = liability_pv(consumption, rate, years_remaining)
    if pv < 1e-10:
        return 0.0

    weighted_sum = 0.0
    for t in range(1, years_remaining + 1):
        weighted_sum += t * consumption / ((1 + rate) ** (t + 1))

    return weighted_sum / pv


def liability_duration_vectorized(
    consumption: float,
    rates: np.ndarray,
    years_remaining: int,
    r_bar: float = None,
    phi: float = None
) -> np.ndarray:
    """Vectorized version of liability_duration."""
    if years_remaining <= 0:
        return np.zeros_like(rates)

    # Use effective duration under mean reversion
    if r_bar is not None and phi is not None:
        pv = liability_pv_vectorized(consumption, rates, years_remaining, r_bar, phi)

        weighted_sums = np.zeros_like(rates)
        for t in range(1, years_remaining + 1):
            P_t = zero_coupon_price_vectorized(rates, t, r_bar, phi)
            B_t = effective_duration(t, phi)
            weighted_sums += consumption * P_t * B_t

        return np.where(pv > 1e-10, weighted_sums / pv, 0.0)

    # Fallback to traditional modified duration
    pv = liability_pv_vectorized(consumption, rates, years_remaining)

    weighted_sums = np.zeros_like(rates)
    for t in range(1, years_remaining + 1):
        weighted_sums += t * consumption / ((1 + rates) ** (t + 1))

    return np.where(pv > 1e-10, weighted_sums / pv, 0.0)


# =============================================================================
# Portfolio Allocation
# =============================================================================

def compute_bond_weights(
    stock_weight: float,
    liability_eff_duration: float,
    bond_params: BondParams,
    strategy: BondStrategy,
    phi: float = 0.85
) -> Tuple[float, float]:
    """
    Compute money market and long bond weights.

    For duration matching under mean reversion:
        - liability_eff_duration: effective duration of liabilities (sensitivity to short rate)
        - Long bond effective duration: B(τ_lb) = (1 - φ^τ)/(1 - φ)

    We match: w_lb × B(τ_lb) = liability_eff_duration

    Returns:
        Tuple of (w_mm, w_lb)
    """
    bond_allocation = 1 - stock_weight

    if strategy == BondStrategy.MONEY_MARKET:
        return bond_allocation, 0.0

    # Duration matching using EFFECTIVE durations
    # The long bond's effective duration (sensitivity to short rate) under mean reversion
    lb_eff_duration = effective_duration(bond_params.D_lb, phi)

    # We need: w_lb × lb_eff_duration = liability_eff_duration (for the bond portion)
    target_lb_weight = min(liability_eff_duration / lb_eff_duration, 1.0)
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
        # (sensitivity to short rate under mean reversion)
        liab_dur = liability_duration_vectorized(
            sim_params.annual_consumption,
            current_rate,
            years_remaining,
            r_bar=econ_params.r_bar,
            phi=econ_params.phi
        )

        # Compute portfolio weights using effective durations
        # For simplicity, use mean liability duration across sims
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

    # Compute bond returns using consistent zero-coupon pricing
    # Returns reflect that long rates don't move 1-for-1 with short rates under mean reversion
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
# =============================================================================

@dataclass
class LifecycleParams:
    """Parameters for lifecycle asset allocation with Human Capital."""
    current_age: int = 25
    retirement_age: int = 65
    life_expectancy: int = 90
    current_wage: float = 250_000
    wage_growth: float = 0.00      # Real wage growth rate
    subsistence: float = 100_000   # Mandatory spending (inflation-adjusted)
    beta_labor: float = 0.0        # Labor income correlation with stocks
                                   # 0.0 = bond-like (professor, consultant)
                                   # 1.0 = stock-like (tech founder, banker)
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

    Args:
        age: Current age
        wage: Current annual wage
        retirement_age: Age at retirement
        beta_labor: Correlation of labor income with stock market
        subsistence: Mandatory annual spending
        wage_growth: Real wage growth rate
        rf: Risk-free rate
        mkt_excess: Equity risk premium

    Returns:
        Present value of future net labor income
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


def get_merton_share(mkt_excess: float, sigma: float, gamma: float) -> float:
    """
    Calculate the Merton optimal share of risky assets in TOTAL wealth.

    The Merton share is: (mu - rf) / (gamma * sigma^2)

    This represents the theoretically optimal fraction of total wealth
    (Human Capital + Financial Wealth) to hold in risky assets.

    Args:
        mkt_excess: Equity risk premium (expected excess return)
        sigma: Standard deviation of stock returns
        gamma: Risk aversion coefficient

    Returns:
        Optimal fraction of total wealth in stocks
    """
    variance = sigma ** 2
    target_share = mkt_excess / (gamma * variance)
    return target_share


def solve_financial_stock_allocation(
    fin_wealth: float,
    human_capital: float,
    beta_labor: float,
    target_total_share: float
) -> float:
    """
    Solve for the optimal stock allocation in the financial portfolio.

    Given:
    - Total Wealth = Financial Wealth + Human Capital
    - Target Stock Dollars = Total Wealth * Merton Share
    - Implicit Stock in HC = Human Capital * beta_labor

    The financial portfolio must fill the gap:
    Required Financial Stock = Target Stock - Implicit Stock in HC

    Constraints: No leverage (cap at 100%), no shorting (floor at 0%)

    Args:
        fin_wealth: Current financial wealth
        human_capital: Current human capital value
        beta_labor: Labor income beta (stock-likeness of HC)
        target_total_share: Merton share (target % of total wealth in stocks)

    Returns:
        Optimal stock allocation for financial portfolio (0 to 1)
    """
    total_wealth = fin_wealth + human_capital

    if total_wealth <= 0:
        return 0.5  # Default allocation if no wealth

    # Target stock dollars across entire balance sheet
    target_stock_dollars = total_wealth * target_total_share

    # Implicit stock exposure from Human Capital
    # If beta=0, HC is bond-like (no implicit stock)
    # If beta=1, HC is stock-like (all implicit stock)
    implicit_stock_in_hc = human_capital * beta_labor

    # Required stock purchase in financial portfolio
    required_financial_stock = target_stock_dollars - implicit_stock_in_hc

    # Calculate weight in financial portfolio
    if fin_wealth <= 0:
        return 1.0  # If broke, theoretical 100% stocks

    optimal_weight = required_financial_stock / fin_wealth

    # Apply leverage constraints (no borrowing, no shorting)
    constrained_weight = np.clip(optimal_weight, 0.0, 1.0)

    return constrained_weight


@dataclass
class LifecycleResult:
    """Results from lifecycle simulation."""
    ages: np.ndarray                    # Age at each period
    human_capital: np.ndarray           # Human Capital over time
    financial_wealth: np.ndarray        # Financial wealth over time
    total_wealth: np.ndarray            # Total wealth (HC + FW)
    stock_allocation: np.ndarray        # Stock % in financial portfolio
    bond_allocation: np.ndarray         # Bond % in financial portfolio
    consumption: np.ndarray             # Consumption each period
    savings: np.ndarray                 # Savings each period
    merton_share: float                 # Target total wealth stock share


def run_lifecycle_simulation(
    lifecycle_params: LifecycleParams,
    econ_params: EconomicParams,
    use_random_walk: bool = False,
    rw_params: 'RandomWalkParams' = None
) -> LifecycleResult:
    """
    Run a deterministic lifecycle simulation showing the glide path.

    This implements the Total Wealth framework where:
    - Human Capital is valued as PV of future net labor income
    - Financial portfolio allocation adjusts to achieve target total wealth risk
    - Creates the characteristic "hump-shaped" or "flat-then-falling" equity path

    Args:
        lifecycle_params: Lifecycle model parameters
        econ_params: Economic environment parameters
        use_random_walk: Whether to use random walk for rate evolution
        rw_params: Random walk parameters (if use_random_walk=True)

    Returns:
        LifecycleResult with allocation path over lifecycle
    """
    lp = lifecycle_params
    ep = econ_params

    # Use current rate as risk-free rate
    rf = ep.r_bar
    mkt_excess = ep.mu_excess
    sigma = ep.sigma_s

    # Calculate Merton share (target for total wealth)
    merton_share = get_merton_share(mkt_excess, sigma, lp.gamma)

    # Initialize arrays
    n_periods = lp.life_expectancy - lp.current_age + 1
    ages = np.arange(lp.current_age, lp.life_expectancy + 1)
    human_capital = np.zeros(n_periods)
    financial_wealth = np.zeros(n_periods)
    total_wealth = np.zeros(n_periods)
    stock_allocation = np.zeros(n_periods)
    bond_allocation = np.zeros(n_periods)
    consumption = np.zeros(n_periods)
    savings = np.zeros(n_periods)

    # Initial conditions
    fin_wealth = lp.initial_wealth

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
            mkt_excess=mkt_excess
        )

        # Solve for optimal stock allocation in financial portfolio
        stock_pct = solve_financial_stock_allocation(
            fin_wealth=fin_wealth,
            human_capital=hc,
            beta_labor=lp.beta_labor,
            target_total_share=merton_share
        )

        # Store data
        human_capital[i] = hc
        financial_wealth[i] = fin_wealth
        total_wealth[i] = hc + fin_wealth
        stock_allocation[i] = stock_pct
        bond_allocation[i] = 1.0 - stock_pct

        # Determine consumption and savings
        if age < lp.retirement_age:
            # Working years: earn wage, spend subsistence, save the rest
            annual_consumption = lp.subsistence
            annual_savings = lp.current_wage - lp.subsistence
        else:
            # Retirement: consume from wealth
            years_remaining = lp.life_expectancy - age
            if years_remaining > 0:
                # Simple consumption rule: spend proportion of wealth
                annual_consumption = fin_wealth / (years_remaining + 5)
            else:
                annual_consumption = fin_wealth
            annual_savings = -annual_consumption  # Negative savings = spending down

        consumption[i] = annual_consumption
        savings[i] = annual_savings

        # Simulate next year's wealth (deterministic: use expected returns)
        if i < n_periods - 1:
            # Portfolio return = rf + stock_pct * excess return
            port_return = rf + stock_pct * mkt_excess

            if age < lp.retirement_age:
                # Accumulation: wealth grows + savings
                fin_wealth = fin_wealth * (1 + port_return) + annual_savings
            else:
                # Decumulation: wealth grows - consumption
                fin_wealth = fin_wealth * (1 + port_return) - annual_consumption

            fin_wealth = max(fin_wealth, 0)

    return LifecycleResult(
        ages=ages,
        human_capital=human_capital,
        financial_wealth=financial_wealth,
        total_wealth=total_wealth,
        stock_allocation=stock_allocation,
        bond_allocation=bond_allocation,
        consumption=consumption,
        savings=savings,
        merton_share=merton_share
    )


# =============================================================================
# Random Walk Benchmark Simulation
# =============================================================================

@dataclass
class RandomWalkParams:
    """Parameters for random walk interest rate model."""
    sigma_r: float = 0.012     # Rate shock volatility
    drift: float = 0.0         # Expected rate change per period (0 = pure random walk)
    r_floor: float = 0.001     # Minimum interest rate


@dataclass
class MedianPathResult:
    """Results from a median-path (deterministic) simulation."""
    strategy_name: str
    years: np.ndarray                  # (n_periods + 1,) year indices
    rates: np.ndarray                  # (n_periods + 1,) interest rate path
    wealth: np.ndarray                 # (n_periods + 1,) wealth path
    consumption: np.ndarray            # (n_periods,) consumption each period
    stock_weight: np.ndarray           # (n_periods,) stock allocation
    mm_weight: np.ndarray              # (n_periods,) money market allocation
    lb_weight: np.ndarray              # (n_periods,) long bond allocation
    liability_pv: np.ndarray           # (n_periods + 1,) present value of liabilities
    funded_ratio: np.ndarray           # (n_periods + 1,) funded status


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

    For normally distributed shocks, median = mean = 0, so this computes the
    expected path with no uncertainty. This is useful for understanding the
    baseline trajectory of wealth and allocation.

    Args:
        strategy: Strategy to simulate
        r0: Initial interest rate
        sim_params: Simulation parameters
        bond_params: Bond parameters
        econ_params: Economic parameters (for mean-reverting model)
        rw_params: Random walk parameters (if use_random_walk=True)
        use_random_walk: If True, use random walk model; else use mean-reverting

    Returns:
        MedianPathResult with deterministic path and allocations
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
        r_bar_for_pricing = r0  # Use initial rate as "anchor"
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
            # Mean-reverting: r_{t+1} = r_bar + phi * (r_t - r_bar) + 0
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

        # Compute expected returns (median = expected for symmetric distributions)
        # Stock: r_t + mu_excess (no shock)
        expected_stock_return = current_rate + econ_params.mu_excess

        # Bond returns at median (zero shock means rate doesn't change unexpectedly)
        # For money market: approximately equal to current short rate
        # For long bond: yield on the bond
        expected_mm_return = current_rate

        # Long bond return under median scenario
        # Price appreciates as maturity shortens, plus any yield
        if use_random_walk:
            # Under random walk, long rates don't anchor, so duration = maturity
            expected_lb_return = current_rate + bond_params.D_lb * 0  # Just yield
        else:
            # Under mean reversion, compute expected return
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
        else:  # Variable
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

    This serves as a benchmark to compare against the mean-reverting model.
    Under a random walk, there is no anchor for rates, so duration effects
    are more pronounced.

    Returns:
        Tuple of:
        - Dictionary mapping strategy names to SimulationResult
        - Interest rate paths (n_sims, n_periods + 1)
        - Stock returns (n_sims, n_periods)
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

    # Simulate stock returns (still uses rates for risk-free component)
    stock_returns = simulate_stock_returns(rates, econ_params, stock_shocks)

    # For random walk, use phi=1 for bond pricing (no mean reversion)
    # Create modified economic params with phi=1
    rw_econ_params = EconomicParams(
        r_bar=initial_rate,  # Use initial rate as reference
        phi=1.0,             # No mean reversion
        sigma_r=rw_params.sigma_r,
        mu_excess=econ_params.mu_excess,
        sigma_s=econ_params.sigma_s,
        rho=econ_params.rho,
        r_floor=rw_params.r_floor
    )

    # Compute bond returns
    # Under random walk, bond pricing is different - use traditional duration
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


def compute_funded_ratio(
    wealth: np.ndarray,
    rates: np.ndarray,
    consumption_target: float,
    years_remaining: int,
    r_bar: float = None,
    phi: float = None
) -> np.ndarray:
    """
    Compute funded ratio = Assets / PV(Liabilities)

    If r_bar and phi provided, uses mean-reverting term structure for liability PV.
    """
    liab_pv = liability_pv_vectorized(
        consumption_target, rates, years_remaining, r_bar=r_bar, phi=phi
    )
    # Avoid division by zero
    return np.where(liab_pv > 0, wealth / liab_pv, np.inf)
