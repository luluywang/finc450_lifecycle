"""
Economic primitives for lifecycle investment strategy.

This module contains core economic functions including:
- Bond pricing under mean-reverting term structure
- Present value and duration calculations
- Mean-variance optimization
- Correlated shock generation
- Interest rate and stock return simulation
"""

import numpy as np
from typing import Tuple

from .params import EconomicParams


# =============================================================================
# Zero-Coupon Bond Pricing Under Mean Reversion
# =============================================================================

def effective_duration(tau: float, phi: float) -> float:
    """
    Effective duration of a zero-coupon bond under mean-reverting rates.

    For the AR(1) process r_{t+1} = r_bar + phi(r_t - r_bar) + eps, the sensitivity
    of a tau-year zero-coupon bond price to the current short rate is:

        B(tau) = (1 - phi^tau) / (1 - phi)

    This is LESS than tau because mean reversion anchors long-term rates.

    As tau -> infinity, B(tau) -> 1/(1-phi). With phi=0.85, max duration ~= 6.67 years.

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

    Under the expectations hypothesis, the tau-period spot rate is the average
    of expected future short rates. Given:
        E_t[r_{t+k}] = r_bar + phi^k(r_t - r_bar)

    The price is:
        P(tau) = exp(-tau*r_bar - B(tau)*(r - r_bar))

    where B(tau) = (1 - phi^tau)/(1 - phi) is the effective duration.

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
    Spot rate (yield) for a tau-year zero-coupon bond.

    y(tau) = r_bar + (r - r_bar) * B(tau)/tau

    As tau -> infinity, y -> r_bar (long rates converge to long-run mean)
    As tau -> 0, y -> r (short rates equal current rate)
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
    - Buy a tau-year zero at time t
    - Sell a (tau-1)-year zero at time t+1
    - Return = P(t+1, tau-1) / P(t, tau) - 1

    Args:
        rates: Interest rate paths of shape (n_sims, n_periods + 1)
        tau: Target maturity of the zero-coupon bond
        econ_params: Economic parameters (r_bar, phi)

    Returns:
        Array of shape (n_sims, n_periods) with bond returns
    """
    r_bar = econ_params.r_bar
    phi = econ_params.phi

    # Price at start of each period (maturity = tau)
    P_start = zero_coupon_price_vectorized(rates[:, :-1], tau, r_bar, phi)

    # Price at end of each period (maturity = tau - 1, rate = r_{t+1})
    P_end = zero_coupon_price_vectorized(rates[:, 1:], tau - 1, r_bar, phi)

    # Return = P_end / P_start - 1
    returns = P_end / P_start - 1

    return returns


def compute_duration_approx_returns(
    rates: np.ndarray,
    duration: float,
    econ_params: EconomicParams
) -> np.ndarray:
    """
    Compute bond returns using duration approximation.

    Uses the classic duration formula:
        bond_return ≈ yield - duration × Δr

    where Δr = r_{t+1} - r_t is the change in interest rates.

    This is simpler than the exact zero-coupon formula and avoids
    dealing with specific maturities.

    Args:
        rates: Interest rate paths of shape (n_sims, n_periods + 1)
        duration: Duration of the bond portfolio
        econ_params: Economic parameters (for mu_bond spread)

    Returns:
        Array of shape (n_sims, n_periods) with bond returns
    """
    # Rate at start of period
    r_t = rates[:, :-1]

    # Rate change over period
    delta_r = rates[:, 1:] - rates[:, :-1]

    # Duration approximation: yield - duration × Δr
    # Note: mu_bond (spread) is added separately by caller
    returns = r_t - duration * delta_r

    return returns


# =============================================================================
# Present Value and Duration Calculations
# =============================================================================

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



def compute_pv_consumption_realized(consumption: np.ndarray, rates: np.ndarray) -> float:
    """
    Compute PV of consumption using realized rate path.

    Instead of using a constant discount rate, this uses the actual realized
    interest rate path to compute present value. This is more accurate for
    comparing strategies under stochastic interest rates.

    Formula: PV = sum(C_t / prod_{s=0}^{t-1}(1 + r_s))

    Args:
        consumption: Array of consumption values over time
        rates: Array of realized interest rates over time

    Returns:
        Present value of total lifetime consumption at time 0
    """
    T = len(consumption)
    r = rates[:T] if len(rates) > T else rates

    # Cumulative discount factors
    discount_factors = np.ones(T)
    cumulative_growth = 1.0
    for t in range(1, T):
        cumulative_growth *= (1 + r[t-1])
        discount_factors[t] = cumulative_growth

    return np.sum(consumption / discount_factors)


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
        PV = sum C * P(t) where P(t) = exp(-t*r_bar - B(t)*(r - r_bar))

    Otherwise falls back to flat discount rate:
        PV = C * [1 - (1+r)^(-T)] / r
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
        D_eff = (1/PV) * sum C * P(t) * B(t)

    where B(t) = (1 - phi^t)/(1 - phi) is the effective duration of a t-year zero.

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
# Mean-Variance Optimization
# =============================================================================

def compute_full_merton_allocation(
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

    where:
    - mu is the vector of excess returns [mu_stock, mu_bond]
    - Sigma is the variance-covariance matrix of returns

    Asset return models:
    - Stock: R_s = r + mu_stock + sigma_s * eps_s
    - Bond:  R_b = r + mu_bond - D * sigma_r * eps_r
      (negative sign because rising rates hurt bond prices)

    The VCV matrix is:
    Sigma = [ sigma_s^2                   -D * sigma_s * sigma_r * rho ]
            [ -D * sigma_s * sigma_r * rho  (D * sigma_r)^2            ]

    Args:
        mu_stock: Stock excess return (equity risk premium)
        mu_bond: Bond excess return (risk premium over short rate)
        sigma_s: Stock return volatility
        sigma_r: Interest rate shock volatility
        rho: Correlation between rate shocks and stock return shocks
        duration: Effective duration of the bond portfolio
        gamma: Risk aversion coefficient

    Returns:
        Tuple of (stock_weight, bond_weight, cash_weight) summing to 1.0
    """
    if gamma <= 0:
        raise ValueError("Risk aversion gamma must be positive for MV optimization")

    # Bond return volatility from duration and rate volatility
    sigma_b = duration * sigma_r

    # Covariance between stock and bond returns
    # Cov(R_s, R_b) = Cov(sigma_s * eps_s, -D * sigma_r * eps_r)
    #               = -D * sigma_s * sigma_r * rho
    cov_sb = -duration * sigma_s * sigma_r * rho

    # Build variance-covariance matrix
    # Sigma = [[var_s, cov_sb], [cov_sb, var_b]]
    var_s = sigma_s ** 2
    var_b = sigma_b ** 2

    # Compute determinant for matrix inversion
    det = var_s * var_b - cov_sb ** 2

    if abs(det) < 1e-12:
        # Near-singular matrix: fall back to single-asset solution
        stock_weight = mu_stock / (gamma * var_s)
        bond_weight = mu_bond / (gamma * var_b) if var_b > 1e-12 else 0.0
    else:
        # Inverse of 2x2 matrix: [[a, b], [c, d]]^(-1) = (1/det) * [[d, -b], [-c, a]]
        inv_00 = var_b / det
        inv_01 = -cov_sb / det
        inv_10 = -cov_sb / det
        inv_11 = var_s / det

        # Optimal weights: w* = (1/gamma) * Sigma^(-1) * mu
        stock_weight = (inv_00 * mu_stock + inv_01 * mu_bond) / gamma
        bond_weight = (inv_10 * mu_stock + inv_11 * mu_bond) / gamma

    # Cash weight is the remainder
    cash_weight = 1.0 - stock_weight - bond_weight

    return stock_weight, bond_weight, cash_weight


def compute_full_merton_allocation_constrained(
    mu_stock: float,
    mu_bond: float,
    sigma_s: float,
    sigma_r: float,
    rho: float,
    duration: float,
    gamma: float
) -> Tuple[float, float, float]:
    """
    Compute optimal portfolio allocation with no-short-selling constraints.

    Same as compute_full_merton_allocation but applies constraints:
    - No short selling (all weights >= 0)
    - No leverage (all weights <= 1)
    - Weights sum to 1

    Args:
        mu_stock: Stock excess return (equity risk premium)
        mu_bond: Bond excess return (risk premium over short rate)
        sigma_s: Stock return volatility
        sigma_r: Interest rate shock volatility
        rho: Correlation between rate shocks and stock return shocks
        duration: Effective duration of the bond portfolio
        gamma: Risk aversion coefficient

    Returns:
        Tuple of (stock_weight, bond_weight, cash_weight) summing to 1.0
    """
    # Get unconstrained solution
    w_stock, w_bond, w_cash = compute_full_merton_allocation(
        mu_stock, mu_bond, sigma_s, sigma_r, rho, duration, gamma
    )

    # Apply no-short constraint for each asset
    w_stock = max(0.0, w_stock)
    w_bond = max(0.0, w_bond)
    w_cash = max(0.0, w_cash)

    # Normalize to sum to 1.0
    total = w_stock + w_bond + w_cash
    if total > 0:
        w_stock /= total
        w_bond /= total
        w_cash /= total
    else:
        # Edge case: all weights were negative, default to cash
        w_stock, w_bond, w_cash = 0.0, 0.0, 1.0

    return w_stock, w_bond, w_cash


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

    This is an alias for compute_full_merton_allocation_constrained for convenience.

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


# =============================================================================
# Shock Generation and Return Simulation
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

    return rates


def simulate_interest_rates_random_walk(
    r0: float,
    n_periods: int,
    n_sims: int,
    sigma_r: float,
    drift: float,
    rate_shocks: np.ndarray,
) -> np.ndarray:
    """
    Simulate interest rates following a random walk process.

    r_{t+1} = r_t + drift + sigma_r * epsilon_r

    This is a benchmark model without mean reversion.

    Args:
        r0: Initial interest rate
        n_periods: Number of periods to simulate
        n_sims: Number of simulation paths
        sigma_r: Volatility of rate shocks
        drift: Drift term (expected change per period)
        rate_shocks: Standard normal shocks of shape (n_sims, n_periods)

    Returns:
        Array of shape (n_sims, n_periods + 1) with rate paths
    """
    rates = np.zeros((n_sims, n_periods + 1))
    rates[:, 0] = r0

    for t in range(n_periods):
        rates[:, t + 1] = rates[:, t] + drift + sigma_r * rate_shocks[:, t]

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
    # Stock return = risk-free rate + equity premium + shock
    stock_returns = (
        rates[:, :-1]  # r_t at start of period
        + params.mu_excess
        + params.sigma_s * stock_shocks
    )

    return stock_returns


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
