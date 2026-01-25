"""
Core parameter dataclasses for lifecycle investment strategy.

This module contains all parameter dataclasses used throughout the lifecycle
investment system, consolidated from various modules into a single source of truth.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .simulation import LifecycleResult


# =============================================================================
# Economic Environment Parameters
# =============================================================================

@dataclass
class EconomicParams:
    """Parameters for the economic environment (VAR structure)."""
    r_bar: float = 0.02        # Long-run mean real rate
    phi: float = 1.0           # Interest rate persistence (1.0 = random walk)
    sigma_r: float = 0.006     # Rate shock volatility (was 0.012)
    mu_excess: float = 0.04    # Equity risk premium (stock excess return)
    bond_sharpe: float = 0.037  # Bond Sharpe ratio (replaces fixed mu_bond)
    sigma_s: float = 0.18      # Stock return volatility
    rho: float = 0.0           # Correlation between rate and stock shocks (was -0.2)
    r_floor: float = 0.001     # Minimum interest rate (0.1%)
    bond_duration: float = 20.0 # Duration for HC decomposition and MV optimization

    @property
    def mu_bond(self) -> float:
        """Bond excess return = Sharpe * volatility, where vol = duration * sigma_r."""
        return self.bond_sharpe * self.bond_duration * self.sigma_r


@dataclass
class BondParams:
    """Parameters for bond instruments (maturity in years)."""
    D_mm: float = 0.25         # Money market maturity (approximately 3 months)
    D_lb: float = 15.0         # Long bond maturity (15-year zero)


@dataclass
class RandomWalkParams:
    """Parameters for random walk interest rate model."""
    sigma_r: float = 0.012     # Rate shock volatility
    drift: float = 0.0         # Expected rate change per period (0 = pure random walk)
    r_floor: float = 0.001     # Minimum interest rate


# =============================================================================
# Lifecycle Model Parameters
# =============================================================================

@dataclass
class LifecycleParams:
    """Parameters for lifecycle model."""
    # Age parameters
    start_age: int = 25          # Age at career start
    retirement_age: int = 65     # Age at retirement
    end_age: int = 95            # Planning horizon (death or planning end)

    # Income parameters (in $000s for cleaner numbers)
    initial_earnings: float = 150    # Starting annual earnings ($150k)
    earnings_growth: float = 0.0     # Real earnings growth rate (flat)
    earnings_hump_age: int = 65      # Age at peak earnings (at retirement = flat)
    earnings_decline: float = 0.0    # Decline rate after peak

    # Expense parameters (subsistence/baseline)
    base_expenses: float = 100       # Base annual subsistence expenses ($100k)
    expense_growth: float = 0.0      # Real expense growth rate (flat)
    retirement_expenses: float = 100 # Retirement subsistence expenses ($100k)

    # Consumption parameters
    consumption_share: float = 0.05  # Share of net worth consumed above subsistence
    consumption_boost: float = 0.0   # Boost above median return for consumption rate

    # Asset allocation parameters
    stock_beta_human_capital: float = 0.0    # Beta of human capital to stocks

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
    initial_wealth: float = 100      # Starting financial wealth ($100k, negative allowed)


@dataclass
class MonteCarloParams:
    """Parameters for Monte Carlo simulation."""
    n_simulations: int = 1000    # Number of simulation paths
    random_seed: int = 42        # Random seed for reproducibility


# =============================================================================
# Simulation Enums and Parameters (from retirement_simulation)
# =============================================================================

class BondStrategy(Enum):
    """Bond allocation strategy."""
    MONEY_MARKET = "mm"        # All bonds in money market
    DURATION_MATCH = "dm"      # Duration-matched to liabilities


class ConsumptionRule(Enum):
    """Consumption rule."""
    FIXED = "fixed"            # Fixed annual consumption
    VARIABLE = "variable"      # Percentage of wealth


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


@dataclass
class SimulationParams:
    """Parameters for the retirement simulation."""
    initial_wealth: float = 2_500_000
    annual_consumption: float = 100_000
    horizon: int = 30          # Years in retirement
    stock_weight: float = 0.40 # Fixed stock allocation
    n_simulations: int = 10_000
    random_seed: int = 42


# =============================================================================
# Result Dataclasses
# =============================================================================

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
    savings_rate: float  # 0.15
    withdrawal_rate: float  # 0.04
    target_duration: float = 6.0  # Target FI duration
    subsistence_consumption: np.ndarray = None
    variable_consumption: np.ndarray = None


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


@dataclass
class MedianPathComparisonResult:
    """Results from comparing LDI vs Rule-of-Thumb on deterministic median paths."""
    ages: np.ndarray

    # LDI (optimal) strategy paths
    ldi_financial_wealth: np.ndarray
    ldi_total_consumption: np.ndarray
    ldi_stock_weight: np.ndarray
    ldi_bond_weight: np.ndarray
    ldi_cash_weight: np.ndarray
    ldi_human_capital: np.ndarray
    ldi_net_worth: np.ndarray

    # Rule-of-thumb strategy paths
    rot_financial_wealth: np.ndarray
    rot_total_consumption: np.ndarray
    rot_stock_weight: np.ndarray
    rot_bond_weight: np.ndarray
    rot_cash_weight: np.ndarray

    # Earnings (same for both)
    earnings: np.ndarray

    # PV consumption at time 0
    ldi_pv_consumption: float
    rot_pv_consumption: float

    # Strategy parameters for display
    rot_savings_rate: float
    rot_target_duration: float
    rot_withdrawal_rate: float


@dataclass
class SimulationResult:
    """Results from a single strategy simulation (retirement_simulation style)."""
    strategy_name: str
    wealth_paths: np.ndarray          # (n_sims, n_periods + 1)
    consumption_paths: np.ndarray     # (n_sims, n_periods)
    defaulted: np.ndarray             # (n_sims,) boolean
    default_year: np.ndarray          # (n_sims,) year of default or -1
    total_consumption: np.ndarray     # (n_sims,)
    final_wealth: np.ndarray          # (n_sims,)


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
