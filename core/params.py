"""
Core parameter dataclasses for lifecycle investment strategy.

This module contains all parameter dataclasses used throughout the lifecycle
investment system, consolidated from various modules into a single source of truth.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING, Protocol, runtime_checkable
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

    # Portfolio constraint parameters
    allow_leverage: bool = False     # Allow shorting and leverage in portfolio

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
    """
    DEPRECATED: Use SimulationResult with strategy_name='RuleOfThumb' instead.

    Results from rule-of-thumb strategy simulation.
    """
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
class SimulationResult:
    """
    Unified, strategy-agnostic simulation result.

    Works for both:
    - Single simulation (deterministic median path): arrays are 1D [n_periods]
    - Monte Carlo: arrays are 2D [n_sims, n_periods]

    The strategy is an INPUT to simulation, not part of the OUTPUT type.
    Any strategy (LDI, RuleOfThumb, Fixed, custom) produces this same result format.
    """
    # Identification
    strategy_name: str
    ages: np.ndarray                    # [n_periods] age at each time step

    # Core wealth and consumption paths
    # Shape: [n_periods] for single sim, [n_sims, n_periods] for MC
    financial_wealth: np.ndarray
    consumption: np.ndarray
    subsistence_consumption: np.ndarray
    variable_consumption: np.ndarray

    # Portfolio allocation weights (sum to 1.0)
    stock_weight: np.ndarray
    bond_weight: np.ndarray
    cash_weight: np.ndarray

    # Market conditions used in simulation
    interest_rates: np.ndarray
    stock_returns: np.ndarray

    # Default tracking
    # For single sim: bool/int scalars; for MC: [n_sims] arrays
    defaulted: np.ndarray               # True if ran out of money
    default_age: np.ndarray             # Age at default (NaN if no default)
    final_wealth: np.ndarray            # Terminal financial wealth

    # Optional metadata for scenarios
    description: str = ""

    @property
    def n_sims(self) -> int:
        """Number of simulations (1 for deterministic, N for Monte Carlo)."""
        if self.financial_wealth.ndim == 1:
            return 1
        return self.financial_wealth.shape[0]

    @property
    def n_periods(self) -> int:
        """Number of time periods."""
        if self.financial_wealth.ndim == 1:
            return len(self.financial_wealth)
        return self.financial_wealth.shape[1]

    @property
    def is_monte_carlo(self) -> bool:
        """True if this contains multiple simulation paths."""
        return self.n_sims > 1

    @property
    def cumulative_consumption(self) -> np.ndarray:
        """Cumulative consumption over time."""
        if self.is_monte_carlo:
            return np.cumsum(self.consumption, axis=1)
        return np.cumsum(self.consumption)

    def percentile(self, field: str, percentiles: List[int]) -> np.ndarray:
        """
        Compute percentiles for a field across simulations.

        Args:
            field: Name of field (e.g., 'financial_wealth', 'consumption')
            percentiles: List of percentiles to compute (e.g., [5, 25, 50, 75, 95])

        Returns:
            Array of shape [n_percentiles, n_periods]
        """
        data = getattr(self, field)
        if not self.is_monte_carlo:
            # Single sim: return the same data for all percentiles
            return np.tile(data, (len(percentiles), 1))
        return np.percentile(data, percentiles, axis=0)


@dataclass
class StrategyComparison:
    """
    Comparison of two strategies using identical market conditions.

    Simply holds two SimulationResult objects that used the same shocks.
    Percentiles and summary statistics are computed on demand via properties.
    """
    result_a: SimulationResult
    result_b: SimulationResult

    # Strategy parameters for display (optional)
    strategy_a_params: dict = field(default_factory=dict)
    strategy_b_params: dict = field(default_factory=dict)

    @property
    def ages(self) -> np.ndarray:
        return self.result_a.ages

    @property
    def n_sims(self) -> int:
        return self.result_a.n_sims

    def default_rate(self, which: str = 'a') -> float:
        """Default rate for strategy 'a' or 'b'."""
        result = self.result_a if which == 'a' else self.result_b
        return float(np.mean(result.defaulted))

    def median_final_wealth(self, which: str = 'a') -> float:
        """Median final wealth for strategy 'a' or 'b'."""
        result = self.result_a if which == 'a' else self.result_b
        return float(np.median(result.final_wealth))

    def wealth_percentiles(self, which: str = 'a',
                           percentiles: List[int] = None) -> np.ndarray:
        """Wealth percentiles for strategy 'a' or 'b'."""
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        result = self.result_a if which == 'a' else self.result_b
        return result.percentile('financial_wealth', percentiles)

    def consumption_percentiles(self, which: str = 'a',
                                percentiles: List[int] = None) -> np.ndarray:
        """Consumption percentiles for strategy 'a' or 'b'."""
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        result = self.result_a if which == 'a' else self.result_b
        return result.percentile('consumption', percentiles)


@dataclass
class ScenarioResult:
    """
    DEPRECATED: Use SimulationResult with description field instead.

    Results from a teaching scenario simulation.
    """
    name: str
    description: str
    ages: np.ndarray
    financial_wealth: np.ndarray
    total_consumption: np.ndarray
    stock_weight: np.ndarray
    stock_returns: np.ndarray
    cumulative_consumption: np.ndarray


# =============================================================================
# Generic Strategy Framework
# =============================================================================

@dataclass
class SimulationState:
    """
    State available to strategy at each time step.

    This provides all the information a strategy needs to make decisions,
    including wealth measures, cash flows, market conditions, and precomputed
    hedge components for LDI-style strategies.
    """
    # Time indices
    t: int                          # Current period (0-indexed)
    age: int                        # Current age
    is_working: bool                # True if before retirement

    # Wealth measures
    financial_wealth: float         # Current FW
    human_capital: float            # PV of future earnings
    pv_expenses: float              # PV of future expenses
    net_worth: float                # HC + FW - PV(expenses)
    total_wealth: float             # HC + FW

    # Cash flows
    earnings: float                 # Current period earnings
    expenses: float                 # Current period expenses (subsistence)

    # Market state
    current_rate: float             # Current interest rate

    # HC decomposition (for LDI hedge)
    hc_stock_component: float
    hc_bond_component: float
    hc_cash_component: float

    # Expense liability decomposition (for LDI hedge)
    exp_bond_component: float
    exp_cash_component: float

    # Durations
    duration_hc: float
    duration_expenses: float

    # Target allocations (from MV optimization)
    target_stock: float
    target_bond: float
    target_cash: float

    # Parameters (read-only reference)
    params: 'LifecycleParams'
    econ_params: 'EconomicParams'


@dataclass
class StrategyActions:
    """
    Actions returned by strategy for current period.

    Contains both consumption decisions and portfolio allocation weights.
    Weights should sum to 1.0 (or close to it for leveraged strategies).
    """
    # Consumption
    total_consumption: float
    subsistence_consumption: float
    variable_consumption: float

    # Portfolio weights (should sum to 1.0)
    stock_weight: float
    bond_weight: float
    cash_weight: float


@runtime_checkable
class StrategyProtocol(Protocol):
    """
    Protocol for strategies that map state to actions.

    Strategies are simple functions that take the current simulation state
    and return actions (consumption and portfolio weights). This allows
    strategies to be easily swapped and compared using a generic simulation engine.

    Example:
        >>> class MyStrategy:
        ...     name: str = "My Strategy"
        ...     def __call__(self, state: SimulationState) -> StrategyActions:
        ...         return StrategyActions(...)
    """
    name: str

    def __call__(self, state: SimulationState) -> StrategyActions:
        """Compute actions given current state."""
        ...
