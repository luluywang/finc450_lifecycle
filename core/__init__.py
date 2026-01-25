"""
Core module for lifecycle investment strategy.

This module provides the foundational components for lifecycle investment analysis:
- Parameter dataclasses (params.py)
- Economic primitives: bond pricing, PV calculations, MV optimization (economics.py)
- Simulation engines: Monte Carlo, strategy comparison (simulation.py)
"""

# Constants
from .params import DEFAULT_RISKY_BETA

# Parameter dataclasses
from .params import (
    EconomicParams,
    BondParams,
    RandomWalkParams,
    LifecycleParams,
    MonteCarloParams,
    BondStrategy,
    ConsumptionRule,
    Strategy,
    SimulationParams,
    STRATEGIES,
    # Result dataclasses
    LifecycleResult,
    MonteCarloResult,
    SimulationResult,       # New unified result type
    StrategyComparison,     # New unified comparison type
    # Deprecated (kept for backward compatibility)
    RuleOfThumbResult,
    ScenarioResult,
    # Generic strategy framework
    SimulationState,
    StrategyActions,
    StrategyProtocol,
)

# Strategy implementations
from .strategies import (
    LDIStrategy,
    RuleOfThumbStrategy,
    FixedConsumptionStrategy,
)

# Economic primitives
from .economics import (
    # Bond pricing
    effective_duration,
    effective_duration_vectorized,
    zero_coupon_price,
    zero_coupon_price_vectorized,
    spot_rate,
    compute_zero_coupon_returns,
    compute_duration_approx_returns,
    # Present value and duration
    compute_present_value,
    compute_pv_consumption,
    compute_pv_consumption_realized,
    compute_duration,
    liability_pv,
    liability_pv_vectorized,
    liability_duration,
    liability_duration_vectorized,
    # Mean-variance optimization
    compute_full_merton_allocation,
    compute_full_merton_allocation_constrained,
    compute_mv_optimal_allocation,
    # Portfolio allocation
    compute_bond_weights,
    compute_portfolio_return,
    # Shock generation and simulation
    generate_correlated_shocks,
    simulate_interest_rates,
    simulate_interest_rates_random_walk,
    simulate_stock_returns,
    compute_funded_ratio,
)

# Teaching scenarios
from .scenarios import (
    create_teaching_scenario,
    generate_teaching_scenarios,
    create_scenario_from_simulation_result,
)

# Simulation engines
from .simulation import (
    # Helper functions
    compute_target_allocations,
    normalize_portfolio_weights,
    apply_consumption_constraints,
    compute_dynamic_pv,
    # Generic strategy simulation engine
    simulate_with_strategy,
    # Unified simulation engine (legacy)
    simulate_paths,
    # Profile generators
    compute_earnings_profile,
    compute_expense_profile,
    # Strategy computations
    compute_lifecycle_median_path,
    compute_lifecycle_fixed_consumption,
    compute_median_path_comparison,
    run_lifecycle_monte_carlo,
    run_strategy_comparison,
)

__all__ = [
    # Constants
    'DEFAULT_RISKY_BETA',
    # Params
    'EconomicParams',
    'BondParams',
    'RandomWalkParams',
    'LifecycleParams',
    'MonteCarloParams',
    'BondStrategy',
    'ConsumptionRule',
    'Strategy',
    'SimulationParams',
    'STRATEGIES',
    # Results
    'LifecycleResult',
    'MonteCarloResult',
    'SimulationResult',       # New unified result type
    'StrategyComparison',     # New unified comparison type
    # Deprecated results (kept for backward compatibility)
    'RuleOfThumbResult',
    'ScenarioResult',
    # Generic strategy framework
    'SimulationState',
    'StrategyActions',
    'StrategyProtocol',
    'LDIStrategy',
    'RuleOfThumbStrategy',
    'FixedConsumptionStrategy',
    # Economics
    'effective_duration',
    'effective_duration_vectorized',
    'zero_coupon_price',
    'zero_coupon_price_vectorized',
    'spot_rate',
    'compute_zero_coupon_returns',
    'compute_duration_approx_returns',
    'compute_present_value',
    'compute_pv_consumption',
    'compute_pv_consumption_realized',
    'compute_duration',
    'liability_pv',
    'liability_pv_vectorized',
    'liability_duration',
    'liability_duration_vectorized',
    'compute_full_merton_allocation',
    'compute_full_merton_allocation_constrained',
    'compute_mv_optimal_allocation',
    'compute_bond_weights',
    'compute_portfolio_return',
    'generate_correlated_shocks',
    'simulate_interest_rates',
    'simulate_interest_rates_random_walk',
    'simulate_stock_returns',
    'compute_funded_ratio',
    # Teaching scenarios
    'create_teaching_scenario',
    'generate_teaching_scenarios',
    'create_scenario_from_simulation_result',
    # Simulation helpers
    'compute_target_allocations',
    'normalize_portfolio_weights',
    'apply_consumption_constraints',
    'compute_dynamic_pv',
    'simulate_with_strategy',
    'simulate_paths',
    # Simulation
    'compute_earnings_profile',
    'compute_expense_profile',
    'compute_lifecycle_median_path',
    'compute_lifecycle_fixed_consumption',
    'compute_median_path_comparison',
    'run_lifecycle_monte_carlo',
    'run_strategy_comparison',
]
