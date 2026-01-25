# FINC450 Lifecycle Investment Strategy

## Requirements

- Python 3.8+

## Installation

Install all required packages:

```bash
pip install -r requirements.txt
```

## Module Structure

```
finc450_lifecycle/
├── core/                           # Core module (SINGLE SOURCE OF TRUTH)
│   ├── __init__.py                 # Public API exports
│   ├── params.py                   # All dataclasses (LifecycleParams, EconomicParams, etc.)
│   ├── economics.py                # Bond pricing, PV calculations, MV optimization
│   ├── simulation.py               # Monte Carlo engines, strategy comparison
│   └── strategies.py               # Generic strategy implementations (LDI, RoT, Fixed)
│
├── visualization/                  # Consolidated matplotlib visualization code
│   ├── __init__.py                 # Public API exports
│   ├── styles.py                   # Colors, fonts, style constants
│   ├── helpers.py                  # Plot utilities (apply_wealth_log_scale, setup_figure, etc.)
│   ├── lifecycle_plots.py          # Median path charts (earnings, wealth, allocations)
│   ├── monte_carlo_plots.py        # Fan charts, distributions, teaching scenarios
│   ├── comparison_plots.py         # Strategy comparisons (LDI vs RoT)
│   └── sensitivity_plots.py        # Parameter sensitivity (beta, gamma, volatility)
│
├── deprecated/                     # Deprecated modules (backward compatibility only)
│   ├── __init__.py
│   ├── retirement_simulation.py    # Use core/ instead
│   └── visualizations.py           # Use visualization/ instead
│
├── lifecycle_strategy.py           # PDF generation entry point
├── dashboard.py                    # Strategy comparison dashboard
├── generate_lecture_figures.py     # Educational figure generation
├── retirement_simulation.py        # Stub → deprecated/retirement_simulation.py
└── visualizations.py               # Stub → deprecated/visualizations.py
```

### Core Module

The `core/` module is the single source of truth for all simulation logic:

```python
from core import (
    LifecycleParams,
    EconomicParams,
    ScenarioResult,
    compute_lifecycle_median_path,
    run_strategy_comparison,
)
```

### Generic Strategy Framework

Strategies are simple functions mapping `SimulationState -> StrategyActions`:

```python
from core import (
    # State/Action dataclasses
    SimulationState,
    StrategyActions,
    StrategyProtocol,
    # Strategy implementations
    LDIStrategy,
    RuleOfThumbStrategy,
    FixedConsumptionStrategy,
    # Generic simulation engine
    simulate_with_strategy,
    generate_correlated_shocks,
)

# Run any strategy with the generic engine
ldi = LDIStrategy(allow_leverage=False)
rot = RuleOfThumbStrategy(savings_rate=0.15, withdrawal_rate=0.04)

# Zero shocks = deterministic median path
rate_shocks = np.zeros((1, n_periods))
stock_shocks = np.zeros((1, n_periods))
result = simulate_with_strategy(ldi, params, econ, rate_shocks, stock_shocks)

# Random shocks = Monte Carlo
rate_shocks, stock_shocks = generate_correlated_shocks(n_periods, n_sims, rho, rng)
mc_result = simulate_with_strategy(rot, params, econ, rate_shocks, stock_shocks)
```

Key insight: **Static calculations = Dynamic with zero shocks**. This unifies the codebase.

### Visualization Module

The `visualization/` module consolidates all matplotlib plotting code:

```python
from visualization import (
    # Styles and helpers
    COLORS,
    apply_wealth_log_scale,
    setup_figure,

    # Lifecycle plots
    create_lifecycle_figure,
    plot_earnings_expenses_profile,

    # Monte Carlo plots
    create_monte_carlo_fan_chart,
    create_teaching_scenarios_figure,

    # Comparison plots
    create_strategy_comparison_figure,
    create_median_path_comparison_figure,

    # Sensitivity plots
    create_beta_comparison_figure,
    create_gamma_comparison_figure,
)
```

### Deprecated Modules

The following modules are deprecated and maintained only for backward compatibility:

- `retirement_simulation.py` → Use `from core import ...` instead
- `visualizations.py` → Use `from visualization import ...` instead

These stub files redirect to `deprecated/` and emit deprecation warnings on import.

## Generating PDFs

To regenerate the lifecycle strategy PDF:

```bash
python3 lifecycle_strategy.py
```

This creates `lifecycle_strategy.pdf` with the full lifecycle investment analysis.

### Custom Parameters

```bash
python3 lifecycle_strategy.py -o custom.pdf --initial-earnings 120 --stock-beta 0.4 --bond-duration 5.0
```

Run `python3 lifecycle_strategy.py --help` for all available options.

## Other Entry Points

Generate strategy comparison dashboard:
```bash
python3 dashboard.py
```

Generate lecture figures:
```bash
python3 generate_lecture_figures.py --output-dir figures/
```
