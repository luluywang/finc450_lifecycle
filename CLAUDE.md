# FINC450 Lifecycle Investment Strategy

## Requirements

- Python 3.8+

## Installation

Install all required packages:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
finc450_lifecycle/
├── core/                           # Core computation module
│   ├── __init__.py                 # Public API exports
│   ├── params.py                   # All dataclasses (LifecycleParams, EconomicParams, etc.)
│   ├── economics.py                # Bond pricing, PV calculations, MV optimization
│   ├── simulation.py               # Monte Carlo engines, strategy comparison
│   ├── strategies.py               # Generic strategy implementations (LDI, RoT, Fixed)
│   └── scenarios.py                # Teaching scenarios (good/bad/median paths)
│
├── visualization/                  # Matplotlib visualization code
│   ├── __init__.py                 # Public API exports
│   ├── styles.py                   # Colors, fonts, style constants
│   ├── helpers.py                  # Plot utilities
│   ├── lifecycle_plots.py          # Median path charts
│   ├── monte_carlo_plots.py        # Fan charts, distributions
│   ├── comparison_plots.py         # Strategy comparisons
│   ├── sensitivity_plots.py        # Parameter sensitivity
│   └── report_pages.py             # PDF page layouts
│
├── deprecated/                     # Backward compatibility stubs
│
├── docs/                           # LaTeX documentation
│   ├── model_specification.tex     # DGP specification
│   └── lifecycle_lecture.tex       # Lecture slides
│
├── notebooks/                      # Jupyter notebooks
├── data/                           # Data files
├── output/                         # Generated figures and PDFs (tracked in git)
│
├── generate_report.py              # Main lifecycle report PDF
├── compare_strategies.py           # LDI vs RoT comparison PDF
├── generate_lecture_figures.py     # Lecture figure generation
├── Makefile                        # Build automation
├── requirements.txt                # Python dependencies
│
├── web/                            # Interactive web visualizer
│   ├── lifecycle_visualizer_artifact.tsx  # SOURCE OF TRUTH (tracked in git)
│   ├── ISSUES.md                   # Known issues and bugs
│   └── app/                        # Vite React app (gitignored)
│       ├── src/
│       │   └── LifecycleVisualizer.tsx  # Working copy (NOT tracked)
│       └── ...
```

## Web Visualizer

The `web/` directory contains an interactive TypeScript/React visualizer.

### File Structure & Workflow

**IMPORTANT:** The `web/app/` directory is gitignored. The source of truth is:

```
web/lifecycle_visualizer_artifact.tsx  ← SOURCE OF TRUTH (tracked in git)
web/app/src/LifecycleVisualizer.tsx    ← Working copy (gitignored)
```

### Development Workflow

1. **Copy artifact to app for development:**
   ```bash
   cp web/lifecycle_visualizer_artifact.tsx web/app/src/LifecycleVisualizer.tsx
   ```

2. **Start dev server:**
   ```bash
   cd web/app && npm install && npm run dev
   ```

3. **Make changes** to `web/app/src/LifecycleVisualizer.tsx` and test at `http://localhost:5173/`

4. **Copy back to artifact when done:**
   ```bash
   cp web/app/src/LifecycleVisualizer.tsx web/lifecycle_visualizer_artifact.tsx
   ```

5. **Commit the artifact file** (not the app directory)

### Core Module

The `core/` module is the single source of truth for all simulation logic:

```python
from core import (
    # Parameter dataclasses
    LifecycleParams,
    EconomicParams,
    # Result types
    SimulationResult,      # Returned by simulate_with_strategy()
    StrategyComparison,    # Returned by run_strategy_comparison()
    # Simulation functions
    run_strategy_comparison,
    compute_median_path_comparison,
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
    # Result types
    SimulationResult,
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
result: SimulationResult = simulate_with_strategy(ldi, params, econ, rate_shocks, stock_shocks)

# Access result fields
print(result.ages)           # Array of ages
print(result.wealth)         # Wealth trajectories (n_sims, n_periods)
print(result.consumption)    # Consumption trajectories
print(result.stock_weight)   # Stock allocation weights

# Random shocks = Monte Carlo
rate_shocks, stock_shocks = generate_correlated_shocks(n_periods, n_sims, rho, rng)
mc_result: SimulationResult = simulate_with_strategy(rot, params, econ, rate_shocks, stock_shocks)
```

Key insight: **Static calculations = Dynamic with zero shocks**. This unifies the codebase.

### Result Types

The simulation functions return structured dataclasses:

- **`SimulationResult`** - Returned by `simulate_with_strategy()`. Contains arrays for ages, wealth, consumption, stock weights, and other simulation outputs.

- **`StrategyComparison`** - Returned by `run_strategy_comparison()` and `compute_median_path_comparison()`. Contains results for multiple strategies keyed by strategy name, plus shared metadata like ages.

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

## Building

Generate all outputs using Make:

```bash
make              # Generate all figures and PDFs
make figures      # Generate lecture figures only
make pdfs         # Generate PDF reports only
make clean        # Remove generated files
make help         # Show available targets
```

Outputs are written to `output/`:
- `output/lifecycle_strategy.pdf` - Main lifecycle analysis
- `output/strategy_comparison.pdf` - LDI vs RoT comparison
- `output/figures/` - Lecture figures (PNG)

### Custom Parameters

```bash
python3 generate_report.py -o output/custom.pdf --initial-earnings 120 --stock-beta 0.4
```

Run `python3 generate_report.py --help` for all available options.
