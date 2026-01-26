# FINC450 Lifecycle Investment Strategy

## Requirements

- Python 3.8+
- Node.js 18+ (for web visualizer)

## Installation

Install all required packages:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
finc450_lifecycle/
├── core/                              # Core computation module (Python)
│   ├── __init__.py                    # Public API exports
│   ├── params.py                      # All dataclasses (LifecycleParams, EconomicParams, etc.)
│   ├── economics.py                   # Bond pricing, PV calculations, MV optimization
│   ├── simulation.py                  # Monte Carlo engines, strategy comparison
│   ├── strategies.py                  # Generic strategy implementations (LDI, RoT, Fixed)
│   └── scenarios.py                   # Teaching scenarios (good/bad/median paths)
│
├── visualization/                     # Matplotlib visualization code
│   ├── __init__.py                    # Public API exports
│   ├── styles.py                      # Colors, fonts, style constants
│   ├── helpers.py                     # Plot utilities
│   ├── lifecycle_plots.py             # Median path charts
│   ├── monte_carlo_plots.py           # Fan charts, distributions
│   ├── comparison_plots.py            # Strategy comparisons
│   ├── sensitivity_plots.py           # Parameter sensitivity
│   └── report_pages.py                # PDF page layouts
│
├── deprecated/                        # Backward compatibility stubs
│   ├── __init__.py
│   ├── retirement_simulation.py
│   ├── test_leverage_hypothesis.py
│   └── visualizations.py
│
├── tests/                             # Unit tests
│   ├── __init__.py
│   └── test_risky_hc_hypothesis.py
│
├── docs/                              # LaTeX documentation
│   ├── model_specification.tex        # DGP specification
│   └── lifecycle_lecture.tex          # Lecture slides
│
├── notebooks/                         # Jupyter notebooks
├── data/                              # Data files
│
├── output/                            # Generated outputs (tracked in git)
│   ├── figures/                       # Lecture figures (PNG)
│   ├── lifecycle_strategy.pdf         # Main lifecycle report
│   ├── strategy_comparison.pdf        # LDI vs RoT comparison
│   ├── python_verification.json       # Python verification data
│   ├── typescript_verification.json   # TypeScript verification data
│   └── verification_report.md         # Comparison report
│
├── web/                               # Interactive web visualizer
│   ├── lifecycle_visualizer_artifact.tsx  # SOURCE OF TRUTH (tracked in git)
│   ├── ISSUES.md                      # Known issues and bugs
│   └── app/                           # Vite React app (gitignored)
│       ├── src/
│       │   ├── App.tsx
│       │   ├── main.tsx
│       │   └── LifecycleVisualizer.tsx  # Working copy (NOT tracked)
│       ├── package.json
│       ├── tsconfig.json
│       └── vite.config.ts
│
├── generate_report.py                 # Main lifecycle report PDF generator
├── compare_strategies.py              # LDI vs RoT comparison PDF generator
├── compare_teaching_scenarios.py      # Teaching scenario comparisons
├── generate_lecture_figures.py        # Lecture figure generation
├── compare_implementations.py         # Python vs TypeScript verification
├── generate_ts_verification.ts        # Standalone TypeScript verification
├── Makefile                           # Build automation
├── requirements.txt                   # Python dependencies
└── CLAUDE.md                          # This file
```

---

## Core Module

The `core/` module is the single source of truth for all simulation logic:

### Key Files

| File | Purpose |
|------|---------|
| `params.py` | All dataclasses: `LifecycleParams`, `EconomicParams`, `MonteCarloParams`, `LifecycleResult`, `SimulationResult`, `StrategyComparison` |
| `economics.py` | Bond pricing (`zero_coupon_price`, `effective_duration`), PV/duration calculations, MV optimization (`compute_mv_optimal_allocation`), shock generation |
| `simulation.py` | `simulate_paths()` - unified simulation engine, `compute_lifecycle_median_path()`, `run_strategy_comparison()` |
| `strategies.py` | `LDIStrategy`, `RuleOfThumbStrategy`, `FixedConsumptionStrategy` |
| `scenarios.py` | Teaching scenarios (good/bad/median market conditions) |

### Usage

```python
from core import (
    # Parameter dataclasses
    LifecycleParams,
    EconomicParams,
    # Result types
    SimulationResult,
    StrategyComparison,
    # Simulation functions
    run_strategy_comparison,
    compute_median_path_comparison,
    compute_lifecycle_median_path,
    simulate_paths,
)
```

### Key Insight: Static = Dynamic with Zero Shocks

The codebase unifies static (median path) and dynamic (Monte Carlo) calculations:
- **Median path**: `simulate_paths()` with zero shocks
- **Monte Carlo**: `simulate_paths()` with random shocks from `generate_correlated_shocks()`

---

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

### TypeScript Implementation Details

The TypeScript implementation mirrors the Python `simulate_paths()` function. Key functions:

| TypeScript Function | Python Equivalent |
|---------------------|-------------------|
| `effectiveDuration()` | `effective_duration()` |
| `zeroCouponPrice()` | `zero_coupon_price()` |
| `computePresentValue()` | `compute_present_value()` |
| `computeDuration()` | `compute_duration()` |
| `computeFullMertonAllocationConstrained()` | `compute_mv_optimal_allocation()` |
| `computeLifecycleMedianPath()` | `compute_lifecycle_median_path()` |
| `normalizePortfolioWeights()` | `normalize_portfolio_weights()` |

---

## Critical Implementation Detail: Consumption vs Wealth Evolution

**IMPORTANT:** Python uses **different returns** for consumption rate vs wealth evolution:

### Consumption Rate (lines 720-727 in `simulation.py`)
```python
avg_return = (
    target_stock * (r + mu_excess) +
    target_bond * r +           # Just r, NO mu_bond!
    target_cash * r
)
consumption_rate = avg_return + consumption_boost
```

### Wealth Evolution (lines 882-890 in `simulation.py`)
```python
stock_ret = stock_return_paths[sim, t]  # r + mu_excess + shock
bond_ret = bond_return_paths[sim, t]    # r + mu_bond + duration effect
cash_ret = rate_paths[sim, t]           # r

portfolio_return = w_s * stock_ret + w_b * bond_ret + w_c * cash_ret
```

The bond component in consumption rate is just `r`, but wealth evolution uses `r + mu_bond`. This is because:
- Consumption rate is based on the **certainty-equivalent** return
- Wealth evolution uses the **actual expected** return including risk premium

### TypeScript Must Match This

The TypeScript `computeLifecycleMedianPath()` function must:
1. Use `targetBond * r` (not `r + muBond`) for consumption rate
2. Compute portfolio weights **at each time step** (not after the loop)
3. Use time-varying portfolio returns for wealth evolution

---

## Verification Tools

### Python Verification Script

```bash
# Generate Python verification data
python3 compare_implementations.py --pretty -o output/python_verification.json

# Compare with TypeScript
python3 compare_implementations.py --compare output/typescript_verification.json
```

### TypeScript Verification Script

```bash
# Generate TypeScript verification data (standalone, no browser needed)
npx tsx generate_ts_verification.ts > output/typescript_verification.json
```

### Running Full Verification

```bash
# 1. Generate Python data
python3 compare_implementations.py --pretty -o output/python_verification.json

# 2. Generate TypeScript data
npx tsx generate_ts_verification.ts > output/typescript_verification.json

# 3. Compare
python3 compare_implementations.py --compare output/typescript_verification.json -o output/verification_report.md
```

### Expected Output

All values should match within tolerance:
- Economic functions: < 0.0001 (0.01%)
- PV/Duration calculations: < 0.001 (0.1%)
- Portfolio weights: < 0.01 (1%)
- Wealth paths: < 1% at each time point

---

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

---

## Default Parameters

### Economic Parameters (`EconomicParams`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r_bar` | 0.02 | Long-run mean interest rate |
| `phi` | 1.0 | Mean reversion parameter (1.0 = no mean reversion) |
| `sigma_r` | 0.003 | Interest rate volatility |
| `mu_excess` | 0.04 | Stock excess return (equity risk premium) |
| `sigma_s` | 0.18 | Stock return volatility |
| `rho` | 0.0 | Correlation between rate and stock shocks |
| `bond_sharpe` | 0.037 | Bond Sharpe ratio |
| `bond_duration` | 20.0 | Duration of bond portfolio |

### Lifecycle Parameters (`LifecycleParams`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start_age` | 25 | Starting age |
| `retirement_age` | 65 | Retirement age |
| `end_age` | 95 | End of life age |
| `initial_earnings` | 200 | Initial annual earnings ($K) |
| `base_expenses` | 100 | Base annual expenses ($K) |
| `retirement_expenses` | 100 | Annual retirement expenses ($K) |
| `gamma` | 2.0 | Risk aversion coefficient |
| `initial_wealth` | 100 | Initial financial wealth ($K) |
| `stock_beta_human_capital` | 0.0 | Beta of human capital to stocks |

---

## Strategy Framework

Strategies are simple functions mapping `SimulationState -> StrategyActions`:

```python
from core import (
    SimulationState,
    StrategyActions,
    LDIStrategy,
    RuleOfThumbStrategy,
    simulate_with_strategy,
    generate_correlated_shocks,
)

# Create strategies
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

### Available Strategies

| Strategy | Description |
|----------|-------------|
| `LDIStrategy` | Liability-Driven Investment with optimal consumption |
| `RuleOfThumbStrategy` | 100-age rule with fixed savings/withdrawal rates |
| `FixedConsumptionStrategy` | 4% rule style fixed withdrawal |

---

## Deprecated Modules

The following modules are deprecated and maintained only for backward compatibility:

- `retirement_simulation.py` → Use `from core import ...` instead
- `visualizations.py` → Use `from visualization import ...` instead

These stub files redirect to `deprecated/` and emit deprecation warnings on import.
