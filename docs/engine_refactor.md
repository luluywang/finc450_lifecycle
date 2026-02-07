**Lifecycle Simulation Refactor Plan**

- **Objectives**
  - One engine and state/action contract aligned with `docs/model_specification.tex`.
  - Eliminate duplicated logic for PV, duration, decomposition, normalization, and consumption constraints.
  - Consistent outputs for median path, Monte Carlo, and strategy comparisons.

- **Current Issues**
  - Duplicate engines: `core/simulation.py::simulate_with_strategy` vs `core/simulation.py::simulate_paths` with diverging logic.
  - Duplicate helpers: weight normalization in `core/simulation.py` and `core/strategies.py`; PV/decomposition loops reimplemented inside engines.
  - API inconsistency: `compute_lifecycle_median_path` uses the legacy engine; `run_strategy_comparison` uses the newer one.
  - Inefficient inner-loop PV/duration recomputation via slicing; risk of misplaced calculations and O(T²) cost.
  - Strategy responsibility unclear: engines override consumption/default handling instead of treating strategies as pure `state -> actions`.

- **Target Architecture**
  - New engine `core/engine.py` exporting `rollout(strategy, params, econ, shocks, options) -> SimulationResult`.
  - Single `SimulationState` consumed by strategies and single `StrategyActions` returned; engine enforces constraints, default detection, normalization.
  - Shared primitives in one place (`core/primitives.py` or folded into `core/economics.py`):
    - PV/duration with mean-reverting discounting.
    - HC/expense decomposition to stock/bond/cash.
    - Weight normalization with leverage cap.
    - Consumption constraints.
  - Market generators remain in `core/economics.py`.

- **API Shape**
  - `SimulationState`: age, is_working, financial_wealth, human_capital, pv_expenses, net_worth, earnings, expenses, current_rate, duration_hc, duration_expenses, hc_* components, exp_* components, target_stock/bond/cash, params, econ_params.
  - `StrategyActions`: total_consumption, subsistence_consumption, variable_consumption, target_fin_stock, target_fin_bond, target_fin_cash; engine derives weights.
  - `SimulationResult` (already in `core/params.py`) stays; ensure fields filled identically across wrappers.

- **Implementation Steps**
  1) Create `core/primitives.py` with:
     - `present_value(cashflows, rate, phi, r_bar, max_duration=None)` returning PV and optional duration.
     - `decompose_hc(pv_earnings, duration_earnings, stock_beta, bond_duration)` and `decompose_expenses(pv_expenses, duration_expenses, bond_duration)`.
     - `normalize_weights(target_fin_stock, target_fin_bond, target_fin_cash, fw, target_stock, target_bond, target_cash, max_leverage)`.
     - `apply_consumption_constraints(subsistence, variable, earnings, fw, is_working, defaulted)`.
  2) Add `core/engine.py`:
     - Precompute earnings/expenses arrays once.
     - Precompute static discount vectors for phi=1 to allow dot-product PV/duration per step; fall back to current functions when phi≠1.
     - Loop over sims and periods: build state; call strategy; apply constraints and default check; normalize to weights; update wealth (geometric optional).
     - Assemble `SimulationResult`, handling 1D vs 2D paths.
  3) Refactor strategies in `core/strategies.py`:
     - Remove `_normalize_weights`; import from primitives.
     - Treat strategies as pure decision makers; no wealth-update or constraint logic inside.
  4) Update wrappers in `core/simulation.py`:
     - `compute_lifecycle_median_path` -> call `rollout` with zero shocks, `use_dynamic_revaluation=False`, `use_geometric_returns=True`.
     - `run_lifecycle_monte_carlo` -> call `rollout` with generated shocks, dynamic revaluation.
     - `run_strategy_comparison` -> two `rollout` calls sharing shocks.
     - Deprecate `simulate_paths`; keep as thin alias to `rollout` returning legacy dict for backward compatibility.
  5) Touch ancillary scripts (`compare_implementations.py`, `generate_lecture_figures.py`, `compare_teaching_scenarios.py`, notebooks) to use the unified API.

- **Computation Notes**
  - For phi=1, cache discount factors `exp(-t*r)` and reuse for PV/duration via dot products; slower general path for phi≠1.
  - Duration cap (`max_duration`) applied inside primitives.
  - Portfolio return choice: arithmetic vs geometric (median) via `options.use_geometric_returns`.
  - Wage shocks: keep log-wage accumulator; shocks shift earnings only.

- **Testing Plan**
  - Golden median path: assert new engine matches prior outputs within tolerance.
  - Monte Carlo regression: same seed -> matching final-wealth percentiles and default rates vs legacy.
  - Strategy comparison: LDI vs RoT with shared shocks produce unchanged percentiles.
  - Unit tests for primitives: normalization constraints, PV/duration closed-form for phi=1, decomposition bounds, consumption constraints.
  - Wage-shock test: wage multiplier path equals exp of cumulative shocks scaled by beta*sigma_s.

- **Backward Compatibility**
  - Keep `SimulationResult` fields stable; fill `target_fin_*` only when provided.
  - `simulate_paths` marked deprecated; emits warning but delegates to `rollout`.
  - Document behavioral changes (unified constraints) in `TODO.md` and docstrings.

- **Open Decisions**
  - Actions payload: dollar targets only vs also weights (recommended: targets only; engine derives weights).
  - Home for primitives: new `core/primitives.py` vs expanding `core/economics.py`.
  - Default rule when `FW <= 0` while working: retain current “default immediately” or allow limited borrowing; choose and document.
  - Performance target: require phi=1 fast path; accept slower general path for phi≠1.

- **Documentation**
  - Add `docs/engine_overview.md` (or expand this file) describing state/action flow and mapping to model equations.
  - Update wrapper docstrings to note they are thin shells over `rollout`.
