# Remaining Refactoring Work

## 1. Migrate simulation functions to return unified `SimulationResult`

The new `SimulationResult` and `StrategyComparison` types are defined in `core/params.py` but not yet used by the simulation functions. The following functions still return legacy types:

| Function | Current Return | Target Return | Status |
|----------|---------------|---------------|--------|
| `simulate_with_strategy()` | `SimulationResult` | `SimulationResult` | Done |
| `compute_rule_of_thumb_strategy()` | — | — | Removed |
| `run_strategy_comparison()` | `StrategyComparison` | `StrategyComparison` | Done |
| `compute_median_path_comparison()` | `StrategyComparison` | `StrategyComparison` | Done |

### Steps

1. Update `simulate_with_strategy()` in `core/simulation.py` to return `SimulationResult`
2. Update consumers (visualization functions) to use new field names
3. Update comparison functions to return `StrategyComparison`
4. Remove deprecated result types once all consumers are migrated

### Benefits

- Single result type regardless of strategy used
- Percentile computation built into the result object
- Cleaner comparison: just two `SimulationResult` objects with same market shocks

## 2. Clean up unstaged changes

There are unstaged changes from previous directory reorganization:

```
modified:   Makefile
modified:   deprecated/retirement_simulation.py
modified:   deprecated/visualizations.py
deleted:    retirement_simulation.py (root stub)
deleted:    test_leverage_hypothesis.py (root stub)
deleted:    visualizations.py (root stub)
```

Review and commit these separately.

## 3. Optional: Simplify `LifecycleResult`

`LifecycleResult` has 30+ fields for LDI-specific data:
- HC decomposition (stock, bond, cash components)
- Expense liability decomposition
- Durations
- Target financial holdings
- etc.

Options:
- **Keep as-is**: Useful for teaching (shows all intermediate calculations)
- **Split**: `SimulationResult` + `LDIDetails` composition

## 4. Optional: Update CLAUDE.md

Update documentation to reflect:
- New `SimulationResult` and `StrategyComparison` types in `core/params.py`
- New `core/scenarios.py` module for teaching scenarios
- New `visualization/report_pages.py` module for PDF page layouts

---

## Completed Work (2025-01-25)

### Commit `657d678`: Unify result types and refactor generate_report.py

- Created unified `SimulationResult` dataclass (works for single sim and Monte Carlo)
- Added `StrategyComparison` class with on-demand statistics
- Marked deprecated types with clear docstrings
- Moved teaching scenarios to `core/scenarios.py`
- Moved PDF page creation to `visualization/report_pages.py`
- Reduced `generate_report.py` from 1648 to 551 lines (67% reduction)
- Added `compare_teaching_scenarios.py` for LDI vs RoT analysis

### Migration Table Updates

- `compute_rule_of_thumb_strategy()` removed (functionality replaced by `RuleOfThumbStrategy` class used with `simulate_with_strategy()`)
- All simulation functions now return unified result types (`SimulationResult` or `StrategyComparison`)
