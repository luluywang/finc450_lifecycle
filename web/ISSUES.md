# Web Visualizer Issues - January 26, 2026

## Summary

During verification testing, the following issues were identified in the web visualizer that need to be resolved.

---

## Issue 1: Summary Tab and Individual Scenario Tabs Show Different Numbers

**Status:** Unresolved - Root cause identified but fix incomplete

**Symptom:**
- Summary tab shows: Baseline LDI Default Rate = 2.2%, RoT = 35.8%
- Baseline individual tab shows: Optimal Strategy = 22% (11/50), RoT = 6% (3/50)
- The numbers are completely different and show OPPOSITE performance

**Root Cause:**
The Summary tab and individual scenario tabs were using different simulation systems:
1. Summary uses `runTeachingScenarios()` with 500 simulations
2. Individual tabs were using `runStrategyComparison()` with 50 simulations

**Partial Fix Applied:**
- Removed dead code (`computeOptimalStrategy`, `computeRuleOfThumbStrategy`, `runStrategyComparison`)
- Individual tabs now reference `teachingScenarios[scenarioKey]`

**Remaining Issue:**
Browser caching may be showing old JavaScript. Hard refresh (Cmd+Shift+R) required to see latest code. If issues persist after hard refresh, clear browser cache completely.

---

## Issue 2: Beta Toggle Buttons Should Not Exist in Scenarios Tab

**Status:** Partially resolved

**Symptom:**
Verification found "beta=0 (Bond-like)" and "beta=0.4 (Risky)" toggle buttons in the Teaching Scenarios area.

**Expected Behavior:**
- There should be ONLY ONE beta control: "HC stock beta" stepper in the left Parameters sidebar
- Teaching scenarios should use whatever beta value is set in the Parameters section
- No separate beta toggles in the Scenarios tab

**Fix Applied:**
- Task 8 removed `scenarioBeta` state and the toggle buttons
- `lifecycleParams` now uses `params.stockBetaHC` directly

**Verification Needed:**
After hard refresh, confirm no beta toggle buttons appear in the Scenarios tab area.

---

## Issue 3: Parameter Defaults Should Match Python

**Status:** Resolved

**Changes Made:**
| Parameter | Old Value | New Value (matches Python) |
|-----------|-----------|---------------------------|
| endAge | 85 | 95 |
| initialEarnings | 100 | 200 ($200k) |
| earningsGrowth | 0.02 | 0.0 (flat) |
| earningsHumpAge | 50 | 65 |
| earningsDecline | 0.01 | 0.0 |
| baseExpenses | 60 | 100 ($100k) |
| expenseGrowth | 0.01 | 0.0 |
| retirementExpenses | 80 | 100 ($100k) |
| stockBetaHC | 0.1 | 0.0 |
| initialWealth | 1 | 100 ($100k) |
| muStock | 0.03 | 0.04 |
| bondSharpe | 0.10 | 0.037 |
| sigmaR | 0.006 | 0.003 |

---

## Issue 4: Median Return Adjustment

**Status:** Implemented in TypeScript only

**Description:**
Added `-0.5 * sigma^2` adjustment to stock returns to ensure median paths represent geometric (not arithmetic) mean returns.

**Note:**
The Python code does NOT have this adjustment. For full consistency, consider updating `core/economics.py` as well.

---

## Issue 5: Individual Scenario Tabs Should Show 8 Panels Matching PDF

**Status:** Implemented

**Expected 8 Panels per scenario (matching teaching_scenarios.pdf):**
1. Cumulative Stock Market Returns (fan chart)
2. Interest Rate Paths (fan chart)
3. Financial Wealth - LDI vs RoT (overlaid comparison)
4. Default Timing (histogram with default rates)
5. Stock Allocation - LDI vs RoT (overlaid fan chart)
6. Bond Allocation - LDI vs RoT (overlaid fan chart)
7. Terminal Wealth Distribution (histogram)
8. PV Consumption Distribution (histogram)

**Implementation:**
Panels are implemented in the individual scenario tabs using `teachingScenarios` data.

---

## Verification Steps

To verify fixes are working:

1. **Hard refresh the browser** (Cmd+Shift+R on Mac, Ctrl+Shift+R on Windows)
2. If issues persist, clear browser cache completely or test in incognito mode
3. Navigate to Scenarios tab
4. Click "Run Simulation"
5. Check Summary tab - note the Baseline default rates
6. Check Baseline tab - the "Panel 4: Default Timing" should show identical rates
7. Confirm no "beta=0 (Bond-like)" buttons exist anywhere in Scenarios tab

---

## Files Modified

Key commits:
- `[econ_ra:task1-17]` - PNG export and scenario panel implementation
- `[econ_ra:fix]` - Parameter sync, median return adjustment, data source unification
- `[econ_ra:verify]` - Readability improvements

Main file: `/web/lifecycle_visualizer_artifact.tsx`
