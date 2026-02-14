#!/usr/bin/env python3
"""
Check if the kink is actually a TIMING CONVENTION issue, not a real kink.

If target dollar positions don't show a sharp kink, but weights do,
the issue might be: WHEN are we computing things?

- Is HC dropped BEFORE or AFTER wealth accumulation?
- Are weights computed before or after new savings?
- Is there a mismatch in timing?
"""

import numpy as np
import pandas as pd
from core import (
    LifecycleParams,
    EconomicParams,
    simulate_with_strategy,
    LDIStrategy,
)


def check_timing():
    params = LifecycleParams()
    econ = EconomicParams()

    # Median path
    total_years = params.end_age - params.start_age
    result = simulate_with_strategy(
        LDIStrategy(),
        params,
        econ,
        np.zeros((1, total_years)),
        np.zeros((1, total_years)),
    )

    def get_1d(arr):
        return arr if arr.ndim == 1 else arr[0]

    retirement_idx = params.retirement_age - params.start_age
    before = retirement_idx - 1
    after = retirement_idx

    fw = get_1d(result.financial_wealth)
    target_fin_stock = get_1d(result.target_fin_stock)
    target_fin_bond = get_1d(result.target_fin_bond)
    target_fin_cash = get_1d(result.target_fin_cash)
    stock_weight = get_1d(result.stock_weight)
    bond_weight = get_1d(result.bond_weight)
    cash_weight = get_1d(result.cash_weight)

    print("\n" + "=" * 120)
    print("TIMING CONVENTION ANALYSIS")
    print("=" * 120)

    # Look at a wider range around retirement
    print(f"\nDetailed view around retirement (ages 62-67):")
    print(f"{'Age':<5} {'FW':<12} {'Target$':<20} {'Weights':<35}")
    print(f"{'':<5} {'':<12} {'Stock':>6} {'Bond':>6} {'Cash':>6} {'Stock':>8} {'Bond':>8} {'Cash':>8}")
    print("-" * 120)

    for i in range(37, 43):  # Ages 62-67
        age = params.start_age + i
        total_target = target_fin_stock[i] + target_fin_bond[i] + target_fin_cash[i]
        print(
            f"{age:<5} {fw[i]:>11.2f} "
            f"{target_fin_stock[i]:>6.0f} {target_fin_bond[i]:>6.0f} {target_fin_cash[i]:>6.0f} "
            f"{stock_weight[i]:>8.4f} {bond_weight[i]:>8.4f} {cash_weight[i]:>8.4f}"
        )

    print("\n" + "=" * 120)
    print("HYPOTHESIS: The issue is how wealth changes during the retirement year")
    print("=" * 120)

    print(f"""
At age 64 (before retirement):
  FW = {fw[before]:.2f}
  Target_bond = {target_fin_bond[before]:.2f}
  Bond weight = {bond_weight[before]:.4f} = {target_fin_bond[before]:.2f} / {fw[before]:.2f}

At age 65 (after retirement):
  FW = {fw[after]:.2f}  (increased by {fw[after] - fw[before]:+.2f})
  Target_bond = {target_fin_bond[after]:.2f}  (decreased by {target_fin_bond[after] - target_fin_bond[before]:+.2f})
  Bond weight = {bond_weight[after]:.4f} = {target_fin_bond[after]:.2f} / {fw[after]:.2f}

The KEY QUESTION: When during the transition year does:
1. HC drop to 0?
2. Wealth accumulate from returns + savings?
3. Target positions get recomputed?
4. Weights get computed?

TIMING SCENARIO A: HC drops, then wealth grows, then targets computed
  - HC goes to 0 at START of year 65
  - Wealth grows during year 65: {fw[before]:.2f} → {fw[after]:.2f}
  - Targets are computed based on new wealth
  - This would create a DISCONTINUITY at year 65

TIMING SCENARIO B: HC drops and wealth grows SIMULTANEOUSLY
  - HC and wealth change are part of same simulation step
  - Targets reflect new (no HC) + new wealth
  - Still creates discontinuity but for different reason

TIMING SCENARIO C: HC and targets are misaligned
  - Maybe HC is computed at time t
  - But targets are computed at time t+1
  - This could create artificial discontinuities

The fact that TARGET DOLLAR POSITIONS change smoothly but WEIGHTS kink
suggests that the issue might be:
  The DENOMINATOR (total FW) grows faster than the NUMERATOR (bond position)
  creates the weight kink.

Let me decompose this:
""")

    # Decompose the weight change
    bond_weight_before = bond_weight[before]
    bond_weight_after = bond_weight[after]
    bond_target_before = target_fin_bond[before]
    bond_target_after = target_fin_bond[after]
    fw_before = fw[before]
    fw_after = fw[after]

    print(f"\nBond weight decomposition:")
    print(f"  Before: {bond_weight_before:.4f} = {bond_target_before:.2f} / {fw_before:.2f}")
    print(f"  After:  {bond_weight_after:.4f} = {bond_target_after:.2f} / {fw_after:.2f}")
    print(f"  Change: {bond_weight_after - bond_weight_before:+.4f}")
    print(f"")

    # Use math to decompose the ratio change
    # w = B/FW, so dw = (dB * FW - B * dFW) / FW^2
    dB = bond_target_after - bond_target_before
    dFW = fw_after - fw_before
    dw = dB / fw_before - bond_target_before * dFW / (fw_before ** 2)

    print(f"  Effect of target bond change: {dB:.2f} / {fw_before:.2f} = {dB / fw_before:+.4f}")
    print(f"  Effect of FW growth: -{bond_target_before:.2f} * {dFW:.2f} / {fw_before**2:.0f} = {-bond_target_before * dFW / (fw_before ** 2):+.4f}")
    print(f"  Total: {dB / fw_before - bond_target_before * dFW / (fw_before ** 2):+.4f}")
    print(f"  Actual change: {bond_weight_after - bond_weight_before:+.4f}")

    print(f"""
INSIGHT: The weight kink comes from TWO effects:

1. TARGET BOND DECREASES: {dB:+.2f}
   Effect on weight: -{abs(dB) / fw_before:.4f}

2. TOTAL FW INCREASES: {dFW:+.2f}
   The old bond position gets diluted by new wealth
   Effect on weight: -{bond_target_before * dFW / (fw_before ** 2):+.4f}

Both effects make bond weight drop!

TIMING HYPOTHESIS: Is this correct?
The question is whether there's a MISMATCH in when these two effects happen.

If HC and targets are computed BEFORE wealth accumulation:
  - Targets would be based on "old" HC situation
  - Then wealth grows
  - This creates a mismatch

If HC and targets are computed AFTER wealth accumulation:
  - Targets properly respond to both HC loss AND wealth growth
  - No mismatch

Let me check the CODE to see the actual timing...
""")


if __name__ == "__main__":
    check_timing()
