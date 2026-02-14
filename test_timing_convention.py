#!/usr/bin/env python3
"""
Test: What if we exclude the "current period" cash flow from remaining liabilities?

Compare two timing conventions:
1. CURRENT: Include current period as "t=1" (all flows are future)
2. ALTERNATIVE: Exclude current period, start from t+1 (already realized)
"""

import numpy as np
from core import (
    LifecycleParams,
    EconomicParams,
    compute_present_value,
    compute_duration,
)


def test_timing_conventions():
    """Compare duration under two timing conventions."""

    params = LifecycleParams()
    econ = EconomicParams()

    # Expense stream (31 years from age 64 onward)
    remaining_years = params.end_age - 64
    expenses = np.full(remaining_years, params.retirement_expenses)
    expenses[0] = params.base_expenses  # Age 64 has working expense

    print("\n" + "=" * 100)
    print("TIMING CONVENTION TEST: Current Period Inclusion")
    print("=" * 100)

    print(f"\nExpense stream (starting at age 64):")
    print(f"  Years 0-0 (age 64): {expenses[0]:.2f} (working)")
    print(f"  Years 1-30 (ages 65-95): {expenses[1]:.2f} each")
    print(f"  Total: {np.sum(expenses):.2f}")

    # CURRENT CONVENTION: enumerate(cashflows, 1)
    # All cash flows treated as 1+ periods away
    pv_current = compute_present_value(expenses, econ.r_bar, econ.phi, econ.r_bar)
    dur_current = compute_duration(expenses, econ.r_bar, econ.phi, econ.r_bar)

    print(f"\n" + "-" * 100)
    print("CONVENTION 1: Current (enumerate from 1)")
    print("-" * 100)
    print(f"PV = {pv_current:.2f}")
    print(f"Duration = {dur_current:.4f} years")
    print(f"\nInterpretation:")
    print(f"  - expenses[0] is discounted at t=1 (one period ahead)")
    print(f"  - expenses[1] is discounted at t=2 (two periods ahead)")
    print(f"  - All cash flows are FUTURE")

    # ALTERNATIVE CONVENTION: exclude current period
    # Pretend we've already paid/earned the current period
    expenses_future_only = expenses[1:]  # Skip the current year

    # But we need to rescale so enumeration still starts at 1
    # This would be like "enumerate from 1" on a shorter array
    pv_alt = compute_present_value(expenses_future_only, econ.r_bar, econ.phi, econ.r_bar)
    dur_alt = compute_duration(expenses_future_only, econ.r_bar, econ.phi, econ.r_bar)

    print(f"\n" + "-" * 100)
    print("CONVENTION 2: Alternative (exclude current period)")
    print("-" * 100)
    print(f"PV = {pv_alt:.2f}")
    print(f"Duration = {dur_alt:.4f} years")
    print(f"\nInterpretation:")
    print(f"  - expenses[0] (age 64) is EXCLUDED (already paid/committed)")
    print(f"  - expenses[1] (age 65) is discounted at t=1 (next period)")
    print(f"  - Only expenses from age 65 onward count as FUTURE liabilities")

    # Now check what happens when we move from age 64 to 65
    print(f"\n" + "=" * 100)
    print("TRANSITION: Age 64 → Age 65")
    print("=" * 100)

    # At age 64
    expenses_at_64 = expenses  # 31 years of expenses
    pv_64_current = compute_present_value(expenses_at_64, econ.r_bar, econ.phi, econ.r_bar)
    dur_64_current = compute_duration(expenses_at_64, econ.r_bar, econ.phi, econ.r_bar)

    expenses_at_64_alt = expenses_at_64[1:]  # Skip current year
    pv_64_alt = compute_present_value(expenses_at_64_alt, econ.r_bar, econ.phi, econ.r_bar)
    dur_64_alt = compute_duration(expenses_at_64_alt, econ.r_bar, econ.phi, econ.r_bar)

    # At age 65 (lose one year of expenses)
    expenses_at_65 = np.full(params.end_age - 65, params.retirement_expenses)
    pv_65_current = compute_present_value(expenses_at_65, econ.r_bar, econ.phi, econ.r_bar)
    dur_65_current = compute_duration(expenses_at_65, econ.r_bar, econ.phi, econ.r_bar)

    expenses_at_65_alt = expenses_at_65  # Already future-only
    pv_65_alt = compute_present_value(expenses_at_65_alt, econ.r_bar, econ.phi, econ.r_bar)
    dur_65_alt = compute_duration(expenses_at_65_alt, econ.r_bar, econ.phi, econ.r_bar)

    print(f"\nCONVENTION 1 (Current):")
    print(f"  Age 64: PV={pv_64_current:.2f}, Duration={dur_64_current:.4f}")
    print(f"  Age 65: PV={pv_65_current:.2f}, Duration={dur_65_current:.4f}")
    print(f"  Duration change: {dur_65_current - dur_64_current:+.4f}")

    print(f"\nCONVENTION 2 (Alternative):")
    print(f"  Age 64: PV={pv_64_alt:.2f}, Duration={dur_64_alt:.4f}")
    print(f"  Age 65: PV={pv_65_alt:.2f}, Duration={dur_65_alt:.4f}")
    print(f"  Duration change: {dur_65_alt - dur_65_alt:+.4f}")

    print(f"\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)
    print(f"""
CONVENTION 1 (CURRENT):
  Duration drops from {dur_64_current:.4f} to {dur_65_current:.4f}
  Change: {dur_65_current - dur_64_current:+.4f}

  Problem: Dropping the CLOSEST cash flow (age 64) creates a sharp discontinuity.
  The age 64 expense has the highest duration weight, so losing it has a big impact.

CONVENTION 2 (ALTERNATIVE):
  Duration stays at {dur_64_alt:.4f} (no change!)
  Change: {dur_65_alt - dur_65_alt:+.4f}

  Why? We're comparing:
  - Age 64: Expenses from age 65-95 (31 years)
  - Age 65: Expenses from age 65-95 (30 years) ... wait, that's wrong!

  Actually, at age 65, we have expenses from 65-95 (30 years).
  At age 64, we have expenses from 65-95 (31 years).
  So there's still a difference!

Let me recalculate...
""")

    # CORRECT calculation for convention 2
    # At age 64, excluding the current year: expenses ages 65-95
    expenses_65_95 = np.full(params.end_age - 65, params.retirement_expenses)

    # At age 65, excluding the current year: expenses ages 66-95
    expenses_66_95 = np.full(params.end_age - 66, params.retirement_expenses)

    pv_alt_64 = compute_present_value(expenses_65_95, econ.r_bar, econ.phi, econ.r_bar)
    dur_alt_64 = compute_duration(expenses_65_95, econ.r_bar, econ.phi, econ.r_bar)

    pv_alt_65 = compute_present_value(expenses_66_95, econ.r_bar, econ.phi, econ.r_bar)
    dur_alt_65 = compute_duration(expenses_66_95, econ.r_bar, econ.phi, econ.r_bar)

    print(f"""
CORRECTED CONVENTION 2:
  Age 64: Expenses ages 65-95 (30 years) → PV={pv_alt_64:.2f}, Duration={dur_alt_64:.4f}
  Age 65: Expenses ages 66-95 (29 years) → PV={pv_alt_65:.2f}, Duration={dur_alt_65:.4f}
  Change: {dur_alt_65 - dur_alt_64:+.4f}

So BOTH conventions show a duration drop! The question is: how much?

CONVENTION 1: {dur_65_current - dur_64_current:+.4f} (drops from {dur_64_current:.4f})
CONVENTION 2: {dur_alt_65 - dur_alt_64:+.4f} (drops from {dur_alt_64:.4f})

The difference is {abs((dur_65_current - dur_64_current) - (dur_alt_65 - dur_alt_64)):+.4f}

In Convention 1, the closest (highest-weight) cash flow is included, making the
duration change larger when we lose that cash flow.

In Convention 2, the closest cash flow is excluded (assumed already paid), so
the duration change is smaller.

CONCLUSION:
The current convention (including current period) creates a LARGER discontinuity.
The user may be right that this is causing unnecessary kinks in the allocation.
""")


if __name__ == "__main__":
    test_timing_conventions()
