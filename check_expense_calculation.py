#!/usr/bin/env python3
"""
Check if the expense duration calculation is correct at the retirement boundary.

The duration of expenses changes from 14.41 to 14.01 - is this correct?
"""

import numpy as np
from core import (
    LifecycleParams,
    EconomicParams,
    compute_expense_profile,
    compute_present_value,
    compute_duration,
)


def check_expense_duration():
    """Verify expense duration calculation."""

    params = LifecycleParams()
    econ = EconomicParams()

    working_exp, retirement_exp = compute_expense_profile(params)
    working_years = params.retirement_age - params.start_age
    total_years = params.end_age - params.start_age

    print("\n" + "=" * 100)
    print("EXPENSE DURATION CALCULATION AT RETIREMENT")
    print("=" * 100)

    print(f"\nParameters:")
    print(f"  Start age: {params.start_age}")
    print(f"  Retirement age: {params.retirement_age}")
    print(f"  End age: {params.end_age}")
    print(f"  Working years: {working_years}")
    print(f"  Total years: {total_years}")

    ret_exp_val = retirement_exp[0] if hasattr(retirement_exp, '__len__') else retirement_exp
    print(f"\nExpense structure:")
    print(f"  Working expense: {working_exp[0]:.2f} per year (for {working_years} years)")
    print(f"  Retirement expense: {ret_exp_val:.2f} per year (for {total_years - working_years} years)")

    # At age 64 (index 39, which is retirement_age - start_age - 1)
    retirement_idx = working_years
    age_64_idx = retirement_idx - 1  # Last working year
    age_65_idx = retirement_idx  # First retirement year

    print(f"\n" + "-" * 100)
    print(f"AT AGE 64 (index {age_64_idx}, last working year)")
    print("-" * 100)

    # Remaining expense stream from age 64 onwards
    remaining_years_64 = total_years - age_64_idx
    expenses_64 = np.zeros(remaining_years_64)

    # Age 64: have 1 more working year (age 64)
    expenses_64[0] = working_exp[age_64_idx]
    # Ages 65+: retirement years (30 years)
    expenses_64[1:] = ret_exp_val

    print(f"\nRemaining expense stream ({remaining_years_64} years):")
    print(f"  Year 0 (age 64): {expenses_64[0]:.2f} (working expense)")
    print(f"  Years 1-30 (ages 65-95): {ret_exp_val:.2f} each")
    print(f"  Total: {np.sum(expenses_64):.2f}")

    pv_64 = compute_present_value(expenses_64, econ.r_bar, econ.phi, econ.r_bar)
    dur_64 = compute_duration(expenses_64, econ.r_bar, econ.phi, econ.r_bar)

    print(f"\nComputed values:")
    print(f"  PV = {pv_64:.2f}")
    print(f"  Duration = {dur_64:.4f}")

    print(f"\n" + "-" * 100)
    print(f"AT AGE 65 (index {age_65_idx}, first retirement year)")
    print("-" * 100)

    # Remaining expense stream from age 65 onwards
    remaining_years_65 = total_years - age_65_idx
    expenses_65 = np.zeros(remaining_years_65)

    # Ages 65-95: retirement years (29 years remaining)
    expenses_65[:] = ret_exp_val

    print(f"\nRemaining expense stream ({remaining_years_65} years):")
    print(f"  Years 0-29 (ages 65-95): {ret_exp_val:.2f} each")
    print(f"  Total: {np.sum(expenses_65):.2f}")

    pv_65 = compute_present_value(expenses_65, econ.r_bar, econ.phi, econ.r_bar)
    dur_65 = compute_duration(expenses_65, econ.r_bar, econ.phi, econ.r_bar)

    print(f"\nComputed values:")
    print(f"  PV = {pv_65:.2f}")
    print(f"  Duration = {dur_65:.4f}")

    print(f"\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)

    print(f"""
Duration Change:
  At age 64: {dur_64:.4f} years
  At age 65: {dur_65:.4f} years
  Change: {dur_65 - dur_64:+.4f} years

This is DISCONTINUOUS but is it WRONG?

Analysis:
1. At age 64, you have:
   - 1 unit of expense at time t=0 (NOW, age 64)
   - 30 units of expense at times t=1..30 (future years 65-95)
   Duration ≈ weighted average of time to expense

2. At age 65, you have:
   - 0 units of expense at time t=0 (that year already passed)
   - 29 units of expense at times t=1..29 (future years 66-95)

3. The duration drops because:
   - You have one fewer year of expenses
   - The "current year" (age 65) gets dropped from the stream

Is this CORRECT or a DISCONTINUITY ARTIFACT?

The expense stream conceptually is:
  Remaining payments for expenses = what you haven't paid yet

At age 64: You still owe age 64 expense + all future
At age 65: You've already paid age 64, so you only owe age 65 onwards

This makes sense! The duration should decrease slightly because you're
deleting the earliest (and thus time-weighted) expense.

BUT the question is: should we be discounting from age 64 or from the
current moment?

The code discounts from the current time (age 64 or 65), which means
it's computing: "duration from NOW until the obligation ends"

That's correct! The duration SHOULD change because the time to nearest
obligation changed.

HOWEVER, there's a question about interpretation:
- Is the expense at age 64 paid AT the beginning of the year?
- Or at the END of the year?
- Or continuously throughout?

If it's paid at the beginning, duration should drop more sharply.
If it's paid continuously, it should be smoother.
""")

    print(f"\n" + "=" * 100)
    print("CHECK: Is the working/retirement expense split the issue?")
    print("=" * 100)

    print(f"""
The code creates the expense stream as:
  expenses[t] = working_exp if t < retirement_age - start_age else retirement_exp

At the retirement boundary (t = {working_years - 1} to {working_years}):
  expenses[{working_years - 1}] = working_exp = {working_exp[0]:.2f}
  expenses[{working_years}] = retirement_exp = {ret_exp_val:.2f}

Since they're equal, the expense AMOUNT doesn't change, only the
count of future years changes.

The duration decrease is purely because we lose one year of expenses,
which is mathematically correct.

But the QUESTION: Is this discontinuity the source of the portfolio kink?

If yes, then the issue is that the model treats retirement as an
instantaneous transition, creating sharp breaks in allocations.

Potential fixes:
1. Smooth the transition over 1-2 years
2. Use a different model for the transition year
3. Recognize this as a necessary feature of the model
""")


if __name__ == "__main__":
    check_expense_duration()
