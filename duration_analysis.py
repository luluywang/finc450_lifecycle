#!/usr/bin/env python3
"""
Analyze whether bond_duration represents Macaulay or modified duration,
and whether compute_duration returns the same type as bond_duration.
"""

import numpy as np
from core import (
    EconomicParams,
    LifecycleParams,
    compute_present_value,
    compute_duration,
    effective_duration,
    zero_coupon_price,
)


def analyze_duration_types():
    """Analyze the duration types used throughout the code."""

    econ = EconomicParams()
    params = LifecycleParams()

    print("=" * 100)
    print("DURATION TYPE ANALYSIS: Macaulay vs Modified")
    print("=" * 100)

    print(f"\nEconomic Parameters:")
    print(f"  r_bar = {econ.r_bar} (2%)")
    print(f"  phi = {econ.phi} (1.0 = no mean reversion = random walk)")
    print(f"  bond_duration = {econ.bond_duration}")
    print(f"  sigma_r = {econ.sigma_r}")

    print("\n" + "=" * 100)
    print("1. EFFECTIVE_DURATION FORMULA")
    print("=" * 100)

    print(f"""
effective_duration(tau, phi) computes:
    B(tau) = (1 - phi^tau) / (1 - phi)

With phi = {econ.phi}:
    B(tau) = (1 - {econ.phi}^tau) / (1 - {econ.phi})
    B(tau) = tau  (because phi = 1.0 = random walk)

This B(tau) is the "rate duration" - it's the sensitivity of zero-coupon
bond price to the current short rate r in the Vasicek model:

    P(tau) = exp(-tau*r_bar - B(tau)*(r - r_bar))
    dP/P = -B(tau) * dr
""")

    print("This is essentially MODIFIED DURATION because it directly gives price elasticity!")

    print("\n" + "=" * 100)
    print("2. COMPUTE_DURATION FUNCTION")
    print("=" * 100)

    print(f"""
compute_duration() computes a PV-weighted average of effective durations:

    D = sum(CF_t * P_t * B_t) / PV

    where:
    - CF_t = cashflow at time t
    - P_t = zero_coupon_price(r, t, r_bar, phi) = discounted price
    - B_t = effective_duration(t, phi)
    - PV = present value of all cashflows

Example: Earnings stream
""")

    # Compute HC duration
    from core import compute_earnings_profile

    earnings_profile = compute_earnings_profile(params)
    r = econ.r_bar
    phi = econ.phi
    r_bar = econ.r_bar

    hc_pv = compute_present_value(earnings_profile, r, phi, r_bar)
    hc_duration = compute_duration(earnings_profile, r, phi, r_bar)

    print(f"  Earnings profile (40 years, 1st year): {earnings_profile[:5]}")
    print(f"  PV of earnings = {hc_pv:.2f}")
    print(f"  Duration of earnings = {hc_duration:.4f}")

    print(f"\n  Expected duration ≈ weighted average of times")
    print(f"  (should be around 15-20 for a 40-year stream with front-loaded value)")

    print("\n" + "=" * 100)
    print("3. WHAT IS bond_duration = 20.0?")
    print("=" * 100)

    print(f"""
The parameter bond_duration = {econ.bond_duration} appears to be a REFERENCE DURATION
used for:

a) NORMALIZATION in HC decomposition:
   hc_bond_frac = duration_hc / bond_duration

   This compares the computed duration of earnings against a standard duration.

b) VOLATILITY CALCULATION in MV optimization:
   sigma_b = bond_duration * sigma_r = {econ.bond_duration} * {econ.sigma_r} = {econ.bond_duration * econ.sigma_r:.5f}

   This determines bond volatility and hence portfolio risk.

c) BOND RETURN APPROXIMATION:
   bond_return = r - bond_duration * Δr

   This approximates bond returns using duration.

CRITICAL INSIGHT:
Since effective_duration already returns MODIFIED DURATION (price elasticity),
and bond_duration is compared to it directly, bond_duration should also be
MODIFIED DURATION.

But wait - 20.0 is a very long duration. Let me check if this makes sense...
""")

    print("\n" + "=" * 100)
    print("4. IS bond_duration = 20.0 REASONABLE?")
    print("=" * 100)

    print(f"""
With r_bar = {econ.r_bar}, modified duration would be:
    D* = D_macaulay / (1 + r_bar)
    D* = D / {1 + econ.r_bar:.4f}
    D* ≈ 0.98 * D

So if bond_duration = 20.0 is modified duration, the corresponding
Macaulay duration would be:
    D_macaulay = bond_duration * (1 + r_bar)
    D_macaulay = {econ.bond_duration} * {1 + econ.r_bar:.4f}
    D_macaulay ≈ {econ.bond_duration * (1 + econ.r_bar):.2f}

This would correspond to a bond portfolio with ~{econ.bond_duration * (1 + econ.r_bar):.0f} years maturity,
which is a LONG-DURATION portfolio (like 20-30 year bonds).

For comparison:
- 10-year Treasury: duration ≈ 8-9 years
- 20-year Treasury: duration ≈ 15-17 years
- 30-year Treasury: duration ≈ 18-20 years

So bond_duration = 20 seems to represent a LONG BOND portfolio duration.
This is reasonable for a lifecycle model (matching long-term liabilities).
""")

    print("\n" + "=" * 100)
    print("5. CONSISTENCY CHECK: HC vs Bond Duration")
    print("=" * 100)

    hc_duration_computed = compute_duration(earnings_profile, r, phi, r_bar)

    print(f"""
Human Capital (earnings stream):
  Computed duration = {hc_duration_computed:.4f}

Bond Portfolio:
  Reference duration = {econ.bond_duration}

Ratio = {hc_duration_computed / econ.bond_duration:.4f}

This means:
  HC_bond_frac = {hc_duration_computed / econ.bond_duration:.4f}

  Of the non-stock HC (which is {100 * (1 - params.stock_beta_human_capital):.0f}% of HC),
  {hc_duration_computed / econ.bond_duration * 100:.2f}% is treated as bond-like,
  {(1 - hc_duration_computed / econ.bond_duration) * 100:.2f}% is treated as cash-like.

This makes sense: HC is shorter duration than the bond portfolio,
so it's partially cash-like.
""")

    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)

    print(f"""
1. effective_duration(tau, phi) returns MODIFIED DURATION (price elasticity)

2. compute_duration() returns a PV-weighted average of effective durations,
   also in MODIFIED DURATION terms

3. bond_duration = 20.0 should be interpreted as:
   - A reference modified duration for the bond portfolio
   - When phi=1.0 (random walk), this directly enters the bond return formula
   - When phi<1.0 (mean reverting), the effective duration would be less

4. The HC decomposition:
   hc_bond_frac = duration_hc / bond_duration

   Compares computed HC modified duration to the reference bond duration.
   Both are in the same units, so this is CORRECT.

5. The issue of "modified vs Macaulay" is a RED HERRING:
   The code consistently uses EFFECTIVE DURATION (= modified duration in Vasicek model)
   throughout. There's no mismatch between duration types.

POSSIBLE REMAINING ISSUE:
   The kink at retirement might still be caused by something else.
   Let me check if the issue is in how duration changes discontinuously...
""")


if __name__ == "__main__":
    analyze_duration_types()
