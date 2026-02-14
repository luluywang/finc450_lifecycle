#!/usr/bin/env python3
"""
Deeper investigation: Is the kink a sign of a fundamental problem?

Check:
1. How is the expense decomposition behaving at retirement?
2. Is the HC duration calculation breaking down?
3. Is the normalization algorithm doing something weird?
4. What about the surplus calculation?
"""

import numpy as np
import pandas as pd
from core import (
    LifecycleParams,
    EconomicParams,
    simulate_with_strategy,
    LDIStrategy,
)


def investigate():
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

    # Extract all paths
    hc = get_1d(result.human_capital)
    duration_hc = get_1d(result.duration_hc)
    hc_bond = get_1d(result.hc_bond_component)
    hc_cash = get_1d(result.hc_cash_component)

    exp_bond = get_1d(result.exp_bond_component)
    exp_cash = get_1d(result.exp_cash_component)
    duration_exp = get_1d(result.duration_expenses)
    pv_exp = get_1d(result.pv_expenses)

    fw = get_1d(result.financial_wealth)
    net_worth = get_1d(result.net_worth)

    target_fin_stock = get_1d(result.target_fin_stock)
    target_fin_bond = get_1d(result.target_fin_bond)
    target_fin_cash = get_1d(result.target_fin_cash)

    stock_weight = get_1d(result.stock_weight)
    bond_weight = get_1d(result.bond_weight)
    cash_weight = get_1d(result.cash_weight)

    print("\n" + "=" * 120)
    print("DEEPER INVESTIGATION: What's Really Happening at Retirement?")
    print("=" * 120)

    # Create detailed comparison
    data = {
        "Component": [
            "WEALTH",
            "  HC",
            "  FW",
            "  PV(Exp)",
            "  Surplus",
            "",
            "HC DECOMPOSITION",
            "  duration_hc",
            "  hc_bond_frac (duration/20)",
            "  HC_bond",
            "  HC_cash",
            "",
            "EXPENSE DECOMPOSITION",
            "  duration_exp",
            "  exp_bond_frac (duration/20)",
            "  Exp_bond",
            "  Exp_cash",
            "",
            "TARGET ALLOCATION (LDI)",
            "  target_fin_stock",
            "  target_fin_bond",
            "  target_fin_cash",
            "  Total target",
            "",
            "ACTUAL WEIGHTS (After Normalization)",
            "  Stock weight",
            "  Bond weight",
            "  Cash weight",
        ],
        "Age 64 (Before)": [
            "",
            f"{hc[before]:.2f}",
            f"{fw[before]:.2f}",
            f"{pv_exp[before]:.2f}",
            f"{net_worth[before]:.2f}",
            "",
            "",
            f"{duration_hc[before]:.4f}",
            f"{duration_hc[before]/20.0:.4f}",
            f"{hc_bond[before]:.4f}",
            f"{hc_cash[before]:.2f}",
            "",
            "",
            f"{duration_exp[before]:.4f}",
            f"{duration_exp[before]/20.0:.4f}",
            f"{exp_bond[before]:.2f}",
            f"{exp_cash[before]:.2f}",
            "",
            "",
            f"{target_fin_stock[before]:.2f}",
            f"{target_fin_bond[before]:.2f}",
            f"{target_fin_cash[before]:.2f}",
            f"{target_fin_stock[before] + target_fin_bond[before] + target_fin_cash[before]:.2f}",
            "",
            "",
            f"{stock_weight[before]:.4f}",
            f"{bond_weight[before]:.4f}",
            f"{cash_weight[before]:.4f}",
        ],
        "Age 65 (After)": [
            "",
            f"{hc[after]:.2f}",
            f"{fw[after]:.2f}",
            f"{pv_exp[after]:.2f}",
            f"{net_worth[after]:.2f}",
            "",
            "",
            f"{duration_hc[after]:.4f}",
            f"{duration_hc[after]/20.0:.4f}",
            f"{hc_bond[after]:.4f}",
            f"{hc_cash[after]:.2f}",
            "",
            "",
            f"{duration_exp[after]:.4f}",
            f"{duration_exp[after]/20.0:.4f}",
            f"{exp_bond[after]:.2f}",
            f"{exp_cash[after]:.2f}",
            "",
            "",
            f"{target_fin_stock[after]:.2f}",
            f"{target_fin_bond[after]:.2f}",
            f"{target_fin_cash[after]:.2f}",
            f"{target_fin_stock[after] + target_fin_bond[after] + target_fin_cash[after]:.2f}",
            "",
            "",
            f"{stock_weight[after]:.4f}",
            f"{bond_weight[after]:.4f}",
            f"{cash_weight[after]:.4f}",
        ],
        "Change": [
            "",
            f"{hc[after] - hc[before]:+.2f}",
            f"{fw[after] - fw[before]:+.2f}",
            f"{pv_exp[after] - pv_exp[before]:+.2f}",
            f"{net_worth[after] - net_worth[before]:+.2f}",
            "",
            "",
            f"{duration_hc[after] - duration_hc[before]:+.4f}",
            f"{(duration_hc[after] - duration_hc[before])/20.0:+.4f}",
            f"{hc_bond[after] - hc_bond[before]:+.4f}",
            f"{hc_cash[after] - hc_cash[before]:+.2f}",
            "",
            "",
            f"{duration_exp[after] - duration_exp[before]:+.4f}",
            f"{(duration_exp[after] - duration_exp[before])/20.0:+.4f}",
            f"{exp_bond[after] - exp_bond[before]:+.2f}",
            f"{exp_cash[after] - exp_cash[before]:+.2f}",
            "",
            "",
            f"{target_fin_stock[after] - target_fin_stock[before]:+.2f}",
            f"{target_fin_bond[after] - target_fin_bond[before]:+.2f}",
            f"{target_fin_cash[after] - target_fin_cash[before]:+.2f}",
            f"{(target_fin_stock[after] + target_fin_bond[after] + target_fin_cash[after]) - (target_fin_stock[before] + target_fin_bond[before] + target_fin_cash[before]):+.2f}",
            "",
            "",
            f"{stock_weight[after] - stock_weight[before]:+.4f}",
            f"{bond_weight[after] - bond_weight[before]:+.4f}",
            f"{cash_weight[after] - cash_weight[before]:+.4f}",
        ],
    }

    df = pd.DataFrame(data)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    print("\n" + df.to_string(index=False))

    print("\n" + "=" * 120)
    print("ANOMALIES TO INVESTIGATE")
    print("=" * 120)

    print(f"\n1. HC DECOMPOSITION SINGULARITY")
    print(f"   When HC → 0, we get hc_bond → 0 and hc_cash → 0")
    print(f"   But the ratio of bond/cash in HC was:")
    print(f"   Age 64: HC_bond / HC = {hc_bond[before] / hc[before]:.4f}")
    print(f"   Age 65: HC_bond / HC = 0 / 0 = undefined!")
    print(f"")
    print(f"   Question: Is this ratio being preserved correctly at the boundary?")

    print(f"\n2. SURPLUS CHANGE PARADOX")
    print(f"   Age 64: Surplus = {net_worth[before]:.2f}")
    print(f"   Age 65: Surplus = {net_worth[after]:.2f}")
    print(f"   Change = +{net_worth[after] - net_worth[before]:.2f}")
    print(f"")
    print(f"   BUT we're losing HC ({hc[before]:.2f}) and gaining FW ({fw[after] - fw[before]:.2f})")
    print(f"   FW gain is from returns + savings, which is LESS than HC loss.")
    print(f"   So why does surplus INCREASE?")
    print(f"")
    print(f"   Answer: PV(Exp) decreases by {abs(pv_exp[after] - pv_exp[before]):.2f}")
    print(f"   Because the expense stream is now shorter (29 years instead of 30+ years)")

    print(f"\n3. EXPENSE DURATION DISCONTINUITY")
    print(f"   Age 64: duration_exp = {duration_exp[before]:.4f}")
    print(f"   Age 65: duration_exp = {duration_exp[after]:.4f}")
    print(f"   Change = {duration_exp[after] - duration_exp[before]:+.4f}")
    print(f"")
    print(f"   This is ALSO discontinuous! Not just HC.")
    print(f"   Question: Is this discontinuity in expense duration creating a secondary kink?")

    print(f"\n4. TARGET BOND POSITION DECOMPOSITION")
    target_fin_bond_change = target_fin_bond[after] - target_fin_bond[before]
    hc_bond_loss = hc_bond[before] - hc_bond[after]  # This is a NEGATIVE contribution (loss of offset)
    exp_bond_change = exp_bond[after] - exp_bond[before]
    surplus_effect = econ.bond_duration * (net_worth[after] - net_worth[before])

    print(f"   Target_fin_bond changes by: {target_fin_bond_change:+.2f}")
    print(f"")
    print(f"   Decomposing this change:")
    print(f"   - Loss of HC_bond offset: +{hc_bond_loss:.2f} (less negative = need MORE bonds)")
    print(f"   - Change in Exp_bond: {exp_bond_change:+.2f}")
    print(f"   - Surplus effect: depends on target_bond coefficient")
    print(f"")
    print(f"   The issue: We're losing the HC bond offset (+9.8) while also losing")
    print(f"   expense bond position (-83.4), net = -73.6")

    print(f"\n5. POSSIBLE ROOT CAUSE: EXPENSE STREAM COMPOSITION")
    print(f"   At age 64, you're paying for:")
    print(f"     - 1 year of working expenses")
    print(f"     - 30 years of retirement expenses")
    print(f"")
    print(f"   At age 65, you're paying for:")
    print(f"     - 0 years of working expenses")
    print(f"     - 29 years of retirement expenses")
    print(f"")
    print(f"   The COMPOSITION changes. If working_exp ≠ retirement_exp,")
    print(f"   this could cause duration to jump.")
    print(f"")

    from core import compute_expense_profile

    working_exp, retirement_exp = compute_expense_profile(params)
    ret_exp_val = retirement_exp[0] if hasattr(retirement_exp, '__len__') else retirement_exp
    print(f"   Working expenses: {working_exp[0]:.2f}")
    print(f"   Retirement expenses: {ret_exp_val:.2f}")
    print(f"   Are they equal? {np.isclose(working_exp[0], ret_exp_val)}")

    print(f"\n" + "=" * 120)
    print("HYPOTHESIS: The kink is caused by EXPENSE STREAM STRUCTURE, not HC")
    print("=" * 120)
    print(f"""
The expense profile changes at retirement:
  Before: {params.base_expenses:.0f}/year for {params.retirement_age - params.start_age} years, then {params.retirement_expenses:.0f}/year for {params.end_age - params.retirement_age} years
  After: {params.retirement_expenses:.0f}/year for {params.end_age - params.retirement_age} years

This causes:
  - Duration of remaining expenses to shift
  - Exp_bond to jump from 1648 to 1565
  - The entire asset allocation to rebalance

Combined with the HC discontinuity, this creates the kink.

But the question is: Is this kink NECESSARY or is there a smoothing issue?
""")


if __name__ == "__main__":
    investigate()
