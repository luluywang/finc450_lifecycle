#!/usr/bin/env python3
"""
Detailed analysis of the portfolio kink at retirement.

Traces through the exact calculations to identify the source of the discontinuity.
"""

import numpy as np
from core import (
    LifecycleParams,
    EconomicParams,
    simulate_with_strategy,
    LDIStrategy,
    RuleOfThumbStrategy,
)


def analyze_kink():
    """Analyze the portfolio kink by examining component changes."""

    # Default parameters
    params = LifecycleParams()
    econ_params = EconomicParams()

    # Zero shocks for deterministic median path
    total_years = params.end_age - params.start_age
    rate_shocks = np.zeros((1, total_years))
    stock_shocks = np.zeros((1, total_years))

    # Run simulation with LDI (the problematic strategy)
    ldi = LDIStrategy()
    result = simulate_with_strategy(ldi, params, econ_params, rate_shocks, stock_shocks)

    retirement_idx = params.retirement_age - params.start_age
    before = retirement_idx - 1
    after = retirement_idx

    def get_1d(arr):
        """Extract 1D array from result."""
        if arr is None:
            return None
        return arr if arr.ndim == 1 else arr[0]

    # Extract paths
    hc = get_1d(result.human_capital)
    hc_bond = get_1d(result.hc_bond_component)
    hc_stock = get_1d(result.hc_stock_component)
    hc_cash = get_1d(result.hc_cash_component)

    exp_bond = get_1d(result.exp_bond_component)
    exp_cash = get_1d(result.exp_cash_component)

    pv_exp = get_1d(result.pv_expenses)
    fw = get_1d(result.financial_wealth)
    net_worth = get_1d(result.net_worth)

    target_fin_stock = get_1d(result.target_fin_stock)
    target_fin_bond = get_1d(result.target_fin_bond)
    target_fin_cash = get_1d(result.target_fin_cash)

    stock_weight = get_1d(result.stock_weight)
    bond_weight = get_1d(result.bond_weight)
    cash_weight = get_1d(result.cash_weight)

    print("\n" + "="*100)
    print("RETIREMENT KINK ANALYSIS - Component Decomposition")
    print("="*100)

    print(f"\n{'YEAR BEFORE RETIREMENT':-^100}")
    print(f"Age: {params.start_age + before}")
    print(f"\nWealth:")
    print(f"  HC = {hc[before]:.4f}")
    print(f"  FW = {fw[before]:.4f}")
    print(f"  PV(Expenses) = {pv_exp[before]:.4f}")
    print(f"  Net Worth = HC + FW - PV(Exp) = {hc[before]:.4f} + {fw[before]:.4f} - {pv_exp[before]:.4f}")
    print(f"           = {net_worth[before]:.4f}")

    print(f"\nHuman Capital Components:")
    print(f"  HC = {hc[before]:.4f}")
    print(f"    HC_stock = {hc_stock[before]:.4f}")
    print(f"    HC_bond  = {hc_bond[before]:.4f} ← CRITICAL: This will disappear at retirement!")
    print(f"    HC_cash  = {hc_cash[before]:.4f}")

    print(f"\nExpense Components:")
    print(f"  PV(Exp) = {pv_exp[before]:.4f}")
    print(f"    Exp_bond = {exp_bond[before]:.4f}")
    print(f"    Exp_cash = {exp_cash[before]:.4f}")

    print(f"\nTarget Allocation (LDI):")
    print(f"  Surplus = max(0, Net Worth) = {max(0, net_worth[before]):.4f}")
    print(f"  target_fin_stock = target_stock × surplus - HC_stock")
    print(f"                   = {target_fin_stock[before]:.4f}")
    print(f"  target_fin_bond = target_bond × surplus - HC_bond + Exp_bond")
    print(f"                  = {target_fin_bond[before]:.4f}")
    print(f"  target_fin_cash = target_cash × surplus - HC_cash + Exp_cash")
    print(f"                  = {target_fin_cash[before]:.4f}")

    total_target = target_fin_stock[before] + target_fin_bond[before] + target_fin_cash[before]
    print(f"  Total target = {total_target:.4f}")

    print(f"\nActual Weights (after normalization):")
    print(f"  Stock weight = {stock_weight[before]:.4f}")
    print(f"  Bond weight  = {bond_weight[before]:.4f} ← Current allocation")
    print(f"  Cash weight  = {cash_weight[before]:.4f}")

    print(f"\n{'YEAR AFTER RETIREMENT':-^100}")
    print(f"Age: {params.start_age + after}")
    print(f"\nWealth:")
    print(f"  HC = {hc[after]:.4f} ← JUMP: Goes from {hc[before]:.4f} to {hc[after]:.4f}")
    print(f"  FW = {fw[after]:.4f}")
    print(f"  PV(Expenses) = {pv_exp[after]:.4f}")
    print(f"  Net Worth = {net_worth[after]:.4f}")

    print(f"\nHuman Capital Components:")
    print(f"  HC = {hc[after]:.4f}")
    print(f"    HC_stock = {hc_stock[after]:.4f}")
    print(f"    HC_bond  = {hc_bond[after]:.4f} ← Loss of {hc_bond[before] - hc_bond[after]:.4f}")
    print(f"    HC_cash  = {hc_cash[after]:.4f}")

    print(f"\nExpense Components:")
    print(f"  PV(Exp) = {pv_exp[after]:.4f}")
    print(f"    Exp_bond = {exp_bond[after]:.4f}")
    print(f"    Exp_cash = {exp_cash[after]:.4f}")

    print(f"\nTarget Allocation (LDI):")
    print(f"  Surplus = max(0, Net Worth) = {max(0, net_worth[after]):.4f}")
    print(f"  target_fin_stock = target_stock × surplus - HC_stock")
    print(f"                   = {target_fin_stock[after]:.4f}")
    print(f"  target_fin_bond = target_bond × surplus - HC_bond + Exp_bond")
    print(f"                  = {target_fin_bond[after]:.4f}")
    print(f"    • Loss of HC_bond contribution: {hc_bond[before] - hc_bond[after]:+.4f}")
    print(f"    • Change in Exp_bond: {exp_bond[after] - exp_bond[before]:+.4f}")
    print(f"  target_fin_cash = target_cash × surplus - HC_cash + Exp_cash")
    print(f"                  = {target_fin_cash[after]:.4f}")

    total_target = target_fin_stock[after] + target_fin_bond[after] + target_fin_cash[after]
    print(f"  Total target = {total_target:.4f}")

    print(f"\nActual Weights (after normalization):")
    print(f"  Stock weight = {stock_weight[after]:.4f}")
    print(f"  Bond weight  = {bond_weight[after]:.4f} ← New allocation (changed by {bond_weight[after] - bond_weight[before]:+.4f})")
    print(f"  Cash weight  = {cash_weight[after]:.4f}")

    print(f"\n{'DIAGNOSIS':-^100}")
    print(f"\n1. DISCONTINUOUS LOSS OF HUMAN CAPITAL AT RETIREMENT")
    print(f"   HC goes from {hc[before]:.4f} to {hc[after]:.4f}")
    print(f"   This is economically correct but creates a discontinuity in duration.")

    print(f"\n2. LOSS OF HC BOND OFFSET")
    print(f"   HC_bond goes from {hc_bond[before]:.4f} to {hc_bond[after]:.4f} (Δ = {hc_bond[after] - hc_bond[before]:.4f})")
    print(f"   This affects target_fin_bond = target_bond × surplus - HC_bond + Exp_bond")
    print(f"   When HC_bond disappears, target_fin_bond INCREASES (less negative offset)")

    hc_bond_contribution = hc_bond[before]
    exp_bond_change = exp_bond[after] - exp_bond[before]
    fw_change = fw[after] - fw[before]
    surplus_change = net_worth[after] - net_worth[before]

    print(f"\n3. SIMULTANEOUS CHANGES")
    print(f"   • Surplus changes by: {surplus_change:+.4f}")
    print(f"   • FW changes by: {fw_change:+.4f}")
    print(f"   • PV(Exp) changes by: {pv_exp[after] - pv_exp[before]:+.4f}")
    print(f"   • Exp_bond changes by: {exp_bond_change:+.4f}")

    print(f"\n4. BOND WEIGHT KINK")
    bond_weight_change = bond_weight[after] - bond_weight[before]
    print(f"   Bond weight changes from {bond_weight[before]:.4f} to {bond_weight[after]:.4f}")
    print(f"   Δ Bond weight = {bond_weight_change:+.4f} ({bond_weight_change*100:+.2f}%)")

    if abs(bond_weight_change) > 0.01:
        print(f"   ⚠️  SIGNIFICANT KINK DETECTED (>1%)")
    else:
        print(f"   ✓ Small kink (<1%)")

    print("\n" + "="*100)


if __name__ == "__main__":
    analyze_kink()
