#!/usr/bin/env python3
"""
Diagnostic script to identify portfolio weight kinks around retirement.

Shows detailed HC, expenses, and allocation decomposition around retirement.
"""

import numpy as np
import pandas as pd
from core import (
    LifecycleParams,
    EconomicParams,
    simulate_with_strategy,
    generate_correlated_shocks,
    LDIStrategy,
    RuleOfThumbStrategy,
)


def diagnose_retirement_kink(strategy, params, econ_params, n_years_around_retirement=10):
    """Run a median-path simulation and print detailed diagnostics around retirement."""

    # Zero shocks for deterministic median path
    total_years = params.end_age - params.start_age
    rate_shocks = np.zeros((1, total_years))
    stock_shocks = np.zeros((1, total_years))

    # Run simulation
    result = simulate_with_strategy(strategy, params, econ_params, rate_shocks, stock_shocks)

    # Extract relevant paths (squeeze to 1D since we have 1 simulation)
    ages = np.arange(params.start_age, params.end_age)
    retirement_idx = params.retirement_age - params.start_age

    start_idx = max(0, retirement_idx - n_years_around_retirement)
    end_idx = min(total_years, retirement_idx + n_years_around_retirement)

    # Create DataFrame with key variables
    # Handle both 1D (single sim) and 2D (Monte Carlo) arrays
    def extract_sim(arr, start_idx, end_idx):
        if arr is None:
            return None
        if arr.ndim == 1:
            return arr[start_idx:end_idx]
        else:
            return arr[0, start_idx:end_idx]

    data = {
        'Age': ages[start_idx:end_idx],
        'HC': extract_sim(result.human_capital, start_idx, end_idx),
        'Duration_HC': extract_sim(result.duration_hc, start_idx, end_idx),
        'HC_Bond': extract_sim(result.hc_bond_component, start_idx, end_idx),
        'HC_Cash': extract_sim(result.hc_cash_component, start_idx, end_idx),
        'PV_Exp': extract_sim(result.pv_expenses, start_idx, end_idx),
        'Duration_Exp': extract_sim(result.duration_expenses, start_idx, end_idx),
        'Exp_Bond': extract_sim(result.exp_bond_component, start_idx, end_idx),
        'Exp_Cash': extract_sim(result.exp_cash_component, start_idx, end_idx),
        'FW': extract_sim(result.financial_wealth, start_idx, end_idx),
        'Net_Worth': extract_sim(result.net_worth, start_idx, end_idx),
        'Stock_Weight': extract_sim(result.stock_weight, start_idx, end_idx),
        'Bond_Weight': extract_sim(result.bond_weight, start_idx, end_idx),
        'Cash_Weight': extract_sim(result.cash_weight, start_idx, end_idx),
    }

    # Only add optional fields if they're populated
    if result.target_fin_stock is not None:
        data['Target_Fin_Stock'] = extract_sim(result.target_fin_stock, start_idx, end_idx)
    if result.target_fin_bond is not None:
        data['Target_Fin_Bond'] = extract_sim(result.target_fin_bond, start_idx, end_idx)
    if result.target_fin_cash is not None:
        data['Target_Fin_Cash'] = extract_sim(result.target_fin_cash, start_idx, end_idx)

    df = pd.DataFrame(data)

    print(f"\n{'='*120}")
    print(f"Retirement Kink Diagnosis for {strategy.name} Strategy")
    print(f"Retirement Age: {params.retirement_age} (index {retirement_idx})")
    print(f"{'='*120}\n")

    # Print with more detail
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print(df.to_string())

    print(f"\n{'='*120}")
    print("Key Observations:")
    print(f"{'='*120}\n")

    # Find the year before and year after retirement
    if retirement_idx - 1 >= start_idx and retirement_idx < len(df):
        before_idx = retirement_idx - start_idx - 1
        after_idx = retirement_idx - start_idx

        if after_idx < len(df):
            print(f"Year BEFORE retirement (age {df.iloc[before_idx]['Age']:.0f}):")
            print(f"  HC: {df.iloc[before_idx]['HC']:.2f}, Duration: {df.iloc[before_idx]['Duration_HC']:.2f}")
            print(f"  HC_Bond: {df.iloc[before_idx]['HC_Bond']:.2f}, HC_Cash: {df.iloc[before_idx]['HC_Cash']:.2f}")
            print(f"  Bond Weight: {df.iloc[before_idx]['Bond_Weight']:.4f}")
            print()

            print(f"Year AFTER retirement (age {df.iloc[after_idx]['Age']:.0f}):")
            print(f"  HC: {df.iloc[after_idx]['HC']:.2f}, Duration: {df.iloc[after_idx]['Duration_HC']:.2f}")
            print(f"  HC_Bond: {df.iloc[after_idx]['HC_Bond']:.2f}, HC_Cash: {df.iloc[after_idx]['HC_Cash']:.2f}")
            print(f"  Bond Weight: {df.iloc[after_idx]['Bond_Weight']:.4f}")
            print()

            # Calculate changes
            hc_change = df.iloc[after_idx]['HC'] - df.iloc[before_idx]['HC']
            hc_bond_change = df.iloc[after_idx]['HC_Bond'] - df.iloc[before_idx]['HC_Bond']
            bond_weight_change = df.iloc[after_idx]['Bond_Weight'] - df.iloc[before_idx]['Bond_Weight']
            target_fin_bond_change = df.iloc[after_idx]['Target_Fin_Bond'] - df.iloc[before_idx]['Target_Fin_Bond']

            print(f"Changes at retirement:")
            print(f"  ΔHC: {hc_change:+.2f}")
            print(f"  ΔHC_Bond: {hc_bond_change:+.2f}")
            print(f"  ΔBond_Weight: {bond_weight_change:+.4f}")
            print(f"  ΔTarget_Fin_Bond: {target_fin_bond_change:+.2f}")
            print()

            # Check if it's a kink
            if abs(bond_weight_change) > 0.01:
                print(f"⚠️  KINK DETECTED: Bond weight changes by {bond_weight_change*100:.2f}% at retirement")


def main():
    # Default parameters
    params = LifecycleParams()
    econ_params = EconomicParams()

    # Test both strategies
    for strategy_class, name in [
        (LDIStrategy, "LDI"),
        (RuleOfThumbStrategy, "RuleOfThumb"),
    ]:
        strategy = strategy_class()
        diagnose_retirement_kink(strategy, params, econ_params, n_years_around_retirement=8)


if __name__ == "__main__":
    main()
