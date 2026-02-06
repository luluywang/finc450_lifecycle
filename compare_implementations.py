#!/usr/bin/env python3
"""
Python vs TypeScript Implementation Comparison Script.

This script exports all key values from the Python implementation to JSON
for comparison with the TypeScript web visualizer.

Usage:
    python compare_implementations.py > output/python_verification.json
    python compare_implementations.py --pretty  # Human-readable output
    python compare_implementations.py --compare output/typescript_verification.json
"""

import json
import sys
import argparse
import numpy as np
from typing import Dict, Any, List

from core import (
    EconomicParams,
    LifecycleParams,
    MonteCarloParams,
)
from core.economics import (
    effective_duration,
    zero_coupon_price,
    compute_present_value,
    compute_duration,
    compute_mv_optimal_allocation,
    compute_full_merton_allocation,
    generate_correlated_shocks,
    simulate_interest_rates,
    simulate_stock_returns,
    compute_duration_approx_returns,
)
from core.simulation import (
    compute_static_pvs,
    decompose_hc_to_components,
    decompose_expenses_to_components,
    compute_earnings_profile,
    compute_expense_profile,
    compute_lifecycle_median_path,
    simulate_paths,
)


def numpy_to_python(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(v) for v in obj]
    return obj


def compute_economic_functions(econ: EconomicParams) -> Dict[str, Any]:
    """Test core economic functions with standard test values."""
    return {
        "effective_duration_20_085": effective_duration(20, 0.85),
        "effective_duration_20_100": effective_duration(20, 1.0),
        "effective_duration_10_085": effective_duration(10, 0.85),
        "effective_duration_5_085": effective_duration(5, 0.85),
        "zero_coupon_price_002_20": zero_coupon_price(0.02, 20, 0.02, 1.0),
        "zero_coupon_price_003_20": zero_coupon_price(0.03, 20, 0.02, 1.0),
        "zero_coupon_price_002_10_085": zero_coupon_price(0.02, 10, 0.02, 0.85),
        "mu_bond": econ.mu_bond,
        "mu_bond_formula": econ.bond_sharpe * econ.bond_duration * econ.sigma_r,
    }


def compute_pv_functions(econ: EconomicParams) -> Dict[str, Any]:
    """Test PV and duration functions with sample cashflows."""
    # Test cashflows: 40 years of $100 earnings (working years)
    earnings = np.full(40, 100.0)
    # Test cashflows: 30 years of $100 expenses (retirement)
    retirement_exp = np.full(30, 100.0)

    r = econ.r_bar
    phi = econ.phi

    return {
        # PV tests with flat discounting
        "pv_earnings_40yr_flat": compute_present_value(earnings, r),
        "pv_expenses_30yr_flat": compute_present_value(retirement_exp, r),
        # PV tests with mean-reverting term structure
        "pv_earnings_40yr_vcv": compute_present_value(earnings, r, phi, r),
        "pv_expenses_30yr_vcv": compute_present_value(retirement_exp, r, phi, r),
        # Duration tests
        "duration_earnings_40yr_flat": compute_duration(earnings, r),
        "duration_expenses_30yr_flat": compute_duration(retirement_exp, r),
        "duration_earnings_40yr_vcv": compute_duration(earnings, r, phi, r),
        "duration_expenses_30yr_vcv": compute_duration(retirement_exp, r, phi, r),
    }


def compute_mv_optimization(econ: EconomicParams, params: LifecycleParams) -> Dict[str, Any]:
    """Test MV optimization with default parameters."""
    # Unconstrained
    stock_u, bond_u, cash_u = compute_full_merton_allocation(
        mu_stock=econ.mu_excess,
        mu_bond=econ.mu_bond,
        sigma_s=econ.sigma_s,
        sigma_r=econ.sigma_r,
        rho=econ.rho,
        duration=econ.bond_duration,
        gamma=params.gamma
    )

    # Constrained (no short)
    stock_c, bond_c, cash_c = compute_mv_optimal_allocation(
        mu_stock=econ.mu_excess,
        mu_bond=econ.mu_bond,
        sigma_s=econ.sigma_s,
        sigma_r=econ.sigma_r,
        rho=econ.rho,
        duration=econ.bond_duration,
        gamma=params.gamma
    )

    return {
        "target_stock_unconstrained": stock_u,
        "target_bond_unconstrained": bond_u,
        "target_cash_unconstrained": cash_u,
        "target_stock": stock_c,
        "target_bond": bond_c,
        "target_cash": cash_c,
        "gamma": params.gamma,
    }


def compute_lifecycle_arrays(params: LifecycleParams, econ: EconomicParams) -> Dict[str, Any]:
    """Compute earnings, expenses, and other lifecycle arrays."""
    working_years = params.retirement_age - params.start_age
    retirement_years = params.end_age - params.retirement_age
    total_years = params.end_age - params.start_age

    # Compute profiles
    earnings_profile = compute_earnings_profile(params)
    working_exp, retirement_exp = compute_expense_profile(params)

    # Build full arrays
    base_earnings = np.zeros(total_years)
    expenses = np.zeros(total_years)
    base_earnings[:working_years] = earnings_profile
    expenses[:working_years] = working_exp
    expenses[working_years:] = retirement_exp

    # Compute PV values and durations
    r = econ.r_bar
    phi = econ.phi
    pv_earnings, pv_expenses, duration_earnings, duration_expenses = compute_static_pvs(
        base_earnings, expenses, working_years, total_years, r, phi
    )

    # Compute HC decomposition
    hc_stock, hc_bond, hc_cash = decompose_hc_to_components(
        pv_earnings, duration_earnings, params.stock_beta_human_capital,
        econ.bond_duration, total_years
    )

    # Compute expense decomposition
    exp_bond, exp_cash = decompose_expenses_to_components(
        pv_expenses, duration_expenses, econ.bond_duration, total_years
    )

    # Ages
    ages = list(range(params.start_age, params.end_age))

    return {
        "ages": ages,
        "earnings": base_earnings.tolist(),
        "expenses": expenses.tolist(),
        "pv_earnings": pv_earnings.tolist(),
        "pv_expenses": pv_expenses.tolist(),
        "duration_earnings": duration_earnings.tolist(),
        "duration_expenses": duration_expenses.tolist(),
        "human_capital": pv_earnings.tolist(),  # HC = PV(earnings) for bond-like HC
        "hc_stock": hc_stock.tolist(),
        "hc_bond": hc_bond.tolist(),
        "hc_cash": hc_cash.tolist(),
        "exp_bond": exp_bond.tolist(),
        "exp_cash": exp_cash.tolist(),
        # Sample values at key ages for quick comparison
        "pv_earnings_age_25": float(pv_earnings[0]),
        "pv_earnings_age_45": float(pv_earnings[20]),
        "pv_earnings_age_64": float(pv_earnings[39]),
        "pv_expenses_age_25": float(pv_expenses[0]),
        "pv_expenses_age_65": float(pv_expenses[40]),
        "pv_expenses_age_85": float(pv_expenses[60]),
        "duration_earnings_age_25": float(duration_earnings[0]),
        "duration_earnings_age_45": float(duration_earnings[20]),
        "duration_expenses_age_65": float(duration_expenses[40]),
    }


def compute_median_path(params: LifecycleParams, econ: EconomicParams) -> Dict[str, Any]:
    """Compute the deterministic median path (zero shocks)."""
    result = compute_lifecycle_median_path(params, econ)

    # Extract key arrays
    return {
        "ages": result.ages.tolist(),
        "financial_wealth": result.financial_wealth.tolist(),
        "net_worth": result.net_worth.tolist(),
        "stock_weight": result.stock_weight_no_short.tolist(),
        "bond_weight": result.bond_weight_no_short.tolist(),
        "cash_weight": result.cash_weight_no_short.tolist(),
        "total_consumption": result.total_consumption.tolist(),
        "subsistence_consumption": result.subsistence_consumption.tolist(),
        "variable_consumption": result.variable_consumption.tolist(),
        # Sample values at key ages
        "fw_age_25": float(result.financial_wealth[0]),
        "fw_age_45": float(result.financial_wealth[20]),
        "fw_age_65": float(result.financial_wealth[40]),
        "fw_age_85": float(result.financial_wealth[60]),
        "stock_weight_age_25": float(result.stock_weight_no_short[0]),
        "stock_weight_age_45": float(result.stock_weight_no_short[20]),
        "stock_weight_age_65": float(result.stock_weight_no_short[40]),
        "stock_weight_age_85": float(result.stock_weight_no_short[60]),
        "consumption_age_65": float(result.total_consumption[40]),
        "consumption_age_85": float(result.total_consumption[60]),
    }


def compute_simulation_test(params: LifecycleParams, econ: EconomicParams) -> Dict[str, Any]:
    """Run simulation with known shocks for verification."""
    total_years = params.end_age - params.start_age

    # Test with zero shocks (should match median path exactly)
    zero_rate_shocks = np.zeros((1, total_years))
    zero_stock_shocks = np.zeros((1, total_years))

    result = simulate_paths(
        params, econ, zero_rate_shocks, zero_stock_shocks,
        initial_rate=econ.r_bar,
        use_dynamic_revaluation=False
    )

    return {
        "zero_shock_fw": result['financial_wealth_paths'][0].tolist(),
        "zero_shock_hc": result['human_capital_paths'][0].tolist(),
        "zero_shock_stock_weight": result['stock_weight_paths'][0].tolist(),
        "zero_shock_consumption": result['total_consumption_paths'][0].tolist(),
        "target_stock": float(result['target_stock']),
        "target_bond": float(result['target_bond']),
        "target_cash": float(result['target_cash']),
    }


def compute_monte_carlo_stats(params: LifecycleParams, econ: EconomicParams) -> Dict[str, Any]:
    """Run Monte Carlo simulation and compute summary statistics."""
    total_years = params.end_age - params.start_age
    n_sims = 1000
    seed = 42

    rng = np.random.default_rng(seed)
    rate_shocks, stock_shocks = generate_correlated_shocks(
        total_years, n_sims, econ.rho, rng
    )

    result = simulate_paths(
        params, econ, rate_shocks, stock_shocks,
        initial_rate=econ.r_bar,
        use_dynamic_revaluation=True
    )

    # Compute statistics
    final_wealth = result['financial_wealth_paths'][:, -1]
    default_rate = np.mean(result['default_flags'])

    # Wealth percentiles at key ages
    fw_paths = result['financial_wealth_paths']

    return {
        "n_simulations": n_sims,
        "random_seed": seed,
        "terminal_wealth_median": float(np.median(final_wealth)),
        "terminal_wealth_mean": float(np.mean(final_wealth)),
        "terminal_wealth_p5": float(np.percentile(final_wealth, 5)),
        "terminal_wealth_p25": float(np.percentile(final_wealth, 25)),
        "terminal_wealth_p75": float(np.percentile(final_wealth, 75)),
        "terminal_wealth_p95": float(np.percentile(final_wealth, 95)),
        "default_rate": float(default_rate),
        "default_count": int(np.sum(result['default_flags'])),
        # FW percentiles at retirement (age 65)
        "fw_age_65_median": float(np.median(fw_paths[:, 40])),
        "fw_age_65_p5": float(np.percentile(fw_paths[:, 40], 5)),
        "fw_age_65_p95": float(np.percentile(fw_paths[:, 40], 95)),
        # FW percentiles at age 85
        "fw_age_85_median": float(np.median(fw_paths[:, 60])),
        "fw_age_85_p5": float(np.percentile(fw_paths[:, 60], 5)),
        "fw_age_85_p95": float(np.percentile(fw_paths[:, 60], 95)),
    }


def generate_verification_data() -> Dict[str, Any]:
    """Generate complete verification data for comparison."""
    # Use default parameters
    econ = EconomicParams()
    params = LifecycleParams()

    return {
        "metadata": {
            "source": "Python (core module)",
            "version": "1.0",
            "parameters": {
                "r_bar": econ.r_bar,
                "phi": econ.phi,
                "sigma_r": econ.sigma_r,
                "mu_excess": econ.mu_excess,
                "bond_sharpe": econ.bond_sharpe,
                "sigma_s": econ.sigma_s,
                "rho": econ.rho,
                "bond_duration": econ.bond_duration,
                "start_age": params.start_age,
                "retirement_age": params.retirement_age,
                "end_age": params.end_age,
                "initial_earnings": params.initial_earnings,
                "base_expenses": params.base_expenses,
                "gamma": params.gamma,
                "initial_wealth": params.initial_wealth,
                "stock_beta_human_capital": params.stock_beta_human_capital,
            }
        },
        "economic_functions": compute_economic_functions(econ),
        "pv_functions": compute_pv_functions(econ),
        "mv_optimization": compute_mv_optimization(econ, params),
        "lifecycle_arrays": compute_lifecycle_arrays(params, econ),
        "median_path": compute_median_path(params, econ),
        "simulation_test": compute_simulation_test(params, econ),
        "monte_carlo_stats": compute_monte_carlo_stats(params, econ),
    }


def compare_values(py_val: Any, ts_val: Any, path: str, tolerance: float = 0.001) -> List[Dict]:
    """Compare Python and TypeScript values, returning list of discrepancies."""
    discrepancies = []

    if isinstance(py_val, dict) and isinstance(ts_val, dict):
        for key in set(py_val.keys()) | set(ts_val.keys()):
            new_path = f"{path}.{key}"
            if key not in py_val:
                discrepancies.append({
                    "path": new_path,
                    "type": "missing_python",
                    "typescript": ts_val[key],
                })
            elif key not in ts_val:
                discrepancies.append({
                    "path": new_path,
                    "type": "missing_typescript",
                    "python": py_val[key],
                })
            else:
                discrepancies.extend(compare_values(py_val[key], ts_val[key], new_path, tolerance))
    elif isinstance(py_val, list) and isinstance(ts_val, list):
        if len(py_val) != len(ts_val):
            discrepancies.append({
                "path": path,
                "type": "length_mismatch",
                "python_len": len(py_val),
                "typescript_len": len(ts_val),
            })
        else:
            for i, (pv, tv) in enumerate(zip(py_val, ts_val)):
                discrepancies.extend(compare_values(pv, tv, f"{path}[{i}]", tolerance))
    elif isinstance(py_val, (int, float)) and isinstance(ts_val, (int, float)):
        if py_val == 0 and ts_val == 0:
            pass  # Both zero, OK
        elif abs(py_val) < 1e-10 and abs(ts_val) < 1e-10:
            pass  # Both effectively zero, OK
        elif abs(py_val - ts_val) / max(abs(py_val), abs(ts_val)) > tolerance:
            discrepancies.append({
                "path": path,
                "type": "value_mismatch",
                "python": py_val,
                "typescript": ts_val,
                "relative_diff": abs(py_val - ts_val) / max(abs(py_val), abs(ts_val)),
            })
    elif py_val != ts_val:
        discrepancies.append({
            "path": path,
            "type": "value_mismatch",
            "python": str(py_val),
            "typescript": str(ts_val),
        })

    return discrepancies


def generate_comparison_report(py_data: Dict, ts_data: Dict) -> str:
    """Generate markdown comparison report."""
    discrepancies = compare_values(py_data, ts_data, "root")

    report = ["# Python vs TypeScript Implementation Verification Report\n"]
    report.append(f"## Summary\n")
    report.append(f"- Total discrepancies found: {len(discrepancies)}\n")

    if len(discrepancies) == 0:
        report.append("\n✅ **All values match within tolerance!**\n")
    else:
        report.append("\n## Discrepancies\n")

        # Group by type
        by_type = {}
        for d in discrepancies:
            dtype = d['type']
            if dtype not in by_type:
                by_type[dtype] = []
            by_type[dtype].append(d)

        for dtype, items in by_type.items():
            report.append(f"\n### {dtype.replace('_', ' ').title()}\n")
            for item in items[:20]:  # Limit to first 20 per type
                report.append(f"- `{item['path']}`\n")
                if dtype == "value_mismatch":
                    report.append(f"  - Python: {item['python']}\n")
                    report.append(f"  - TypeScript: {item['typescript']}\n")
                    if 'relative_diff' in item:
                        report.append(f"  - Relative diff: {item['relative_diff']:.4%}\n")
                elif dtype == "length_mismatch":
                    report.append(f"  - Python length: {item['python_len']}\n")
                    report.append(f"  - TypeScript length: {item['typescript_len']}\n")
            if len(items) > 20:
                report.append(f"\n... and {len(items) - 20} more\n")

    return "".join(report)


def main():
    parser = argparse.ArgumentParser(description="Python verification data export")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--compare", metavar="FILE", help="Compare with TypeScript JSON file")
    parser.add_argument("--output", "-o", metavar="FILE", help="Output file (default: stdout)")
    args = parser.parse_args()

    data = generate_verification_data()
    data = numpy_to_python(data)

    if args.compare:
        with open(args.compare, 'r') as f:
            ts_data = json.load(f)
        report = generate_comparison_report(data, ts_data)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
        else:
            print(report)
    else:
        json_str = json.dumps(data, indent=2 if args.pretty else None)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_str)
        else:
            print(json_str)


if __name__ == "__main__":
    main()
