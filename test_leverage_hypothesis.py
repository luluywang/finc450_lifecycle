#!/usr/bin/env python3
"""
Test the LDI leverage hypothesis.

Hypothesis: If we remove the no-short/no-leverage restriction, net worth should
rarely (or never) transition from positive to negative, because the LDI hedge
can be properly sized regardless of financial wealth constraints.

This script:
1. Runs Monte Carlo simulations with constrained vs unconstrained portfolios
2. Compares the frequency of paths where net worth crosses from + to -
3. Compares the distribution of minimum net worth values
4. Generates side-by-side fan charts showing net worth paths
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from core import (
    LifecycleParams,
    EconomicParams,
    MonteCarloParams,
)
from core.simulation import simulate_paths, compute_lifecycle_median_path
from core.economics import generate_correlated_shocks


def compute_zero_crossing_stats(net_worth_paths: np.ndarray) -> dict:
    """
    Compute statistics about net worth zero crossings.

    Returns:
        dict with:
        - paths_crossing_zero: number of paths that cross from + to -
        - crossing_rate: percentage of paths that cross
        - min_net_worth: array of minimum net worth for each path
        - crossing_ages: list of ages where crossings occur
    """
    n_sims, n_periods = net_worth_paths.shape

    paths_crossing_zero = 0
    crossing_ages = []
    min_net_worth = np.min(net_worth_paths, axis=1)

    for sim in range(n_sims):
        path = net_worth_paths[sim, :]
        # Check if net worth transitions from positive to negative
        for t in range(1, n_periods):
            if path[t-1] > 0 and path[t] < 0:
                paths_crossing_zero += 1
                crossing_ages.append(t)
                break  # Only count first crossing

    return {
        'paths_crossing_zero': paths_crossing_zero,
        'crossing_rate': paths_crossing_zero / n_sims * 100,
        'min_net_worth': min_net_worth,
        'crossing_ages': crossing_ages,
    }


def run_comparison(n_simulations: int = 1000, random_seed: int = 42):
    """
    Run Monte Carlo comparison between constrained and unconstrained portfolios.
    """
    # Common parameters
    params_constrained = LifecycleParams(allow_leverage=False)
    params_unconstrained = LifecycleParams(allow_leverage=True)
    econ_params = EconomicParams()

    n_periods = params_constrained.end_age - params_constrained.start_age
    ages = np.arange(params_constrained.start_age, params_constrained.end_age)

    # Generate same random shocks for both simulations
    rng = np.random.default_rng(random_seed)
    rate_shocks, stock_shocks = generate_correlated_shocks(
        n_periods, n_simulations, econ_params.rho, rng
    )

    # Run constrained simulation
    print("Running constrained simulation (no leverage)...")
    result_constrained = simulate_paths(
        params_constrained, econ_params, rate_shocks, stock_shocks,
        initial_rate=econ_params.r_bar,
        use_dynamic_revaluation=True
    )

    # Run unconstrained simulation with same shocks
    print("Running unconstrained simulation (leverage allowed)...")
    result_unconstrained = simulate_paths(
        params_unconstrained, econ_params, rate_shocks, stock_shocks,
        initial_rate=econ_params.r_bar,
        use_dynamic_revaluation=True
    )

    return {
        'ages': ages,
        'constrained': result_constrained,
        'unconstrained': result_unconstrained,
        'params': params_constrained,
        'econ_params': econ_params,
    }


def run_median_path_comparison():
    """
    Run deterministic median path comparison between constrained and unconstrained.
    Uses zero shocks to show expected-value behavior.
    """
    params_constrained = LifecycleParams(allow_leverage=False)
    params_unconstrained = LifecycleParams(allow_leverage=True)
    econ_params = EconomicParams()

    n_periods = params_constrained.end_age - params_constrained.start_age
    ages = np.arange(params_constrained.start_age, params_constrained.end_age)

    # Zero shocks = deterministic median path
    rate_shocks = np.zeros((1, n_periods))
    stock_shocks = np.zeros((1, n_periods))

    # Run both simulations with zero shocks
    result_constrained = simulate_paths(
        params_constrained, econ_params, rate_shocks, stock_shocks,
        initial_rate=econ_params.r_bar,
        use_dynamic_revaluation=False  # Static PV for median path
    )

    result_unconstrained = simulate_paths(
        params_unconstrained, econ_params, rate_shocks, stock_shocks,
        initial_rate=econ_params.r_bar,
        use_dynamic_revaluation=False
    )

    # Extract single path (sim=0) from 2D arrays, keep scalars as-is
    def extract_single_path(result_dict):
        out = {}
        for k, v in result_dict.items():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                out[k] = v[0]
            else:
                out[k] = v
        return out

    return {
        'ages': ages,
        'constrained': extract_single_path(result_constrained),
        'unconstrained': extract_single_path(result_unconstrained),
        'params': params_constrained,
        'econ_params': econ_params,
    }


def print_median_path_stats(median_results: dict):
    """Print median path comparison statistics."""
    ages = median_results['ages']
    constrained = median_results['constrained']
    unconstrained = median_results['unconstrained']

    print("\n" + "="*70)
    print("MEDIAN PATH COMPARISON (Deterministic Expected-Value Path)")
    print("="*70)

    print("\n--- Portfolio Weights Along Median Path ---")
    print(f"{'Age':<8} {'--- Constrained ---':^30} {'--- Unconstrained ---':^30}")
    print(f"{'':8} {'Stock':>8} {'Bond':>8} {'Cash':>8} {'Stock':>10} {'Bond':>10} {'Cash':>10}")
    print("-"*70)

    for target_age in [25, 35, 45, 55, 65, 75, 85]:
        if target_age <= ages[-1]:
            idx = target_age - ages[0]
            c_s = constrained['stock_weight_paths'][idx]
            c_b = constrained['bond_weight_paths'][idx]
            c_c = constrained['cash_weight_paths'][idx]
            u_s = unconstrained['stock_weight_paths'][idx]
            u_b = unconstrained['bond_weight_paths'][idx]
            u_c = unconstrained['cash_weight_paths'][idx]
            print(f"{target_age:<8} {c_s:>8.1%} {c_b:>8.1%} {c_c:>8.1%} {u_s:>10.1%} {u_b:>10.1%} {u_c:>10.1%}")

    print("\n--- Target Financial Holdings (Before Constraints) ---")
    print(f"{'Age':<8} {'--- Target Holdings ($000s) ---':^50}")
    print(f"{'':8} {'Stock':>12} {'Bond':>12} {'Cash':>12} {'FW':>12}")
    print("-"*70)

    for target_age in [25, 35, 45, 55, 65, 75, 85]:
        if target_age <= ages[-1]:
            idx = target_age - ages[0]
            t_s = constrained['target_fin_stocks_paths'][idx]
            t_b = constrained['target_fin_bonds_paths'][idx]
            t_c = constrained['target_fin_cash_paths'][idx]
            fw = constrained['financial_wealth_paths'][idx]
            print(f"{target_age:<8} {t_s:>12.1f} {t_b:>12.1f} {t_c:>12.1f} {fw:>12.1f}")

    print("\n--- Wealth Comparison Along Median Path ---")
    print(f"{'Age':<8} {'Financial Wealth':^24} {'Net Worth':^24}")
    print(f"{'':8} {'Constrained':>12} {'Unconstrained':>12} {'Constrained':>12} {'Unconstrained':>12}")
    print("-"*70)

    for target_age in [25, 35, 45, 55, 65, 75, 85, 95]:
        if target_age <= ages[-1]:
            idx = target_age - ages[0]
            c_fw = constrained['financial_wealth_paths'][idx]
            u_fw = unconstrained['financial_wealth_paths'][idx]
            c_nw = constrained['net_worth_paths'][idx]
            u_nw = unconstrained['net_worth_paths'][idx]
            print(f"{target_age:<8} {c_fw:>12.1f} {u_fw:>12.1f} {c_nw:>12.1f} {u_nw:>12.1f}")

    print("\n--- Key Insight: Why Do They Differ? ---")
    print("On the median path (zero shocks), both strategies start identically.")
    print("Differences arise because:")
    print("  - Constrained: Clips negative weights to 0, normalizes to sum=1")
    print("  - Unconstrained: Uses raw target weights (can be >100% or negative)")
    print("\nWith expected returns (no randomness), unconstrained strategy can:")
    print("  - Take leveraged positions when target holdings > financial wealth")
    print("  - Short assets when target holdings are negative")


def create_median_path_figure(median_results: dict):
    """Create figure showing median path comparison."""
    ages = median_results['ages']
    constrained = median_results['constrained']
    unconstrained = median_results['unconstrained']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = ['#1f77b4', '#ff7f0e']

    # Row 1: Allocations
    # Stock weights
    ax1 = axes[0, 0]
    ax1.plot(ages, constrained['stock_weight_paths'], color=colors[0],
             label='Constrained', linewidth=2)
    ax1.plot(ages, unconstrained['stock_weight_paths'], color=colors[1],
             label='Unconstrained', linewidth=2, linestyle='--')
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=65, color='gray', linestyle=':', alpha=0.7)
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Weight')
    ax1.set_title('Stock Weight')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bond weights
    ax2 = axes[0, 1]
    ax2.plot(ages, constrained['bond_weight_paths'], color=colors[0],
             label='Constrained', linewidth=2)
    ax2.plot(ages, unconstrained['bond_weight_paths'], color=colors[1],
             label='Unconstrained', linewidth=2, linestyle='--')
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=65, color='gray', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Weight')
    ax2.set_title('Bond Weight')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Cash weights
    ax3 = axes[0, 2]
    ax3.plot(ages, constrained['cash_weight_paths'], color=colors[0],
             label='Constrained', linewidth=2)
    ax3.plot(ages, unconstrained['cash_weight_paths'], color=colors[1],
             label='Unconstrained', linewidth=2, linestyle='--')
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax3.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(x=65, color='gray', linestyle=':', alpha=0.7)
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Weight')
    ax3.set_title('Cash Weight')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Row 2: Wealth measures
    # Financial Wealth
    ax4 = axes[1, 0]
    ax4.plot(ages, constrained['financial_wealth_paths'], color=colors[0],
             label='Constrained', linewidth=2)
    ax4.plot(ages, unconstrained['financial_wealth_paths'], color=colors[1],
             label='Unconstrained', linewidth=2, linestyle='--')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(x=65, color='gray', linestyle=':', alpha=0.7)
    ax4.set_xlabel('Age')
    ax4.set_ylabel('$000s')
    ax4.set_title('Financial Wealth')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Net Worth
    ax5 = axes[1, 1]
    ax5.plot(ages, constrained['net_worth_paths'], color=colors[0],
             label='Constrained', linewidth=2)
    ax5.plot(ages, unconstrained['net_worth_paths'], color=colors[1],
             label='Unconstrained', linewidth=2, linestyle='--')
    ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax5.axvline(x=65, color='gray', linestyle=':', alpha=0.7)
    ax5.set_xlabel('Age')
    ax5.set_ylabel('$000s')
    ax5.set_title('Net Worth (HC + FW - PV Expenses)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Target Holdings vs FW
    ax6 = axes[1, 2]
    ax6.plot(ages, constrained['target_fin_stocks_paths'], color='green',
             label='Target Stock', linewidth=1.5)
    ax6.plot(ages, constrained['target_fin_bonds_paths'], color='blue',
             label='Target Bond', linewidth=1.5)
    ax6.plot(ages, constrained['target_fin_cash_paths'], color='orange',
             label='Target Cash', linewidth=1.5)
    ax6.plot(ages, constrained['financial_wealth_paths'], color='black',
             label='Financial Wealth', linewidth=2, linestyle='--')
    ax6.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax6.axvline(x=65, color='gray', linestyle=':', alpha=0.7)
    ax6.set_xlabel('Age')
    ax6.set_ylabel('$000s')
    ax6.set_title('Target Holdings vs Financial Wealth')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Median Path Comparison: Constrained vs Unconstrained Portfolio',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def print_comparison_stats(results: dict):
    """Print comparison statistics to console."""
    constrained_nw = results['constrained']['net_worth_paths']
    unconstrained_nw = results['unconstrained']['net_worth_paths']

    stats_constrained = compute_zero_crossing_stats(constrained_nw)
    stats_unconstrained = compute_zero_crossing_stats(unconstrained_nw)

    print("\n" + "="*70)
    print("LEVERAGE HYPOTHESIS TEST RESULTS")
    print("="*70)

    print("\n--- Zero Crossing Statistics ---")
    print(f"{'Metric':<40} {'Constrained':>12} {'Unconstrained':>12}")
    print("-"*70)
    print(f"{'Paths crossing from + to -':<40} {stats_constrained['paths_crossing_zero']:>12} {stats_unconstrained['paths_crossing_zero']:>12}")
    print(f"{'Crossing rate (%)':<40} {stats_constrained['crossing_rate']:>12.2f} {stats_unconstrained['crossing_rate']:>12.2f}")

    print("\n--- Minimum Net Worth Distribution ---")
    print(f"{'Percentile':<40} {'Constrained':>12} {'Unconstrained':>12}")
    print("-"*70)

    for p in [1, 5, 10, 25, 50]:
        c_val = np.percentile(stats_constrained['min_net_worth'], p)
        u_val = np.percentile(stats_unconstrained['min_net_worth'], p)
        print(f"{f'{p}th percentile ($000s)':<40} {c_val:>12.1f} {u_val:>12.1f}")

    print("\n--- Net Worth at Key Ages (Median) ---")
    ages = results['ages']
    print(f"{'Age':<40} {'Constrained':>12} {'Unconstrained':>12}")
    print("-"*70)

    for target_age in [40, 50, 65, 75, 85, 95]:
        if target_age <= ages[-1]:
            idx = target_age - ages[0]
            c_median = np.median(constrained_nw[:, idx])
            u_median = np.median(unconstrained_nw[:, idx])
            print(f"{f'Age {target_age} ($000s)':<40} {c_median:>12.1f} {u_median:>12.1f}")

    print("\n--- Financial Wealth at Key Ages (Median) ---")
    constrained_fw = results['constrained']['financial_wealth_paths']
    unconstrained_fw = results['unconstrained']['financial_wealth_paths']

    print(f"{'Age':<40} {'Constrained':>12} {'Unconstrained':>12}")
    print("-"*70)

    for target_age in [40, 50, 65, 75, 85, 95]:
        if target_age <= ages[-1]:
            idx = target_age - ages[0]
            c_median = np.median(constrained_fw[:, idx])
            u_median = np.median(unconstrained_fw[:, idx])
            print(f"{f'Age {target_age} ($000s)':<40} {c_median:>12.1f} {u_median:>12.1f}")

    print("\n" + "="*70)

    return stats_constrained, stats_unconstrained


def create_comparison_figure(results: dict, stats_constrained: dict, stats_unconstrained: dict):
    """Create side-by-side fan charts comparing net worth paths."""
    ages = results['ages']
    constrained_nw = results['constrained']['net_worth_paths']
    unconstrained_nw = results['unconstrained']['net_worth_paths']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    percentiles = [5, 25, 50, 75, 95]
    colors = ['#1f77b4', '#ff7f0e']  # Blue for constrained, orange for unconstrained

    # Top left: Constrained net worth fan chart
    ax1 = axes[0, 0]
    plot_fan_chart(ax1, ages, constrained_nw, percentiles, colors[0],
                   "Net Worth: Constrained (No Leverage)")
    highlight_zero_crossings(ax1, ages, constrained_nw, stats_constrained)

    # Top right: Unconstrained net worth fan chart
    ax2 = axes[0, 1]
    plot_fan_chart(ax2, ages, unconstrained_nw, percentiles, colors[1],
                   "Net Worth: Unconstrained (Leverage Allowed)")
    highlight_zero_crossings(ax2, ages, unconstrained_nw, stats_unconstrained)

    # Bottom left: Min net worth histograms
    ax3 = axes[1, 0]
    ax3.hist(stats_constrained['min_net_worth'], bins=50, alpha=0.6,
             label=f"Constrained (crossing: {stats_constrained['crossing_rate']:.1f}%)",
             color=colors[0])
    ax3.hist(stats_unconstrained['min_net_worth'], bins=50, alpha=0.6,
             label=f"Unconstrained (crossing: {stats_unconstrained['crossing_rate']:.1f}%)",
             color=colors[1])
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax3.set_xlabel('Minimum Net Worth ($000s)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Minimum Net Worth')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Bottom right: Financial wealth comparison
    ax4 = axes[1, 1]
    constrained_fw = results['constrained']['financial_wealth_paths']
    unconstrained_fw = results['unconstrained']['financial_wealth_paths']

    # Plot median and percentile bands
    for data, label, color in [(constrained_fw, 'Constrained', colors[0]),
                                (unconstrained_fw, 'Unconstrained', colors[1])]:
        median = np.median(data, axis=0)
        p25 = np.percentile(data, 25, axis=0)
        p75 = np.percentile(data, 75, axis=0)
        ax4.plot(ages, median, color=color, label=label, linewidth=2)
        ax4.fill_between(ages, p25, p75, color=color, alpha=0.2)

    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax4.set_xlabel('Age')
    ax4.set_ylabel('Financial Wealth ($000s)')
    ax4.set_title('Financial Wealth: Median with IQR')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_fan_chart(ax, ages, data, percentiles, color, title):
    """Plot a fan chart with percentile bands."""
    # Compute percentiles
    pct_values = {p: np.percentile(data, p, axis=0) for p in percentiles}

    # Plot bands
    ax.fill_between(ages, pct_values[5], pct_values[95], alpha=0.15, color=color, label='5-95%')
    ax.fill_between(ages, pct_values[25], pct_values[75], alpha=0.25, color=color, label='25-75%')
    ax.plot(ages, pct_values[50], color=color, linewidth=2, label='Median')

    # Zero line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Zero')

    # Retirement line
    ax.axvline(x=65, color='gray', linestyle=':', alpha=0.7)

    ax.set_xlabel('Age')
    ax.set_ylabel('Net Worth ($000s)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)


def highlight_zero_crossings(ax, ages, data, stats):
    """Highlight paths that cross zero."""
    n_sims = data.shape[0]
    n_highlight = min(5, stats['paths_crossing_zero'])  # Show up to 5 crossing paths

    highlighted = 0
    for sim in range(n_sims):
        if highlighted >= n_highlight:
            break
        path = data[sim, :]
        # Check if this path crosses zero
        for t in range(1, len(path)):
            if path[t-1] > 0 and path[t] < 0:
                ax.plot(ages, path, color='red', alpha=0.4, linewidth=0.5)
                highlighted += 1
                break


def main():
    """Main entry point."""
    print("Testing LDI Leverage Hypothesis")
    print("-" * 40)

    # First: Run median path comparison (deterministic)
    print("\n1. Computing median path comparison...")
    median_results = run_median_path_comparison()
    print_median_path_stats(median_results)

    # Second: Run Monte Carlo comparison
    print("\n2. Running Monte Carlo comparison...")
    results = run_comparison(n_simulations=1000, random_seed=42)

    # Print statistics
    stats_constrained, stats_unconstrained = print_comparison_stats(results)

    # Generate figures
    print("\nGenerating comparison figures...")
    fig_mc = create_comparison_figure(results, stats_constrained, stats_unconstrained)
    fig_median = create_median_path_figure(median_results)

    # Save to PDF
    output_file = 'leverage_hypothesis_test.pdf'
    with PdfPages(output_file) as pdf:
        pdf.savefig(fig_median, bbox_inches='tight')
        pdf.savefig(fig_mc, bbox_inches='tight')

    print(f"\nFigures saved to: {output_file}")
    plt.close(fig_mc)
    plt.close(fig_median)

    # Summary
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    constrained_rate = stats_constrained['crossing_rate']
    unconstrained_rate = stats_unconstrained['crossing_rate']

    if unconstrained_rate < constrained_rate:
        reduction = constrained_rate - unconstrained_rate
        print(f"Allowing leverage REDUCES zero crossings by {reduction:.1f} percentage points")
        print(f"  Constrained:   {constrained_rate:.1f}% of paths cross from + to -")
        print(f"  Unconstrained: {unconstrained_rate:.1f}% of paths cross from + to -")
        if unconstrained_rate == 0:
            print("\nHypothesis CONFIRMED: No paths cross zero with unconstrained LDI hedge")
    elif unconstrained_rate == constrained_rate:
        print(f"No difference in zero crossing rates ({constrained_rate:.1f}%)")
    else:
        print(f"Unexpected: Unconstrained has MORE crossings ({unconstrained_rate:.1f}% vs {constrained_rate:.1f}%)")

    print("="*70)


if __name__ == '__main__':
    main()
