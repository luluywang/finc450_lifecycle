"""
Visualization functions for retirement simulation analysis.

This module provides plotting functions to illustrate key concepts
in life cycle investing and retirement planning.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from retirement_simulation import (
    SimulationResult,
    liability_pv_vectorized,
    liability_duration,
    EconomicParams,
    SimulationParams,
)


# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def plot_duration_matching_intuition(
    rates: np.ndarray,
    wealth_paths_mm: np.ndarray,
    wealth_paths_dm: np.ndarray,
    consumption_target: float,
    horizon: int,
    sample_idx: int = 0,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Show how duration matching protects funded status against interest rate changes.

    This visualization demonstrates that when rates change:
    - Assets and liabilities move together under duration matching
    - Funded ratio stays stable even as rates fluctuate
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    years = np.arange(horizon + 1)
    sample_rates = rates[sample_idx, :]

    # Compute liability PV over time for this path
    liab_pv = np.array([
        liability_pv_vectorized(consumption_target, sample_rates[t:t+1], horizon - t)[0]
        for t in range(horizon + 1)
    ])

    # Panel 1: Interest rate path
    ax = axes[0, 0]
    ax.plot(years, sample_rates * 100, 'b-', linewidth=2)
    ax.axhline(y=3.0, color='gray', linestyle='--', alpha=0.7, label='Long-run mean')
    ax.set_xlabel('Year')
    ax.set_ylabel('Interest Rate (%)')
    ax.set_title('Interest Rate Path')
    ax.legend()
    ax.set_xlim(0, horizon)

    # Panel 2: Assets vs Liabilities for MM strategy
    ax = axes[0, 1]
    ax.plot(years, wealth_paths_mm[sample_idx, :] / 1e6, 'b-',
            linewidth=2, label='Assets (MM)')
    ax.plot(years, liab_pv / 1e6, 'r--', linewidth=2, label='PV Liabilities')
    ax.set_xlabel('Year')
    ax.set_ylabel('Value ($M)')
    ax.set_title('Money Market Strategy: Assets vs Liabilities')
    ax.legend()
    ax.set_xlim(0, horizon)

    # Panel 3: Assets vs Liabilities for DM strategy
    ax = axes[1, 0]
    ax.plot(years, wealth_paths_dm[sample_idx, :] / 1e6, 'g-',
            linewidth=2, label='Assets (Duration Matched)')
    ax.plot(years, liab_pv / 1e6, 'r--', linewidth=2, label='PV Liabilities')
    ax.set_xlabel('Year')
    ax.set_ylabel('Value ($M)')
    ax.set_title('Duration Matching Strategy: Assets vs Liabilities')
    ax.legend()
    ax.set_xlim(0, horizon)

    # Panel 4: Funded ratios comparison
    ax = axes[1, 1]
    # Compute funded ratios (avoiding division by zero)
    funded_mm = np.where(liab_pv > 0, wealth_paths_mm[sample_idx, :] / liab_pv, 1.0)
    funded_dm = np.where(liab_pv > 0, wealth_paths_dm[sample_idx, :] / liab_pv, 1.0)

    ax.plot(years, funded_mm, 'b-', linewidth=2, label='MM Strategy')
    ax.plot(years, funded_dm, 'g-', linewidth=2, label='Duration Matched')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Fully Funded')
    ax.set_xlabel('Year')
    ax.set_ylabel('Funded Ratio')
    ax.set_title('Funded Status Comparison')
    ax.legend()
    ax.set_xlim(0, horizon)
    ax.set_ylim(0, max(2.5, max(funded_mm.max(), funded_dm.max()) * 1.1))

    plt.tight_layout()
    return fig


def plot_strategy_comparison_bars(
    results: Dict[str, SimulationResult],
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Bar chart comparing key metrics across strategies.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    strategies = list(results.keys())
    x = np.arange(len(strategies))
    width = 0.6

    # Panel 1: Default Rates
    ax = axes[0]
    default_rates = [100 * results[s].defaulted.mean() for s in strategies]
    bars = ax.bar(x, default_rates, width, color=COLORS[:len(strategies)])
    ax.set_ylabel('Default Rate (%)')
    ax.set_title('Default Probability')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(' + ', '\n') for s in strategies], fontsize=9)
    ax.set_ylim(0, max(default_rates) * 1.2 if max(default_rates) > 0 else 10)

    # Add value labels
    for bar, val in zip(bars, default_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Panel 2: Average Total Consumption
    ax = axes[1]
    avg_consumption = [results[s].total_consumption.mean() / 1e6 for s in strategies]
    bars = ax.bar(x, avg_consumption, width, color=COLORS[:len(strategies)])
    ax.set_ylabel('Avg Total Consumption ($M)')
    ax.set_title('Lifetime Consumption')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(' + ', '\n') for s in strategies], fontsize=9)

    for bar, val in zip(bars, avg_consumption):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'${val:.2f}M', ha='center', va='bottom', fontsize=10)

    # Panel 3: Consumption Volatility
    ax = axes[2]
    cons_vol = [results[s].consumption_paths.std(axis=1).mean() / 1000 for s in strategies]
    bars = ax.bar(x, cons_vol, width, color=COLORS[:len(strategies)])
    ax.set_ylabel('Consumption Volatility ($k/year)')
    ax.set_title('Consumption Uncertainty')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(' + ', '\n') for s in strategies], fontsize=9)

    for bar, val in zip(bars, cons_vol):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'${val:.1f}k', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig


def plot_wealth_paths_spaghetti(
    results: Dict[str, SimulationResult],
    n_paths: int = 100,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Spaghetti plot showing sample wealth trajectories for each strategy.
    """
    strategies = list(results.keys())
    n_strategies = len(strategies)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        result = results[strategy]
        n_periods = result.wealth_paths.shape[1]
        years = np.arange(n_periods)

        # Plot sample paths
        for i in range(min(n_paths, result.wealth_paths.shape[0])):
            alpha = 0.3 if result.defaulted[i] else 0.15
            color = 'red' if result.defaulted[i] else COLORS[idx]
            ax.plot(years, result.wealth_paths[i, :] / 1e6,
                   color=color, alpha=alpha, linewidth=0.5)

        # Plot median path
        median_path = np.median(result.wealth_paths, axis=0)
        ax.plot(years, median_path / 1e6, 'k-', linewidth=2, label='Median')

        # Plot percentile bands
        p10 = np.percentile(result.wealth_paths, 10, axis=0)
        p90 = np.percentile(result.wealth_paths, 90, axis=0)
        ax.fill_between(years, p10 / 1e6, p90 / 1e6, alpha=0.2, color=COLORS[idx])

        ax.set_xlabel('Year')
        ax.set_ylabel('Wealth ($M)')
        ax.set_title(f'{strategy}')
        ax.set_xlim(0, n_periods - 1)
        ax.set_ylim(0, None)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        # Add default rate annotation
        default_rate = 100 * result.defaulted.mean()
        ax.annotate(f'Default: {default_rate:.1f}%',
                   xy=(0.02, 0.98), xycoords='axes fraction',
                   fontsize=10, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def plot_consumption_paths(
    results: Dict[str, SimulationResult],
    target_consumption: float,
    n_paths: int = 50,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot consumption paths for each strategy.
    """
    strategies = list(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        result = results[strategy]
        n_periods = result.consumption_paths.shape[1]
        years = np.arange(1, n_periods + 1)

        # Plot sample paths
        for i in range(min(n_paths, result.consumption_paths.shape[0])):
            ax.plot(years, result.consumption_paths[i, :] / 1000,
                   color=COLORS[idx], alpha=0.1, linewidth=0.5)

        # Plot median consumption
        median_cons = np.median(result.consumption_paths, axis=0)
        ax.plot(years, median_cons / 1000, 'k-', linewidth=2, label='Median')

        # Target line
        ax.axhline(y=target_consumption / 1000, color='gray',
                  linestyle='--', alpha=0.7, label='Target')

        ax.set_xlabel('Year')
        ax.set_ylabel('Consumption ($k)')
        ax.set_title(f'{strategy}')
        ax.set_xlim(1, n_periods)
        ax.legend(loc='upper right')

    plt.tight_layout()
    return fig


def plot_final_wealth_distribution(
    results: Dict[str, SimulationResult],
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Histogram of final wealth distribution across strategies.
    """
    fig, ax = plt.subplots(figsize=figsize)

    strategies = list(results.keys())

    for idx, strategy in enumerate(strategies):
        result = results[strategy]
        # Filter out zeros for better visualization
        final_wealth = result.final_wealth[result.final_wealth > 0] / 1e6

        ax.hist(final_wealth, bins=50, alpha=0.5, label=strategy,
               color=COLORS[idx], density=True)

    ax.set_xlabel('Final Wealth ($M)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Final Wealth (Non-Defaulted Paths)')
    ax.legend()

    plt.tight_layout()
    return fig


def plot_interest_rate_scenarios(
    rates: np.ndarray,
    n_paths: int = 100,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Visualize interest rate scenarios from the simulation.
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_periods = rates.shape[1]
    years = np.arange(n_periods)

    # Plot sample paths
    for i in range(min(n_paths, rates.shape[0])):
        ax.plot(years, rates[i, :] * 100, 'b-', alpha=0.1, linewidth=0.5)

    # Plot median and percentiles
    median_rate = np.median(rates, axis=0)
    p10 = np.percentile(rates, 10, axis=0)
    p90 = np.percentile(rates, 90, axis=0)

    ax.plot(years, median_rate * 100, 'navy', linewidth=2, label='Median')
    ax.fill_between(years, p10 * 100, p90 * 100, alpha=0.3, color='blue',
                   label='10th-90th percentile')

    ax.axhline(y=3.0, color='red', linestyle='--', alpha=0.7,
              label='Long-run mean (3%)')

    ax.set_xlabel('Year')
    ax.set_ylabel('Interest Rate (%)')
    ax.set_title('Simulated Interest Rate Paths')
    ax.legend()
    ax.set_xlim(0, n_periods - 1)

    plt.tight_layout()
    return fig


def plot_rate_sensitivity_analysis(
    starting_rates: List[float] = [0.01, 0.03, 0.05],
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Show how strategy performance varies with initial interest rate level.

    This helps illustrate the asymmetric risk when rates are near the floor.
    """
    from retirement_simulation import run_monte_carlo, STRATEGIES, SimulationParams

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Reduce simulations for faster computation
    sim_params = SimulationParams(n_simulations=2000)

    all_results = {}
    for r0 in starting_rates:
        results, _, _ = run_monte_carlo(
            sim_params=sim_params,
            initial_rate=r0
        )
        all_results[r0] = results

    strategies = [str(s) for s in STRATEGIES]
    x = np.arange(len(strategies))
    width = 0.25

    # Panel 1: Default Rates by Initial Rate
    ax = axes[0]
    for i, r0 in enumerate(starting_rates):
        default_rates = [100 * all_results[r0][s].defaulted.mean() for s in strategies]
        ax.bar(x + i * width, default_rates, width,
              label=f'r0 = {r0*100:.0f}%', color=COLORS[i])

    ax.set_ylabel('Default Rate (%)')
    ax.set_title('Default Rate by Initial Rate')
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.replace(' + ', '\n') for s in strategies], fontsize=8)
    ax.legend()

    # Panel 2: Average Consumption by Initial Rate
    ax = axes[1]
    for i, r0 in enumerate(starting_rates):
        avg_cons = [all_results[r0][s].total_consumption.mean() / 1e6 for s in strategies]
        ax.bar(x + i * width, avg_cons, width,
              label=f'r0 = {r0*100:.0f}%', color=COLORS[i])

    ax.set_ylabel('Avg Total Consumption ($M)')
    ax.set_title('Lifetime Consumption by Initial Rate')
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.replace(' + ', '\n') for s in strategies], fontsize=8)
    ax.legend()

    # Panel 3: Consumption Volatility by Initial Rate
    ax = axes[2]
    for i, r0 in enumerate(starting_rates):
        vol = [all_results[r0][s].consumption_paths.std(axis=1).mean() / 1000
               for s in strategies]
        ax.bar(x + i * width, vol, width,
              label=f'r0 = {r0*100:.0f}%', color=COLORS[i])

    ax.set_ylabel('Consumption Volatility ($k)')
    ax.set_title('Consumption Uncertainty by Initial Rate')
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.replace(' + ', '\n') for s in strategies], fontsize=8)
    ax.legend()

    plt.tight_layout()
    return fig


def create_summary_table(results: Dict[str, SimulationResult]) -> str:
    """
    Create a formatted summary table for display.
    """
    header = f"{'Strategy':<25} {'Default %':>10} {'Avg Cons ($M)':>15} {'Cons Vol ($k)':>15} {'Avg Final ($M)':>15}"
    separator = "-" * len(header)

    rows = [header, separator]

    for name, result in results.items():
        default_pct = 100 * result.defaulted.mean()
        avg_cons = result.total_consumption.mean() / 1e6
        cons_vol = result.consumption_paths.std(axis=1).mean() / 1000
        avg_final = result.final_wealth.mean() / 1e6

        row = f"{name:<25} {default_pct:>10.1f} {avg_cons:>15.2f} {cons_vol:>15.1f} {avg_final:>15.2f}"
        rows.append(row)

    return "\n".join(rows)
