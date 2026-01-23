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
    MedianPathResult,
    liability_pv_vectorized,
    liability_duration,
    EconomicParams,
    SimulationParams,
    RandomWalkParams,
    run_median_path_simulation,
    STRATEGIES,
    Strategy,
    BondParams,
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


# =============================================================================
# Random Walk and Median Path Visualizations
# =============================================================================

def plot_wealth_allocation_lifecycle(
    median_result: MedianPathResult,
    figsize: Tuple[int, int] = (14, 12)
) -> plt.Figure:
    """
    Visualize how a consumer's financial wealth allocation changes over their lifecycle
    when median returns are realized each period.

    This creates a comprehensive 4-panel visualization showing:
    1. Interest rate path over time
    2. Portfolio allocation (stocks, money market, long bonds) as stacked area
    3. Wealth trajectory and consumption
    4. Funded ratio over time

    Args:
        median_result: Result from run_median_path_simulation
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    years = median_result.years

    # Panel 1: Interest Rate Path
    ax = axes[0, 0]
    ax.plot(years, median_result.rates * 100, 'b-', linewidth=2, marker='o', markersize=3)
    ax.axhline(y=3.0, color='gray', linestyle='--', alpha=0.7, label='Long-run mean (3%)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Interest Rate (%)')
    ax.set_title('Interest Rate Path (Median Scenario)')
    ax.legend()
    ax.set_xlim(0, len(years) - 1)
    ax.grid(True, alpha=0.3)

    # Panel 2: Portfolio Allocation (Stacked Area)
    ax = axes[0, 1]
    n_periods = len(median_result.stock_weight)
    alloc_years = np.arange(n_periods)

    # Create stacked area chart
    ax.fill_between(alloc_years, 0, median_result.stock_weight * 100,
                    alpha=0.8, label='Stocks', color='#2ca02c')
    ax.fill_between(alloc_years, median_result.stock_weight * 100,
                    (median_result.stock_weight + median_result.mm_weight) * 100,
                    alpha=0.8, label='Money Market', color='#1f77b4')
    ax.fill_between(alloc_years, (median_result.stock_weight + median_result.mm_weight) * 100,
                    (median_result.stock_weight + median_result.mm_weight + median_result.lb_weight) * 100,
                    alpha=0.8, label='Long Bonds', color='#ff7f0e')

    ax.set_xlabel('Year')
    ax.set_ylabel('Allocation (%)')
    ax.set_title(f'Portfolio Allocation Over Lifecycle\n({median_result.strategy_name})')
    ax.legend(loc='upper right')
    ax.set_xlim(0, n_periods - 1)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Panel 3: Wealth and Consumption
    ax = axes[1, 0]
    ax.plot(years, median_result.wealth / 1e6, 'b-', linewidth=2,
            label='Wealth', marker='o', markersize=3)

    # Add consumption as bars on secondary axis
    ax2 = ax.twinx()
    ax2.bar(alloc_years + 0.5, median_result.consumption / 1000, alpha=0.4,
            color='green', label='Consumption', width=0.8)

    ax.set_xlabel('Year')
    ax.set_ylabel('Wealth ($M)', color='blue')
    ax2.set_ylabel('Annual Consumption ($k)', color='green')
    ax.set_title('Wealth Trajectory and Consumption')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='green')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax.set_xlim(0, len(years) - 1)
    ax.grid(True, alpha=0.3)

    # Panel 4: Funded Ratio
    ax = axes[1, 1]
    # Cap funded ratio for display purposes
    display_funded = np.minimum(median_result.funded_ratio, 5.0)
    ax.plot(years, display_funded, 'purple', linewidth=2, marker='o', markersize=3)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Fully Funded (100%)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Funded Ratio')
    ax.set_title('Funded Status (Assets / PV Liabilities)')
    ax.legend()
    ax.set_xlim(0, len(years) - 1)
    ax.set_ylim(0, min(5.0, display_funded.max() * 1.1))
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Lifecycle Wealth Allocation Analysis\nStrategy: {median_result.strategy_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_allocation_comparison_median_path(
    strategies: List[Strategy] = None,
    r0: float = 0.03,
    sim_params: SimulationParams = None,
    bond_params: BondParams = None,
    econ_params: EconomicParams = None,
    use_random_walk: bool = False,
    rw_params: RandomWalkParams = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Compare wealth allocation across different strategies under median return scenario.

    Args:
        strategies: List of strategies to compare (defaults to all 4 strategies)
        r0: Initial interest rate
        sim_params: Simulation parameters
        bond_params: Bond parameters
        econ_params: Economic parameters
        use_random_walk: Whether to use random walk interest rate model
        rw_params: Random walk parameters
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    if strategies is None:
        strategies = STRATEGIES
    if sim_params is None:
        sim_params = SimulationParams()
    if bond_params is None:
        bond_params = BondParams()
    if econ_params is None:
        econ_params = EconomicParams()
    if rw_params is None:
        rw_params = RandomWalkParams()

    n_strategies = len(strategies)
    fig, axes = plt.subplots(n_strategies, 2, figsize=figsize)

    model_name = "Random Walk" if use_random_walk else "Mean-Reverting"

    for idx, strategy in enumerate(strategies):
        # Run median path simulation
        result = run_median_path_simulation(
            strategy=strategy,
            r0=r0,
            sim_params=sim_params,
            bond_params=bond_params,
            econ_params=econ_params,
            rw_params=rw_params,
            use_random_walk=use_random_walk
        )

        n_periods = len(result.stock_weight)
        alloc_years = np.arange(n_periods)

        # Left panel: Portfolio allocation
        ax = axes[idx, 0] if n_strategies > 1 else axes[0]
        ax.fill_between(alloc_years, 0, result.stock_weight * 100,
                        alpha=0.8, label='Stocks', color='#2ca02c')
        ax.fill_between(alloc_years, result.stock_weight * 100,
                        (result.stock_weight + result.mm_weight) * 100,
                        alpha=0.8, label='Money Market', color='#1f77b4')
        ax.fill_between(alloc_years, (result.stock_weight + result.mm_weight) * 100,
                        (result.stock_weight + result.mm_weight + result.lb_weight) * 100,
                        alpha=0.8, label='Long Bonds', color='#ff7f0e')

        ax.set_ylabel('Allocation (%)')
        ax.set_title(f'{result.strategy_name}')
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0, n_periods - 1)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        if idx == n_strategies - 1:
            ax.set_xlabel('Year')

        # Right panel: Wealth trajectory
        ax = axes[idx, 1] if n_strategies > 1 else axes[1]
        ax.plot(result.years, result.wealth / 1e6, 'b-', linewidth=2)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel('Wealth ($M)')
        ax.set_title(f'{result.strategy_name}')
        ax.set_xlim(0, len(result.years) - 1)
        ax.grid(True, alpha=0.3)
        if idx == n_strategies - 1:
            ax.set_xlabel('Year')

        # Add final wealth annotation
        final_w = result.wealth[-1]
        ax.annotate(f'Final: ${final_w/1e6:.2f}M',
                   xy=(0.98, 0.95), xycoords='axes fraction',
                   fontsize=9, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(f'Lifecycle Wealth Allocation Comparison\n{model_name} Interest Rates (r0 = {r0*100:.1f}%)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_random_walk_vs_mean_reverting(
    r0: float = 0.03,
    sim_params: SimulationParams = None,
    bond_params: BondParams = None,
    econ_params: EconomicParams = None,
    rw_params: RandomWalkParams = None,
    strategy: Strategy = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Compare median paths under random walk vs mean-reverting interest rate models.

    This visualization helps illustrate how the interest rate model assumption
    affects wealth allocation and outcomes.

    Args:
        r0: Initial interest rate
        sim_params: Simulation parameters
        bond_params: Bond parameters
        econ_params: Economic parameters
        rw_params: Random walk parameters
        strategy: Strategy to compare (defaults to Duration Match + Variable)
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    if sim_params is None:
        sim_params = SimulationParams()
    if bond_params is None:
        bond_params = BondParams()
    if econ_params is None:
        econ_params = EconomicParams()
    if rw_params is None:
        rw_params = RandomWalkParams()
    if strategy is None:
        strategy = STRATEGIES[3]  # DurMatch + Variable

    # Run both models
    result_mr = run_median_path_simulation(
        strategy=strategy, r0=r0, sim_params=sim_params,
        bond_params=bond_params, econ_params=econ_params,
        use_random_walk=False
    )

    result_rw = run_median_path_simulation(
        strategy=strategy, r0=r0, sim_params=sim_params,
        bond_params=bond_params, econ_params=econ_params,
        rw_params=rw_params, use_random_walk=True
    )

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: Interest Rate Paths
    ax = axes[0, 0]
    ax.plot(result_mr.years, result_mr.rates * 100, 'b-', linewidth=2,
            label='Mean-Reverting', marker='o', markersize=3)
    ax.plot(result_rw.years, result_rw.rates * 100, 'r-', linewidth=2,
            label='Random Walk', marker='s', markersize=3)
    ax.axhline(y=econ_params.r_bar * 100, color='gray', linestyle='--', alpha=0.7,
               label=f'Long-run mean ({econ_params.r_bar*100:.0f}%)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Interest Rate (%)')
    ax.set_title('Interest Rate Paths (Median Scenario)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Wealth Paths
    ax = axes[0, 1]
    ax.plot(result_mr.years, result_mr.wealth / 1e6, 'b-', linewidth=2,
            label='Mean-Reverting', marker='o', markersize=3)
    ax.plot(result_rw.years, result_rw.wealth / 1e6, 'r-', linewidth=2,
            label='Random Walk', marker='s', markersize=3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Wealth ($M)')
    ax.set_title('Wealth Trajectory Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Long Bond Allocation
    ax = axes[1, 0]
    n_periods = len(result_mr.lb_weight)
    alloc_years = np.arange(n_periods)
    ax.plot(alloc_years, result_mr.lb_weight * 100, 'b-', linewidth=2,
            label='Mean-Reverting', marker='o', markersize=3)
    ax.plot(alloc_years, result_rw.lb_weight * 100, 'r-', linewidth=2,
            label='Random Walk', marker='s', markersize=3)
    ax.set_xlabel('Year')
    ax.set_ylabel('Long Bond Allocation (%)')
    ax.set_title('Long Bond Weight Over Lifecycle')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(result_mr.lb_weight.max(), result_rw.lb_weight.max()) * 100 * 1.1)

    # Panel 4: Funded Ratio
    ax = axes[1, 1]
    display_mr = np.minimum(result_mr.funded_ratio, 5.0)
    display_rw = np.minimum(result_rw.funded_ratio, 5.0)
    ax.plot(result_mr.years, display_mr, 'b-', linewidth=2,
            label='Mean-Reverting', marker='o', markersize=3)
    ax.plot(result_rw.years, display_rw, 'r-', linewidth=2,
            label='Random Walk', marker='s', markersize=3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Fully Funded')
    ax.set_xlabel('Year')
    ax.set_ylabel('Funded Ratio')
    ax.set_title('Funded Status Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Random Walk vs Mean-Reverting Interest Rates\nStrategy: {strategy} | r0 = {r0*100:.1f}%',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_interest_rate_models_comparison(
    r0: float = 0.03,
    n_periods: int = 30,
    n_paths: int = 100,
    econ_params: EconomicParams = None,
    rw_params: RandomWalkParams = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Compare simulated interest rate paths under random walk vs mean-reverting models.

    Shows sample paths and distribution characteristics of both models.

    Args:
        r0: Initial interest rate
        n_periods: Number of periods
        n_paths: Number of sample paths to show
        econ_params: Economic parameters
        rw_params: Random walk parameters
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    from retirement_simulation import (
        simulate_interest_rates, simulate_interest_rates_random_walk,
        generate_correlated_shocks
    )

    if econ_params is None:
        econ_params = EconomicParams()
    if rw_params is None:
        rw_params = RandomWalkParams()

    rng = np.random.default_rng(42)
    n_sims = 1000

    # Generate shocks
    rate_shocks, _ = generate_correlated_shocks(n_periods, n_sims, econ_params.rho, rng)

    # Simulate both models
    rates_mr = simulate_interest_rates(
        r0, n_periods, n_sims, econ_params, rate_shocks
    )
    rates_rw = simulate_interest_rates_random_walk(
        r0, n_periods, n_sims, rw_params.sigma_r, rw_params.drift,
        rate_shocks, rw_params.r_floor
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    years = np.arange(n_periods + 1)

    # Panel 1: Mean-Reverting Model
    ax = axes[0]
    for i in range(min(n_paths, n_sims)):
        ax.plot(years, rates_mr[i, :] * 100, 'b-', alpha=0.1, linewidth=0.5)

    median_mr = np.median(rates_mr, axis=0)
    p10_mr = np.percentile(rates_mr, 10, axis=0)
    p90_mr = np.percentile(rates_mr, 90, axis=0)

    ax.plot(years, median_mr * 100, 'navy', linewidth=2, label='Median')
    ax.fill_between(years, p10_mr * 100, p90_mr * 100, alpha=0.3, color='blue',
                   label='10th-90th percentile')
    ax.axhline(y=econ_params.r_bar * 100, color='red', linestyle='--', alpha=0.7,
              label=f'Long-run mean ({econ_params.r_bar*100:.0f}%)')

    ax.set_xlabel('Year')
    ax.set_ylabel('Interest Rate (%)')
    ax.set_title(f'Mean-Reverting Model (φ = {econ_params.phi})')
    ax.legend(loc='upper right')
    ax.set_xlim(0, n_periods)
    ax.grid(True, alpha=0.3)

    # Panel 2: Random Walk Model
    ax = axes[1]
    for i in range(min(n_paths, n_sims)):
        ax.plot(years, rates_rw[i, :] * 100, 'r-', alpha=0.1, linewidth=0.5)

    median_rw = np.median(rates_rw, axis=0)
    p10_rw = np.percentile(rates_rw, 10, axis=0)
    p90_rw = np.percentile(rates_rw, 90, axis=0)

    ax.plot(years, median_rw * 100, 'darkred', linewidth=2, label='Median')
    ax.fill_between(years, p10_rw * 100, p90_rw * 100, alpha=0.3, color='red',
                   label='10th-90th percentile')
    ax.axhline(y=r0 * 100, color='gray', linestyle='--', alpha=0.7,
              label=f'Initial rate ({r0*100:.0f}%)')

    ax.set_xlabel('Year')
    ax.set_ylabel('Interest Rate (%)')
    ax.set_title('Random Walk Model (No Mean Reversion)')
    ax.legend(loc='upper right')
    ax.set_xlim(0, n_periods)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Interest Rate Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
