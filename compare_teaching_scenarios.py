#!/usr/bin/env python3
"""
Teaching Scenarios: LDI vs Rule-of-Thumb Strategy Comparison.

Creates a multi-page PDF comparing LDI vs Rule-of-Thumb strategies across three
Monte Carlo scenarios designed for teaching lifecycle investment concepts:

1. Baseline - Normal Monte Carlo with random shocks
2. Sequence Risk - Bad stock returns in first 5 years OF retirement
3. Rate Shock - Sudden interest rate drop 5 years BEFORE retirement

Each scenario page shows:
- Financial wealth fan charts (overlaid)
- Default risk (bar chart + timing histogram)
- Terminal wealth distribution
- PV consumption distribution
- Cumulative stock returns fan chart
- Interest rate paths fan chart

Plus a summary comparison figure showing default rates, median PV consumption,
and median terminal wealth across all three scenarios.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import dataclass
from typing import Dict, List, Tuple

from core import (
    LifecycleParams,
    EconomicParams,
    simulate_with_strategy,
    generate_correlated_shocks,
    compute_pv_consumption,
    LDIStrategy,
    RuleOfThumbStrategy,
)
from visualization import plot_fan_chart


# =============================================================================
# Scenario Definitions
# =============================================================================

@dataclass
class ScenarioConfig:
    """Configuration for a teaching scenario."""
    name: str
    title: str
    description: str


SCENARIOS = {
    'baseline': ScenarioConfig(
        name='baseline',
        title='Baseline: Normal Monte Carlo',
        description='Standard random shocks - no scenario manipulation'
    ),
    'sequence_risk': ScenarioConfig(
        name='sequence_risk',
        title='Sequence-of-Returns Risk',
        description='Bad stock returns (-15%/yr) in first 5 years of retirement'
    ),
    'rate_shock': ScenarioConfig(
        name='rate_shock',
        title='Pre-Retirement Rate Shock',
        description='Interest rate drop (~6% cumulative) in 5 years before retirement'
    ),
}


# =============================================================================
# Shock Injection Logic
# =============================================================================

def create_scenario_shocks(
    base_rate_shocks: np.ndarray,
    base_stock_shocks: np.ndarray,
    scenario: str,
    working_years: int,
    total_years: int,
    econ_params: EconomicParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Modify shocks for teaching scenarios.

    Args:
        base_rate_shocks: Baseline rate shocks (n_sims, n_periods)
        base_stock_shocks: Baseline stock shocks (n_sims, n_periods)
        scenario: One of 'baseline', 'sequence_risk', 'rate_shock'
        working_years: Number of working years
        total_years: Total simulation years
        econ_params: Economic parameters

    Returns:
        Tuple of (rate_shocks, stock_shocks) with scenario modifications applied
    """
    rate_shocks = base_rate_shocks.copy()
    stock_shocks = base_stock_shocks.copy()

    if scenario == 'baseline':
        pass  # No modification

    elif scenario == 'sequence_risk':
        # Sequence-of-returns risk: bad returns in FIRST 5 years of retirement
        # This demonstrates the classic problem of withdrawing from a declining portfolio
        shock_start = working_years
        shock_end = min(working_years + 5, total_years)
        # Force ~-15% annual returns: with r_bar=2%, mu=4%, sigma=18%
        # Return = r + mu + sigma*eps => -15% = 2% + 4% + 18%*eps => eps ~ -1.2
        stock_shocks[:, shock_start:shock_end] = -1.2

    elif scenario == 'rate_shock':
        # Rate shock: negative rate shock 5 years BEFORE retirement
        # This affects bond valuations and discount rates
        shock_start = max(0, working_years - 5)
        shock_end = working_years
        # Force rate decline: eps ~ -2 sigma per year (cumulative ~6% rate drop)
        rate_shocks[:, shock_start:shock_end] = -2.0

    return rate_shocks, stock_shocks


# =============================================================================
# Scenario Simulation
# =============================================================================

@dataclass
class ScenarioResult:
    """Results from running a scenario with both strategies."""
    scenario: str
    ages: np.ndarray

    # LDI results
    ldi_financial_wealth_paths: np.ndarray
    ldi_consumption_paths: np.ndarray
    ldi_default_flags: np.ndarray
    ldi_default_ages: np.ndarray
    ldi_pv_consumption: np.ndarray

    # RoT results
    rot_financial_wealth_paths: np.ndarray
    rot_consumption_paths: np.ndarray
    rot_default_flags: np.ndarray
    rot_default_ages: np.ndarray
    rot_pv_consumption: np.ndarray

    # Market conditions
    rate_paths: np.ndarray
    stock_return_paths: np.ndarray
    cumulative_stock_returns: np.ndarray


def run_teaching_scenario(
    scenario: str,
    params: LifecycleParams,
    econ_params: EconomicParams,
    base_rate_shocks: np.ndarray,
    base_stock_shocks: np.ndarray,
) -> ScenarioResult:
    """
    Run both LDI and RoT strategies under a specific scenario.

    Args:
        scenario: Scenario name ('baseline', 'sequence_risk', 'rate_shock')
        params: Lifecycle parameters
        econ_params: Economic parameters
        base_rate_shocks: Baseline rate shocks (shared across scenarios for fair comparison)
        base_stock_shocks: Baseline stock shocks

    Returns:
        ScenarioResult with paths and statistics for both strategies
    """
    total_years = params.end_age - params.start_age
    working_years = params.retirement_age - params.start_age
    n_sims = base_rate_shocks.shape[0]
    ages = np.arange(params.start_age, params.end_age)

    # Apply scenario-specific shock modifications
    rate_shocks, stock_shocks = create_scenario_shocks(
        base_rate_shocks, base_stock_shocks, scenario,
        working_years, total_years, econ_params
    )

    # Create strategy instances
    ldi_strategy = LDIStrategy(allow_leverage=False)
    rot_strategy = RuleOfThumbStrategy(savings_rate=0.15, withdrawal_rate=0.04)

    # Run LDI strategy
    ldi_result = simulate_with_strategy(
        ldi_strategy, params, econ_params, rate_shocks, stock_shocks
    )

    # Run RoT strategy
    rot_result = simulate_with_strategy(
        rot_strategy, params, econ_params, rate_shocks, stock_shocks
    )

    # Compute PV consumption for each simulation
    r = econ_params.r_bar
    ldi_pv_consumption = np.array([
        compute_pv_consumption(ldi_result.consumption[sim], r)
        for sim in range(n_sims)
    ])
    rot_pv_consumption = np.array([
        compute_pv_consumption(rot_result.consumption[sim], r)
        for sim in range(n_sims)
    ])

    # Compute cumulative stock returns
    cumulative_stock_returns = np.cumprod(1 + ldi_result.stock_returns, axis=1)

    return ScenarioResult(
        scenario=scenario,
        ages=ages,
        # LDI
        ldi_financial_wealth_paths=ldi_result.financial_wealth,
        ldi_consumption_paths=ldi_result.consumption,
        ldi_default_flags=ldi_result.defaulted,
        ldi_default_ages=ldi_result.default_age,
        ldi_pv_consumption=ldi_pv_consumption,
        # RoT
        rot_financial_wealth_paths=rot_result.financial_wealth,
        rot_consumption_paths=rot_result.consumption,
        rot_default_flags=rot_result.defaulted,
        rot_default_ages=rot_result.default_age,
        rot_pv_consumption=rot_pv_consumption,
        # Market
        rate_paths=ldi_result.interest_rates,
        stock_return_paths=ldi_result.stock_returns,
        cumulative_stock_returns=cumulative_stock_returns,
    )


def run_all_teaching_scenarios(
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    n_simulations: int = 500,
    random_seed: int = 42,
) -> Dict[str, ScenarioResult]:
    """
    Run all three teaching scenarios.

    Uses the same baseline shocks for all scenarios to ensure fair comparison.
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    total_years = params.end_age - params.start_age
    rng = np.random.default_rng(random_seed)

    # Generate baseline shocks (shared across all scenarios)
    base_rate_shocks, base_stock_shocks = generate_correlated_shocks(
        total_years, n_simulations, econ_params.rho, rng
    )

    results = {}
    for scenario in ['baseline', 'sequence_risk', 'rate_shock']:
        print(f"  Running {scenario} scenario...")
        results[scenario] = run_teaching_scenario(
            scenario, params, econ_params, base_rate_shocks, base_stock_shocks
        )

    return results


# =============================================================================
# Visualization
# =============================================================================

# Colors for strategies
COLOR_LDI = '#2ecc71'   # Green
COLOR_ROT = '#e74c3c'   # Red
COLOR_RATES = '#3498db'  # Blue
COLOR_STOCKS = '#9b59b6'  # Purple


def create_scenario_figure(
    result: ScenarioResult,
    config: ScenarioConfig,
    params: LifecycleParams,
    figsize: Tuple[int, int] = (16, 10),
) -> plt.Figure:
    """
    Create a 2x3 panel figure for a single scenario.

    Panels:
    - (0,0): Financial Wealth fan charts (LDI vs RoT overlaid)
    - (0,1): Default Risk (bar chart + timing histogram)
    - (0,2): Terminal Wealth distribution
    - (1,0): PV Consumption distribution
    - (1,1): Cumulative Stock Returns fan chart
    - (1,2): Interest Rate Paths fan chart
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'{config.title}\n{config.description}', fontsize=14, fontweight='bold')

    x = np.arange(len(result.ages))
    retirement_x = params.retirement_age - params.start_age
    n_sims = result.ldi_financial_wealth_paths.shape[0]

    # ---- (0,0): Financial Wealth Fan Charts ----
    ax = axes[0, 0]
    plot_fan_chart(ax, result.ldi_financial_wealth_paths, x, color=COLOR_LDI, label_prefix='LDI')
    plot_fan_chart(ax, result.rot_financial_wealth_paths, x, color=COLOR_ROT, label_prefix='RoT')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Years from Career Start')
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_yscale('symlog', linthresh=50, linscale=0.5)
    ax.grid(True, alpha=0.3)

    # ---- (0,1): Default Risk ----
    ax = axes[0, 1]
    ldi_default_rate = np.mean(result.ldi_default_flags) * 100
    rot_default_rate = np.mean(result.rot_default_flags) * 100

    # Bar chart for default rates
    bars = ax.bar(['LDI', 'RoT'], [ldi_default_rate, rot_default_rate],
                  color=[COLOR_LDI, COLOR_ROT], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Default Rate (%)')
    ax.set_title(f'Default Risk\nLDI: {ldi_default_rate:.1f}% | RoT: {rot_default_rate:.1f}%')

    # Add value labels on bars
    for bar, rate in zip(bars, [ldi_default_rate, rot_default_rate]):
        height = bar.get_height()
        ax.annotate(f'{rate:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Inset histogram for default timing (if there are defaults)
    if np.any(result.ldi_default_flags) or np.any(result.rot_default_flags):
        ax_inset = ax.inset_axes([0.55, 0.5, 0.4, 0.4])
        ldi_ages = result.ldi_default_ages[result.ldi_default_flags]
        rot_ages = result.rot_default_ages[result.rot_default_flags]
        bins = np.arange(params.retirement_age, params.end_age + 1, 2)
        if len(ldi_ages) > 0:
            ax_inset.hist(ldi_ages, bins=bins, alpha=0.6, color=COLOR_LDI, label='LDI')
        if len(rot_ages) > 0:
            ax_inset.hist(rot_ages, bins=bins, alpha=0.6, color=COLOR_ROT, label='RoT')
        ax_inset.set_xlabel('Age', fontsize=8)
        ax_inset.set_ylabel('Count', fontsize=8)
        ax_inset.set_title('Default Timing', fontsize=9)
        ax_inset.tick_params(labelsize=7)
        ax_inset.legend(fontsize=7)

    # ---- (0,2): Terminal Wealth Distribution ----
    ax = axes[0, 2]
    ldi_final = result.ldi_financial_wealth_paths[:, -1]
    rot_final = result.rot_financial_wealth_paths[:, -1]

    floor_val = 10
    ldi_floored = np.maximum(ldi_final, floor_val)
    rot_floored = np.maximum(rot_final, floor_val)
    max_val = max(np.percentile(ldi_floored, 99), np.percentile(rot_floored, 99))
    if max_val > floor_val:
        bins = np.geomspace(floor_val, max_val, 30)
        ax.hist(ldi_floored, bins=bins, alpha=0.6, color=COLOR_LDI,
                label=f'LDI (med=${np.median(ldi_final):,.0f}k)', edgecolor='white')
        ax.hist(rot_floored, bins=bins, alpha=0.6, color=COLOR_ROT,
                label=f'RoT (med=${np.median(rot_final):,.0f}k)', edgecolor='white')
        ax.set_xscale('log')
    else:
        bins = 30
        ax.hist(ldi_final, bins=bins, alpha=0.6, color=COLOR_LDI,
                label=f'LDI (med=${np.median(ldi_final):,.0f}k)', edgecolor='white')
        ax.hist(rot_final, bins=bins, alpha=0.6, color=COLOR_ROT,
                label=f'RoT (med=${np.median(rot_final):,.0f}k)', edgecolor='white')
    ax.axvline(x=max(np.median(ldi_final), floor_val), color=COLOR_LDI, linestyle='--', linewidth=2)
    ax.axvline(x=max(np.median(rot_final), floor_val), color=COLOR_ROT, linestyle='--', linewidth=2)
    ax.set_xlabel('Terminal Wealth at Age 95 ($ 000s)')
    ax.set_ylabel('Count')
    ax.set_title('Terminal Wealth Distribution')
    ax.legend(loc='upper right', fontsize=9)

    # ---- (1,0): PV Consumption Distribution ----
    ax = axes[1, 0]
    ldi_pv = result.ldi_pv_consumption
    rot_pv = result.rot_pv_consumption

    floor_val = 100
    ldi_pv_floored = np.maximum(ldi_pv, floor_val)
    rot_pv_floored = np.maximum(rot_pv, floor_val)
    min_val = min(ldi_pv_floored.min(), rot_pv_floored.min())
    max_val = max(np.percentile(ldi_pv_floored, 99), np.percentile(rot_pv_floored, 99))
    bins = np.geomspace(min_val, max_val, 30)
    ax.hist(ldi_pv_floored, bins=bins, alpha=0.6, color=COLOR_LDI,
            label=f'LDI (med=${np.median(ldi_pv):,.0f}k)', edgecolor='white')
    ax.hist(rot_pv_floored, bins=bins, alpha=0.6, color=COLOR_ROT,
            label=f'RoT (med=${np.median(rot_pv):,.0f}k)', edgecolor='white')
    ax.axvline(x=max(np.median(ldi_pv), floor_val), color=COLOR_LDI, linestyle='--', linewidth=2)
    ax.axvline(x=max(np.median(rot_pv), floor_val), color=COLOR_ROT, linestyle='--', linewidth=2)
    ax.set_xlabel('PV Lifetime Consumption ($ 000s)')
    ax.set_ylabel('Count')
    ax.set_title('PV Consumption Distribution')
    ax.set_xscale('log')
    ax.legend(loc='upper right', fontsize=9)

    # ---- (1,1): Cumulative Stock Returns ----
    ax = axes[1, 1]
    # Stock returns have n_periods elements; use matching x-axis
    x_returns = np.arange(result.cumulative_stock_returns.shape[1])
    plot_fan_chart(ax, result.cumulative_stock_returns, x_returns, color=COLOR_STOCKS, label_prefix='')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Years from Career Start')
    ax.set_ylabel('Cumulative Return (starting at 1.0)')
    ax.set_title('Cumulative Stock Market Returns')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # ---- (1,2): Interest Rate Paths ----
    ax = axes[1, 2]
    rate_paths_pct = result.rate_paths * 100  # Convert to percentage
    x_rates = np.arange(rate_paths_pct.shape[1])
    plot_fan_chart(ax, rate_paths_pct, x_rates, color=COLOR_RATES, label_prefix='')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Years from Career Start')
    ax.set_ylabel('Interest Rate (%)')
    ax.set_title('Interest Rate Paths')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_summary_figure(
    results: Dict[str, ScenarioResult],
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Create a summary comparison figure across all three scenarios.

    Shows:
    - Default rates by scenario and strategy
    - Median PV consumption by scenario and strategy
    - Median terminal wealth by scenario and strategy
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Teaching Scenarios: LDI vs Rule-of-Thumb Summary Comparison',
                 fontsize=14, fontweight='bold')

    scenarios = ['baseline', 'sequence_risk', 'rate_shock']
    scenario_labels = ['Baseline', 'Sequence\nRisk', 'Rate\nShock']
    x = np.arange(len(scenarios))
    width = 0.35

    # Compute metrics for each scenario
    ldi_default_rates = []
    rot_default_rates = []
    ldi_median_pv = []
    rot_median_pv = []
    ldi_median_terminal = []
    rot_median_terminal = []

    for scenario in scenarios:
        r = results[scenario]
        ldi_default_rates.append(np.mean(r.ldi_default_flags) * 100)
        rot_default_rates.append(np.mean(r.rot_default_flags) * 100)
        ldi_median_pv.append(np.median(r.ldi_pv_consumption))
        rot_median_pv.append(np.median(r.rot_pv_consumption))
        ldi_median_terminal.append(np.median(r.ldi_financial_wealth_paths[:, -1]))
        rot_median_terminal.append(np.median(r.rot_financial_wealth_paths[:, -1]))

    # ---- Panel 1: Default Rates ----
    ax = axes[0]
    bars1 = ax.bar(x - width/2, ldi_default_rates, width, label='LDI', color=COLOR_LDI, alpha=0.8)
    bars2 = ax.bar(x + width/2, rot_default_rates, width, label='RoT', color=COLOR_ROT, alpha=0.8)
    ax.set_ylabel('Default Rate (%)')
    ax.set_title('Default Risk by Scenario')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    # ---- Panel 2: Median PV Consumption ----
    ax = axes[1]
    bars1 = ax.bar(x - width/2, ldi_median_pv, width, label='LDI', color=COLOR_LDI, alpha=0.8)
    bars2 = ax.bar(x + width/2, rot_median_pv, width, label='RoT', color=COLOR_ROT, alpha=0.8)
    ax.set_ylabel('Median PV Consumption ($ 000s)')
    ax.set_title('Lifetime Consumption Value')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'${height:,.0f}k',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # ---- Panel 3: Median Terminal Wealth ----
    ax = axes[2]
    bars1 = ax.bar(x - width/2, ldi_median_terminal, width, label='LDI', color=COLOR_LDI, alpha=0.8)
    bars2 = ax.bar(x + width/2, rot_median_terminal, width, label='RoT', color=COLOR_ROT, alpha=0.8)
    ax.set_ylabel('Median Terminal Wealth ($ 000s)')
    ax.set_title('Wealth at End of Life')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'${height:,.0f}k',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # Add summary statistics text box
    summary_text = "Key Takeaways:\n"
    for i, scenario in enumerate(scenarios):
        ldi_adv = ldi_median_pv[i] - rot_median_pv[i]
        summary_text += f"\n{scenario_labels[i].replace(chr(10), ' ')}: "
        summary_text += f"LDI PV +${ldi_adv:,.0f}k vs RoT"

    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    return fig


def generate_teaching_scenarios_pdf(
    output_path: str = 'output/teaching_scenarios.pdf',
    n_simulations: int = 500,
    random_seed: int = 42,
) -> None:
    """
    Generate the complete teaching scenarios PDF.

    PDF Contents:
    1. Summary comparison figure
    2. Baseline scenario figure
    3. Sequence risk scenario figure
    4. Rate shock scenario figure
    """
    print("Running teaching scenario simulations...")
    params = LifecycleParams(consumption_boost=0.0)
    econ_params = EconomicParams()

    results = run_all_teaching_scenarios(
        params=params,
        econ_params=econ_params,
        n_simulations=n_simulations,
        random_seed=random_seed,
    )

    print("Creating PDF report...")
    with PdfPages(output_path) as pdf:
        # Page 1: Summary comparison
        print("  Creating summary figure...")
        fig_summary = create_summary_figure(results)
        pdf.savefig(fig_summary, bbox_inches='tight')
        plt.close(fig_summary)

        # Pages 2-4: Individual scenario figures
        for scenario_name in ['baseline', 'sequence_risk', 'rate_shock']:
            print(f"  Creating {scenario_name} scenario figure...")
            config = SCENARIOS[scenario_name]
            fig_scenario = create_scenario_figure(
                results[scenario_name], config, params
            )
            pdf.savefig(fig_scenario, bbox_inches='tight')
            plt.close(fig_scenario)

    print(f"Saved to {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate teaching scenarios PDF comparing LDI vs Rule-of-Thumb strategies'
    )
    parser.add_argument(
        '-o', '--output',
        default='output/teaching_scenarios.pdf',
        help='Output PDF path (default: output/teaching_scenarios.pdf)'
    )
    parser.add_argument(
        '-n', '--n-simulations',
        type=int,
        default=500,
        help='Number of Monte Carlo simulations (default: 500)'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    generate_teaching_scenarios_pdf(
        output_path=args.output,
        n_simulations=args.n_simulations,
        random_seed=args.seed,
    )


if __name__ == '__main__':
    main()
