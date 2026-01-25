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
    DEFAULT_RISKY_BETA,
    LifecycleParams,
    EconomicParams,
    simulate_with_strategy,
    generate_correlated_shocks,
    compute_pv_consumption,
    compute_pv_consumption_realized,
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
        description='Bad stock returns (~-12%/yr) in first 5 years of retirement'
    ),
    'rate_shock': ScenarioConfig(
        name='rate_shock',
        title='Pre-Retirement Rate Shock',
        description='Interest rate drop (~4% cumulative) in 5 years before retirement'
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
        # Force ~-12% annual returns: with r_bar=2%, mu=4%, sigma=18%
        # Return = r + mu + sigma*eps => -12% = 2% + 4% + 18%*eps => eps ~ -1.0
        stock_shocks[:, shock_start:shock_end] = -1.0

    elif scenario == 'rate_shock':
        # Rate shock: negative rate shock 5 years BEFORE retirement
        # This affects bond valuations and discount rates
        shock_start = max(0, working_years - 5)
        shock_end = working_years
        # Force rate decline: eps ~ -1.33 sigma per year (cumulative ~4% rate drop)
        rate_shocks[:, shock_start:shock_end] = -1.33

    return rate_shocks, stock_shocks


# =============================================================================
# Scenario Simulation
# =============================================================================

@dataclass
class ScenarioResult:
    """Results from running a scenario with both strategies."""
    scenario: str
    stock_beta: float  # Human capital stock beta used in this scenario
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
    stock_beta: float = 0.0,
) -> ScenarioResult:
    """
    Run both LDI and RoT strategies under a specific scenario.

    Args:
        scenario: Scenario name ('baseline', 'sequence_risk', 'rate_shock')
        params: Lifecycle parameters
        econ_params: Economic parameters
        base_rate_shocks: Baseline rate shocks (shared across scenarios for fair comparison)
        base_stock_shocks: Baseline stock shocks
        stock_beta: Human capital stock beta (default 0.0 = bond-like)

    Returns:
        ScenarioResult with paths and statistics for both strategies
    """
    total_years = params.end_age - params.start_age
    working_years = params.retirement_age - params.start_age
    n_sims = base_rate_shocks.shape[0]
    ages = np.arange(params.start_age, params.end_age)

    # Create params with specified stock beta
    params_with_beta = LifecycleParams(
        **{k: v for k, v in params.__dict__.items() if k != 'stock_beta_human_capital'},
        stock_beta_human_capital=stock_beta
    )

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
        ldi_strategy, params_with_beta, econ_params, rate_shocks, stock_shocks
    )

    # Run RoT strategy
    rot_result = simulate_with_strategy(
        rot_strategy, params_with_beta, econ_params, rate_shocks, stock_shocks
    )

    # Compute PV consumption for each simulation using realized rate paths
    ldi_pv_consumption = np.array([
        compute_pv_consumption_realized(
            ldi_result.consumption[sim],
            ldi_result.interest_rates[sim]
        )
        for sim in range(n_sims)
    ])
    rot_pv_consumption = np.array([
        compute_pv_consumption_realized(
            rot_result.consumption[sim],
            rot_result.interest_rates[sim]
        )
        for sim in range(n_sims)
    ])

    # Compute cumulative stock returns
    cumulative_stock_returns = np.cumprod(1 + ldi_result.stock_returns, axis=1)

    return ScenarioResult(
        scenario=scenario,
        stock_beta=stock_beta,
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
    beta_values: List[float] = None,
) -> Dict[str, Dict[float, ScenarioResult]]:
    """
    Run all teaching scenarios for each beta value.

    Uses the same baseline shocks for all scenarios to ensure fair comparison.

    Args:
        params: Lifecycle parameters
        econ_params: Economic parameters
        n_simulations: Number of Monte Carlo simulations
        random_seed: Random seed for reproducibility
        beta_values: List of human capital stock betas to run (default: [0.0, DEFAULT_RISKY_BETA])

    Returns:
        Nested dict: {scenario_name: {beta: ScenarioResult}}
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()
    if beta_values is None:
        beta_values = [0.0, DEFAULT_RISKY_BETA]

    total_years = params.end_age - params.start_age
    rng = np.random.default_rng(random_seed)

    # Generate baseline shocks (shared across all scenarios and betas)
    base_rate_shocks, base_stock_shocks = generate_correlated_shocks(
        total_years, n_simulations, econ_params.rho, rng
    )

    results = {}
    for scenario in ['baseline', 'sequence_risk', 'rate_shock']:
        results[scenario] = {}
        for beta in beta_values:
            print(f"  Running {scenario} scenario (beta={beta})...")
            results[scenario][beta] = run_teaching_scenario(
                scenario, params, econ_params, base_rate_shocks, base_stock_shocks,
                stock_beta=beta
            )

    return results


# =============================================================================
# Visualization
# =============================================================================

# Colors for strategies (colorblind-friendly)
COLOR_LDI = '#1A759F'   # Deep blue (was green)
COLOR_ROT = '#E9C46A'   # Amber (was red)
COLOR_RATES = '#3498db'  # Blue (unchanged)
COLOR_STOCKS = '#9b59b6'  # Purple (unchanged)


def plot_dodged_histogram(
    ax: plt.Axes,
    data1: np.ndarray,
    data2: np.ndarray,
    bins: np.ndarray,
    color1: str,
    color2: str,
    label1: str,
    label2: str,
    log_scale: bool = False,
) -> None:
    """
    Plot two histograms side-by-side (dodged) instead of overlaid.

    Args:
        ax: Matplotlib axes
        data1, data2: Data arrays for each histogram
        bins: Bin edges
        color1, color2: Colors for each histogram
        label1, label2: Labels for legend
        log_scale: If True, use log scale for x-axis
    """
    # Compute histogram counts
    counts1, _ = np.histogram(data1, bins=bins)
    counts2, _ = np.histogram(data2, bins=bins)

    if log_scale:
        # For log scale, compute bar positions at geometric center of bins
        bin_centers = np.sqrt(bins[:-1] * bins[1:])
        # Width as fraction of bin in log space
        log_bins = np.log10(bins)
        log_widths = np.diff(log_bins)
        # Offset each bar by 1/4 of the log width
        left_centers = 10 ** (np.log10(bin_centers) - log_widths * 0.2)
        right_centers = 10 ** (np.log10(bin_centers) + log_widths * 0.2)
        bar_width = 10 ** (log_widths * 0.35) - 1  # Approximate width ratio

        ax.bar(left_centers, counts1, width=left_centers * (10 ** (log_widths * 0.35) - 1),
               color=color1, label=label1, edgecolor='white', align='center')
        ax.bar(right_centers, counts2, width=right_centers * (10 ** (log_widths * 0.35) - 1),
               color=color2, label=label2, edgecolor='white', align='center')
        ax.set_xscale('log')
    else:
        # For linear scale, simple offset
        bin_width = bins[1] - bins[0]
        bar_width = bin_width * 0.4
        bin_centers = (bins[:-1] + bins[1:]) / 2

        ax.bar(bin_centers - bar_width / 2, counts1, width=bar_width,
               color=color1, label=label1, edgecolor='white', align='center')
        ax.bar(bin_centers + bar_width / 2, counts2, width=bar_width,
               color=color2, label=label2, edgecolor='white', align='center')


def create_scenario_figure(
    result: ScenarioResult,
    config: ScenarioConfig,
    params: LifecycleParams,
    figsize: Tuple[int, int] = (14, 12),
) -> plt.Figure:
    """
    Create a 3x2 panel figure for a single scenario.

    Panels:
    - (0,0): Cumulative Stock Returns fan chart
    - (0,1): Interest Rate Paths fan chart
    - (1,0): Financial Wealth fan charts (LDI vs RoT overlaid)
    - (1,1): Default Risk (histogram + text annotation)
    - (2,0): Terminal Wealth distribution
    - (2,1): PV Consumption distribution (Realized Rates)
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    beta_label = f"(β={result.stock_beta})" if result.stock_beta > 0 else "(β=0, Bond-like HC)"
    fig.suptitle(f'{config.title} {beta_label}\n{config.description}', fontsize=14, fontweight='bold')

    x = np.arange(len(result.ages))
    retirement_x = params.retirement_age - params.start_age
    n_sims = result.ldi_financial_wealth_paths.shape[0]

    # ---- (0,0): Cumulative Stock Returns ----
    ax = axes[0, 0]
    x_returns = np.arange(result.cumulative_stock_returns.shape[1])
    plot_fan_chart(ax, result.cumulative_stock_returns, x_returns, color=COLOR_STOCKS, label_prefix='')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Years from Career Start')
    ax.set_ylabel('Cumulative Return (starting at 1.0)')
    ax.set_title('Cumulative Stock Market Returns')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # ---- (0,1): Interest Rate Paths ----
    ax = axes[0, 1]
    rate_paths_pct = result.rate_paths * 100  # Convert to percentage
    x_rates = np.arange(rate_paths_pct.shape[1])
    plot_fan_chart(ax, rate_paths_pct, x_rates, color=COLOR_RATES, label_prefix='')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Years from Career Start')
    ax.set_ylabel('Interest Rate (%)')
    ax.set_title('Interest Rate Paths')
    ax.grid(True, alpha=0.3)

    # ---- (1,0): Financial Wealth Fan Charts ----
    ax = axes[1, 0]
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

    # ---- (1,1): Default Risk (histogram + text annotation) ----
    ax = axes[1, 1]
    ldi_default_rate = np.mean(result.ldi_default_flags) * 100
    rot_default_rate = np.mean(result.rot_default_flags) * 100

    # Show default timing histogram as main content (dodged)
    ldi_ages = result.ldi_default_ages[result.ldi_default_flags]
    rot_ages = result.rot_default_ages[result.rot_default_flags]
    bins = np.arange(params.retirement_age, params.end_age + 1, 2)

    if len(ldi_ages) > 0 or len(rot_ages) > 0:
        # Compute histogram counts
        ldi_counts, _ = np.histogram(ldi_ages, bins=bins)
        rot_counts, _ = np.histogram(rot_ages, bins=bins)
        bin_width = bins[1] - bins[0]
        bar_width = bin_width * 0.4
        bin_centers = (bins[:-1] + bins[1:]) / 2

        ax.bar(bin_centers - bar_width / 2, ldi_counts, width=bar_width,
               color=COLOR_LDI, label='LDI', edgecolor='white', align='center')
        ax.bar(bin_centers + bar_width / 2, rot_counts, width=bar_width,
               color=COLOR_ROT, label='RoT', edgecolor='white', align='center')
        ax.set_xlabel('Age at Default')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No Defaults', ha='center', va='center',
                fontsize=14, transform=ax.transAxes)
        ax.set_xlabel('Age at Default')
        ax.set_ylabel('Count')

    ax.set_title('Default Timing')

    # Add text annotation box with default rates
    textstr = f'Default Rates:\nLDI: {ldi_default_rate:.1f}%\nRoT: {rot_default_rate:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # ---- (2,0): Terminal Wealth Distribution (dodged) ----
    ax = axes[2, 0]
    ldi_final = result.ldi_financial_wealth_paths[:, -1]
    rot_final = result.rot_financial_wealth_paths[:, -1]

    floor_val = 10
    ldi_floored = np.maximum(ldi_final, floor_val)
    rot_floored = np.maximum(rot_final, floor_val)
    max_val = max(np.percentile(ldi_floored, 99), np.percentile(rot_floored, 99))
    if max_val > floor_val:
        bins = np.geomspace(floor_val, max_val, 20)
        plot_dodged_histogram(
            ax, ldi_floored, rot_floored, bins,
            COLOR_LDI, COLOR_ROT,
            f'LDI (med=${np.median(ldi_final):,.0f}k)',
            f'RoT (med=${np.median(rot_final):,.0f}k)',
            log_scale=True
        )
    else:
        bins = np.linspace(ldi_final.min(), ldi_final.max(), 20)
        plot_dodged_histogram(
            ax, ldi_final, rot_final, bins,
            COLOR_LDI, COLOR_ROT,
            f'LDI (med=${np.median(ldi_final):,.0f}k)',
            f'RoT (med=${np.median(rot_final):,.0f}k)',
            log_scale=False
        )
    ax.axvline(x=max(np.median(ldi_final), floor_val), color=COLOR_LDI, linestyle='--', linewidth=2)
    ax.axvline(x=max(np.median(rot_final), floor_val), color=COLOR_ROT, linestyle='--', linewidth=2)
    ax.set_xlabel('Terminal Wealth at Age 95 ($ 000s)')
    ax.set_ylabel('Count')
    ax.set_title('Terminal Wealth Distribution')
    ax.legend(loc='upper right', fontsize=9)

    # ---- (2,1): PV Consumption Distribution (Realized Rates, dodged) ----
    ax = axes[2, 1]
    ldi_pv = result.ldi_pv_consumption
    rot_pv = result.rot_pv_consumption

    floor_val = 100
    ldi_pv_floored = np.maximum(ldi_pv, floor_val)
    rot_pv_floored = np.maximum(rot_pv, floor_val)
    min_val = min(ldi_pv_floored.min(), rot_pv_floored.min())
    max_val = max(np.percentile(ldi_pv_floored, 99), np.percentile(rot_pv_floored, 99))
    bins = np.geomspace(min_val, max_val, 20)
    plot_dodged_histogram(
        ax, ldi_pv_floored, rot_pv_floored, bins,
        COLOR_LDI, COLOR_ROT,
        f'LDI (med=${np.median(ldi_pv):,.0f}k)',
        f'RoT (med=${np.median(rot_pv):,.0f}k)',
        log_scale=True
    )
    ax.axvline(x=max(np.median(ldi_pv), floor_val), color=COLOR_LDI, linestyle='--', linewidth=2)
    ax.axvline(x=max(np.median(rot_pv), floor_val), color=COLOR_ROT, linestyle='--', linewidth=2)
    ax.set_xlabel('PV Lifetime Consumption ($ 000s)')
    ax.set_ylabel('Count')
    ax.set_title('PV Consumption (Realized Rates)')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


def _create_single_beta_summary(
    results: Dict[str, Dict[float, ScenarioResult]],
    beta: float,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Create a simple summary figure for single-beta case (LDI vs RoT comparison).
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    beta_label = f"(β={beta})" if beta > 0 else "(β=0, Bond-like HC)"
    fig.suptitle(f'Teaching Scenarios: LDI vs Rule-of-Thumb Summary {beta_label}',
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
        r = results[scenario][beta]
        ldi_default_rates.append(np.mean(r.ldi_default_flags) * 100)
        rot_default_rates.append(np.mean(r.rot_default_flags) * 100)
        ldi_median_pv.append(np.median(r.ldi_pv_consumption))
        rot_median_pv.append(np.median(r.rot_pv_consumption))
        ldi_median_terminal.append(np.median(r.ldi_financial_wealth_paths[:, -1]))
        rot_median_terminal.append(np.median(r.rot_financial_wealth_paths[:, -1]))

    # Panel 1: Default Rates
    ax = axes[0]
    bars1 = ax.bar(x - width/2, ldi_default_rates, width, label='LDI', color=COLOR_LDI, alpha=0.8)
    bars2 = ax.bar(x + width/2, rot_default_rates, width, label='RoT', color=COLOR_ROT, alpha=0.8)
    ax.set_ylabel('Default Rate (%)')
    ax.set_title('Default Risk by Scenario')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    # Panel 2: Median PV Consumption
    ax = axes[1]
    bars1 = ax.bar(x - width/2, ldi_median_pv, width, label='LDI', color=COLOR_LDI, alpha=0.8)
    bars2 = ax.bar(x + width/2, rot_median_pv, width, label='RoT', color=COLOR_ROT, alpha=0.8)
    ax.set_ylabel('Median PV Consumption ($ 000s)')
    ax.set_title('Lifetime Consumption Value')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'${height:,.0f}k',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # Panel 3: Median Terminal Wealth
    ax = axes[2]
    bars1 = ax.bar(x - width/2, ldi_median_terminal, width, label='LDI', color=COLOR_LDI, alpha=0.8)
    bars2 = ax.bar(x + width/2, rot_median_terminal, width, label='RoT', color=COLOR_ROT, alpha=0.8)
    ax.set_ylabel('Median Terminal Wealth ($ 000s)')
    ax.set_title('Wealth at End of Life')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'${height:,.0f}k',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def create_summary_figure(
    results: Dict[str, Dict[float, ScenarioResult]],
    beta_values: List[float] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Create summary with two side-by-side comparison charts.

    Layout: 3 rows x 2 columns
    - Left column: LDI vs RoT at beta=0
    - Right column: LDI vs RoT at beta=DEFAULT_RISKY_BETA
    - Row 1: Default Rates comparison
    - Row 2: Median PV Consumption comparison
    - Row 3: Median Terminal Wealth comparison
    """
    if beta_values is None:
        # Extract beta values from results
        first_scenario = list(results.keys())[0]
        beta_values = sorted(results[first_scenario].keys())

    # Handle single-beta case (--no-risky-hc)
    if len(beta_values) == 1:
        return _create_single_beta_summary(results, beta_values[0], figsize)

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle('LDI vs Rule-of-Thumb: Strategy Comparison Across Scenarios',
                 fontsize=14, fontweight='bold')

    scenarios = ['baseline', 'sequence_risk', 'rate_shock']
    scenario_labels = ['Baseline', 'Sequence\nRisk', 'Rate\nShock']
    x = np.arange(len(scenarios))
    width = 0.35

    # Compute metrics for each scenario and beta
    metrics = {beta: {
        'ldi_default': [],
        'rot_default': [],
        'ldi_pv': [],
        'rot_pv': [],
        'ldi_terminal': [],
        'rot_terminal': [],
    } for beta in beta_values}

    for scenario in scenarios:
        for beta in beta_values:
            r = results[scenario][beta]
            metrics[beta]['ldi_default'].append(np.mean(r.ldi_default_flags) * 100)
            metrics[beta]['rot_default'].append(np.mean(r.rot_default_flags) * 100)
            metrics[beta]['ldi_pv'].append(np.median(r.ldi_pv_consumption))
            metrics[beta]['rot_pv'].append(np.median(r.rot_pv_consumption))
            metrics[beta]['ldi_terminal'].append(np.median(r.ldi_financial_wealth_paths[:, -1]))
            metrics[beta]['rot_terminal'].append(np.median(r.rot_financial_wealth_paths[:, -1]))

    # Process each beta value (column)
    for col, beta in enumerate(beta_values[:2]):  # Only use first two beta values
        beta_label = "β=0 (Bond-like HC)" if beta == 0 else f"β={beta} (Risky HC)"

        # ---- Row 0: Default Rates ----
        ax = axes[0, col]
        bars1 = ax.bar(x - width/2, metrics[beta]['ldi_default'], width,
                       label='LDI', color=COLOR_LDI, alpha=0.8)
        bars2 = ax.bar(x + width/2, metrics[beta]['rot_default'], width,
                       label='RoT', color=COLOR_ROT, alpha=0.8)
        ax.set_ylabel('Default Rate (%)')
        ax.set_title(f'Default Rates - {beta_label}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        # Add value annotations
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.5:
                    ax.annotate(f'{height:.1f}%',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

        # ---- Row 1: Median PV Consumption ----
        ax = axes[1, col]
        bars1 = ax.bar(x - width/2, metrics[beta]['ldi_pv'], width,
                       label='LDI', color=COLOR_LDI, alpha=0.8)
        bars2 = ax.bar(x + width/2, metrics[beta]['rot_pv'], width,
                       label='RoT', color=COLOR_ROT, alpha=0.8)
        ax.set_ylabel('Median PV Consumption ($k)')
        ax.set_title(f'PV Lifetime Consumption - {beta_label}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        # Add value annotations
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'${height:,.0f}k',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7)

        # ---- Row 2: Median Terminal Wealth ----
        ax = axes[2, col]
        bars1 = ax.bar(x - width/2, metrics[beta]['ldi_terminal'], width,
                       label='LDI', color=COLOR_LDI, alpha=0.8)
        bars2 = ax.bar(x + width/2, metrics[beta]['rot_terminal'], width,
                       label='RoT', color=COLOR_ROT, alpha=0.8)
        ax.set_ylabel('Median Terminal Wealth ($k)')
        ax.set_title(f'Terminal Wealth at Age 95 - {beta_label}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        # Add value annotations
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'${height:,.0f}k',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def generate_teaching_scenarios_pdf(
    output_path: str = 'output/teaching_scenarios.pdf',
    n_simulations: int = 500,
    random_seed: int = 42,
    beta_values: List[float] = None,
    include_risky_hc: bool = True,
) -> None:
    """
    Generate the complete teaching scenarios PDF.

    PDF Contents (with risky HC enabled):
    1. Summary comparison figure (all scenarios × both betas)
    2-4. Low beta scenarios (baseline, sequence_risk, rate_shock) with β=0
    5-7. High beta scenarios (baseline, sequence_risk, rate_shock) with β=DEFAULT_RISKY_BETA

    Args:
        output_path: Path for output PDF file
        n_simulations: Number of Monte Carlo simulations
        random_seed: Random seed for reproducibility
        beta_values: List of human capital stock betas (default: [0.0, DEFAULT_RISKY_BETA])
        include_risky_hc: If False, only run β=0 scenarios (4 pages total)
    """
    if beta_values is None:
        beta_values = [0.0, DEFAULT_RISKY_BETA] if include_risky_hc else [0.0]

    print("Running teaching scenario simulations...")
    params = LifecycleParams(consumption_boost=0.0)
    econ_params = EconomicParams()

    results = run_all_teaching_scenarios(
        params=params,
        econ_params=econ_params,
        n_simulations=n_simulations,
        random_seed=random_seed,
        beta_values=beta_values,
    )

    print("Creating PDF report...")
    with PdfPages(output_path) as pdf:
        # Page 1: Summary comparison
        print("  Creating summary figure...")
        fig_summary = create_summary_figure(results, beta_values)
        pdf.savefig(fig_summary, bbox_inches='tight')
        plt.close(fig_summary)

        # Pages grouped by beta, then by scenario
        for beta in beta_values:
            beta_label = "Bond-like HC" if beta == 0 else f"Risky HC (β={beta})"
            print(f"  Creating scenarios for {beta_label}...")
            for scenario_name in ['baseline', 'sequence_risk', 'rate_shock']:
                print(f"    Creating {scenario_name} scenario figure...")
                config = SCENARIOS[scenario_name]
                fig_scenario = create_scenario_figure(
                    results[scenario_name][beta], config, params
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
    parser.add_argument(
        '--risky-beta',
        type=float,
        default=DEFAULT_RISKY_BETA,
        help=f'Stock beta for risky HC comparison (default: DEFAULT_RISKY_BETA={DEFAULT_RISKY_BETA})'
    )
    parser.add_argument(
        '--no-risky-hc',
        action='store_true',
        help='Disable risky HC comparison (only run beta=0)'
    )

    args = parser.parse_args()

    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Determine beta values to run
    if args.no_risky_hc:
        beta_values = [0.0]
    else:
        beta_values = [0.0, args.risky_beta]

    generate_teaching_scenarios_pdf(
        output_path=args.output,
        n_simulations=args.n_simulations,
        random_seed=args.seed,
        beta_values=beta_values,
        include_risky_hc=not args.no_risky_hc,
    )


if __name__ == '__main__':
    main()
