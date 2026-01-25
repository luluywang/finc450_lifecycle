"""
Human capital visualization functions.

This module provides visualizations showing the relationship between
human capital (wages) and stock returns, demonstrating that when
stock_beta_human_capital > 0, wages are correlated with stock performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from dataclasses import replace

from .styles import COLORS
from .helpers import plot_fan_chart, compute_x_axis_from_ages


def create_hc_stock_sensitivity_figure(
    params=None,
    econ_params=None,
    stock_beta: float = None,
    n_simulations: int = 1000,
    random_seed: int = 42,
    figsize: Tuple[int, int] = (12, 5),
    use_years: bool = True,
) -> plt.Figure:
    """
    Create visualization showing wage-stock return correlation.

    This figure demonstrates that when stock_beta_human_capital > 0,
    wages are correlated with stock returns:
    - In "bull markets" (high stock returns), wages tend to be ABOVE expected path
    - In "bear markets" (low stock returns), wages tend to be BELOW expected path

    The figure uses a 1x2 panel layout with single representative paths:
    - Left panel: A bull market path (90th percentile stock performance)
    - Right panel: A bear market path (10th percentile stock performance)

    Both stocks and wages from the same simulation are plotted on the
    same y-axis, clearly showing that wages are ~5x less responsive
    than stocks (since β = 0.2).

    Args:
        params: LifecycleParams (uses defaults if None)
        econ_params: EconomicParams (uses defaults if None)
        stock_beta: Beta of human capital to stocks (default DEFAULT_RISKY_BETA)
        n_simulations: Number of Monte Carlo simulations
        random_seed: Random seed for reproducibility
        figsize: Figure size tuple
        use_years: If True, x-axis shows years from career start

    Returns:
        matplotlib Figure object
    """
    # Import here to avoid circular imports
    from core import (
        DEFAULT_RISKY_BETA,
        LifecycleParams,
        EconomicParams,
        LDIStrategy,
        simulate_with_strategy,
        generate_correlated_shocks,
    )

    # Use defaults if not provided
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()
    if stock_beta is None:
        stock_beta = DEFAULT_RISKY_BETA

    # Create params with specified beta
    params_risky = replace(params, stock_beta_human_capital=stock_beta)

    # Setup simulation parameters
    rng = np.random.default_rng(random_seed)
    total_years = params.end_age - params.start_age
    working_years = params.retirement_age - params.start_age

    # Generate correlated shocks
    rate_shocks, stock_shocks = generate_correlated_shocks(
        total_years, n_simulations, econ_params.rho, rng
    )

    # Run simulation with risky human capital
    strategy = LDIStrategy(allow_leverage=False)
    result = simulate_with_strategy(
        strategy, params_risky, econ_params, rate_shocks, stock_shocks
    )

    # Compute cumulative stock returns (growth of $1 invested)
    # stock_returns has shape (n_sims, n_periods-1) for returns between periods
    cum_returns = np.cumprod(1 + result.stock_returns, axis=1)

    # Get cumulative returns at end of working years
    # Note: cum_returns has one fewer column than working_years
    final_cum_returns = cum_returns[:, working_years - 2]  # -2 because returns are between periods

    # Sort simulations by stock performance and pick representative paths
    sorted_idx = np.argsort(final_cum_returns)
    # Pick paths at 90th and 10th percentile
    bull_path_idx = sorted_idx[int(n_simulations * 0.90)]  # 90th percentile
    bear_path_idx = sorted_idx[int(n_simulations * 0.10)]  # 10th percentile

    # Pad cumulative returns with initial value of 1.0 for plotting
    cum_returns_padded = np.ones((n_simulations, working_years))
    cum_returns_padded[:, 1:] = cum_returns[:, :working_years-1]

    # Extract single paths
    bull_stocks = cum_returns_padded[bull_path_idx, :]
    bear_stocks = cum_returns_padded[bear_path_idx, :]
    bull_wages = result.earnings[bull_path_idx, :working_years]
    bear_wages = result.earnings[bear_path_idx, :working_years]

    # Get expected paths (deterministic, zero shocks)
    zero_rate_shocks = np.zeros((1, total_years))
    zero_stock_shocks = np.zeros((1, total_years))
    expected_result = simulate_with_strategy(
        strategy, params_risky, econ_params, zero_rate_shocks, zero_stock_shocks
    )
    # Note: single sim results are 1D (squeezed), so index directly
    expected_wages = expected_result.earnings[:working_years]

    # Compute expected cumulative stock returns (deterministic path)
    expected_stock_returns = expected_result.stock_returns[:working_years-1]
    expected_cum_returns = np.ones(working_years)
    expected_cum_returns[1:] = np.cumprod(1 + expected_stock_returns)

    # Convert to ratios relative to expected
    bull_stocks_rel = bull_stocks / expected_cum_returns
    bear_stocks_rel = bear_stocks / expected_cum_returns
    bull_wages_rel = bull_wages / expected_wages
    bear_wages_rel = bear_wages / expected_wages

    # Setup x-axis
    ages = np.arange(params.start_age, params.start_age + working_years)
    x, xlabel, retirement_x = compute_x_axis_from_ages(
        ages, params.start_age, params.retirement_age, use_years
    )

    # Create figure with 1x2 layout
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Color scheme
    stock_color = COLORS['stock']      # Coral for stocks
    wage_color = COLORS['earnings']    # Teal-blue for wages/earnings

    # Panel 0: Bull market path (90th percentile)
    ax = axes[0]
    ax.plot(x, bull_stocks_rel, color=stock_color, linewidth=2.5, label='Stocks')
    ax.plot(x, bull_wages_rel, color=wage_color, linewidth=2.5, label='Wages')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='Expected')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Value / Expected')
    ax.set_title('Bull Market Path (90th percentile)', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(x[0], x[-1])
    ax.set_yscale('log')

    # Panel 1: Bear market path (10th percentile)
    ax = axes[1]
    ax.plot(x, bear_stocks_rel, color=stock_color, linewidth=2.5, label='Stocks')
    ax.plot(x, bear_wages_rel, color=wage_color, linewidth=2.5, label='Wages')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='Expected')
    ax.set_xlabel(xlabel)
    ax.set_title('Bear Market Path (10th percentile)', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(x[0], x[-1])

    # Ensure y-axis is shared and symmetric around 1.0 on log scale
    all_values = np.concatenate([bull_stocks_rel, bear_stocks_rel,
                                  bull_wages_rel, bear_wages_rel])
    max_ratio = max(all_values.max(), 1.0 / all_values.min())
    # Add some padding
    max_ratio *= 1.1
    axes[0].set_ylim(1.0 / max_ratio, max_ratio)

    # Add overall title with beta annotation
    fig.suptitle(
        f'Human Capital Sensitivity to Stock Returns (β = {stock_beta})\n'
        'Wages move with stocks, but with {:.0f}% of the magnitude'.format(stock_beta * 100),
        fontsize=12, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    return fig
