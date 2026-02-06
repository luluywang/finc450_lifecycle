"""
Generate portfolio rebalancing demo figure for FINC450.

Shows a single stochastic draw zoomed into the biggest drawdown episodes
for stocks and bonds. Each column shows:
- Top: Cumulative return (growth of $1) with the drawdown shaded
- Bottom: Purchases as % of portfolio during the same window

Key teaching point: you buy assets when they go down and sell when they go up.
The "buy low, sell high" discipline is automatic with target-weight rebalancing.

Usage:
    python generate_rebalancing_demo.py [--seed 15] [--output output/figures/rebalancing_demo.png]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

from core import (
    LifecycleParams,
    EconomicParams,
    LDIStrategy,
    simulate_with_strategy,
    generate_correlated_shocks,
)
from visualization import COLORS, apply_standard_style

apply_standard_style()


def compute_rebalancing_data(result, econ_params):
    """
    Compute dollar purchases/sales and rebalancing components.

    Raw dollar purchase at period t+1:
        purchase = fw[t+1] * w[t+1] - fw[t] * w[t] * (1 + asset_ret[t])
    """
    fw = result.financial_wealth
    w_s = result.stock_weight
    w_b = result.bond_weight
    w_c = result.cash_weight

    stock_ret = result.stock_returns
    rates = result.interest_rates

    # Reconstruct bond returns: r_t - D * delta_r + mu_bond
    D = econ_params.bond_duration
    mu_bond = econ_params.mu_bond
    delta_r = np.diff(rates)
    bond_ret = rates[:-1] - D * delta_r + mu_bond

    cash_ret = rates[:-1]

    # Portfolio return each period
    port_ret = w_s * stock_ret + w_b * bond_ret + w_c * cash_ret

    # Raw dollar purchases (period 1 onward)
    stock_after = fw[:-1] * w_s[:-1] * (1 + stock_ret[:-1])
    bond_after = fw[:-1] * w_b[:-1] * (1 + bond_ret[:-1])
    stock_purchase = fw[1:] * w_s[1:] - stock_after
    bond_purchase = fw[1:] * w_b[1:] - bond_after

    # Normalize by portfolio value at time of trade (% of portfolio)
    stock_purchase_pct = stock_purchase / fw[1:]
    bond_purchase_pct = bond_purchase / fw[1:]

    return {
        'stock_purchase': stock_purchase,
        'bond_purchase': bond_purchase,
        'stock_purchase_pct': stock_purchase_pct,
        'bond_purchase_pct': bond_purchase_pct,
        'stock_ret': stock_ret,
        'bond_ret': bond_ret,
        'port_ret': port_ret,
    }


def _find_biggest_drawdown(cum_values):
    """
    Find the biggest peak-to-trough drawdown.

    Returns (peak_idx, trough_idx, drawdown_pct).
    """
    running_max = np.maximum.accumulate(cum_values)
    drawdowns = (cum_values - running_max) / running_max

    trough_idx = np.argmin(drawdowns)
    peak_idx = np.argmax(cum_values[:trough_idx + 1])

    dd_pct = drawdowns[trough_idx]
    return peak_idx, trough_idx, dd_pct


def _drawdown_window(ages, peak_idx, trough_idx, pad_before=3, pad_after=3):
    """Return a slice of ages around the drawdown with padding."""
    start = max(0, peak_idx - pad_before)
    end = min(len(ages) - 1, trough_idx + pad_after)
    return start, end


def generate_rebalancing_demo(seed=15, output_path=None):
    """Generate the 4-panel rebalancing demo with biggest drawdowns highlighted."""
    params = LifecycleParams()
    econ = EconomicParams()

    n_periods = params.end_age - params.start_age

    # Generate one stochastic path
    rng = np.random.default_rng(seed)
    rate_shocks, stock_shocks = generate_correlated_shocks(
        n_periods, n_sims=1, rho=econ.rho, rng=rng
    )

    strategy = LDIStrategy()
    result = simulate_with_strategy(strategy, params, econ, rate_shocks, stock_shocks)

    data = compute_rebalancing_data(result, econ)

    # Growth of $1 (log scale makes drawdowns equally visible)
    growth_stock = np.cumprod(1 + data['stock_ret'])
    growth_bond = np.cumprod(1 + data['bond_ret'])

    ages = result.ages
    purchase_ages = ages[1:]

    # Find biggest drawdowns
    s_peak, s_trough, s_dd = _find_biggest_drawdown(growth_stock)
    b_peak, b_trough, b_dd = _find_biggest_drawdown(growth_bond)

    # --- Create 4-panel figure (share x-axis within columns) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[1, 0].sharex(axes[0, 0])
    axes[1, 1].sharex(axes[0, 1])
    fig.suptitle(
        "Portfolio Rebalancing: Buy Low, Sell High",
        fontsize=18, fontweight='bold', y=0.98,
    )

    stock_color = COLORS['stock']
    bond_color = COLORS['bond']
    buy_color = COLORS['teal']
    sell_color = COLORS['orange']

    # Panel 1: Cumulative stock return (log scale)
    ax1 = axes[0, 0]
    ax1.plot(ages, growth_stock, color=stock_color, linewidth=2.5)
    ax1.set_yscale('log')
    ax1.set_title("Cumulative Stock Return (growth of $1)", fontsize=13,
                   fontweight='bold')
    ax1.set_ylabel("Growth of $1 (log scale)")
    ax1.set_xlabel("Age")
    # Shade only the biggest drawdown
    ax1.axvspan(ages[s_peak], ages[s_trough], color=stock_color, alpha=0.12)
    ax1.annotate(f'{s_dd:.0%}', xy=(ages[(s_peak + s_trough) // 2], growth_stock[s_trough]),
                 fontsize=12, fontweight='bold', color=stock_color, ha='center',
                 va='top', xytext=(0, -8), textcoords='offset points')

    # Panel 2: Cumulative bond return (log scale)
    ax2 = axes[0, 1]
    ax2.plot(ages, growth_bond, color=bond_color, linewidth=2.5)
    ax2.set_yscale('log')
    ax2.set_title("Cumulative Bond Return (growth of $1)", fontsize=13,
                   fontweight='bold')
    ax2.set_ylabel("Growth of $1 (log scale)")
    ax2.set_xlabel("Age")
    ax2.axvspan(ages[b_peak], ages[b_trough], color=bond_color, alpha=0.12)
    ax2.annotate(f'{b_dd:.0%}', xy=(ages[(b_peak + b_trough) // 2], growth_bond[b_trough]),
                 fontsize=12, fontweight='bold', color=bond_color, ha='center',
                 va='top', xytext=(0, -8), textcoords='offset points')

    # Panel 3: Stock purchases as % of portfolio
    ax3 = axes[1, 0]
    sp = data['stock_purchase_pct'] * 100
    colors_s = [buy_color if v >= 0 else sell_color for v in sp]
    ax3.bar(purchase_ages, sp, color=colors_s, alpha=0.85, width=0.8)
    ax3.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
    ax3.axvspan(ages[s_peak], ages[s_trough], color=stock_color, alpha=0.08)
    ax3.set_title("Stock Purchases (% of portfolio)", fontsize=13,
                   fontweight='bold')
    ax3.set_ylabel("Purchase (% of portfolio)")
    ax3.set_xlabel("Age")

    # Panel 4: Bond purchases as % of portfolio
    ax4 = axes[1, 1]
    bp = data['bond_purchase_pct'] * 100
    colors_b = [buy_color if v >= 0 else sell_color for v in bp]
    ax4.bar(purchase_ages, bp, color=colors_b, alpha=0.85, width=0.8)
    ax4.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
    ax4.axvspan(ages[b_peak], ages[b_trough], color=bond_color, alpha=0.08)
    ax4.set_title("Bond Purchases (% of portfolio)", fontsize=13,
                   fontweight='bold')
    ax4.set_ylabel("Purchase (% of portfolio)")
    ax4.set_xlabel("Age")

    # Add retirement lines to all panels
    ret_age = params.retirement_age
    for ax in axes.flat:
        ax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Add annotation
    fig.text(
        0.5, 0.005,
        "Teal = buying  |  Orange = selling  |  "
        "Shaded band = biggest drawdown  |  "
        "Dashed line = retirement (age 65)",
        ha='center', fontsize=10, style='italic', color='gray',
    )

    plt.tight_layout(rect=[0, 0.025, 1, 0.95])

    # Save
    if output_path is None:
        output_path = "output/teaching_panels/rebalancing_demo.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved rebalancing demo to {output_path}")
    plt.close(fig)

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate rebalancing demo figure")
    parser.add_argument("--seed", type=int, default=15, help="Random seed")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path")
    args = parser.parse_args()

    generate_rebalancing_demo(seed=args.seed, output_path=args.output)
