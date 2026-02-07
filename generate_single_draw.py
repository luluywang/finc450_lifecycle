"""
Generate single random draw scenario analysis for FINC450.

Shows what happens to ONE random simulation path across the full lifecycle.
Produces a multi-page PDF + individual PNG panels for teaching slides.

Three pages:
  Page 1: "The Market You Drew" — interest rates, stock/bond returns
  Page 2: "Your Lifecycle Balance Sheet" — earnings, HC, wealth, allocation
  Page 3: "Rebalancing & Outcomes" — wealth paths, rebalancing, summary

Usage:
    python generate_single_draw.py [--seed 42] [--beta 0.0] [-o output/single_draw.pdf]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from pathlib import Path

from core import (
    LifecycleParams,
    EconomicParams,
    LDIStrategy,
    simulate_with_strategy,
    generate_correlated_shocks,
    compute_target_allocations,
)
from core.simulation import _sim_result_to_lifecycle_result
from visualization import COLORS, apply_standard_style
from visualization.report_pages import (
    REPORT_COLORS,
    _plot_to_ax_income_expenses,
    _plot_to_ax_earnings_vs_consumption,
    _plot_to_ax_present_values,
    _plot_to_ax_hc_vs_fw,
    _plot_to_ax_hc_decomposition,
    _plot_to_ax_net_hc_minus_expenses,
    _plot_to_ax_portfolio_allocation,
    _plot_to_ax_consumption_path,
    _plot_to_ax_net_fi_pv,
    _export_panel_as_png,
)
from generate_rebalancing_demo import compute_rebalancing_data, _find_biggest_drawdown

apply_standard_style()

PNG_DIR = "output/teaching_panels/single_draw"


# =============================================================================
# Data generation
# =============================================================================

def generate_single_draw_data(seed, beta=0.0):
    """Generate all data for a single random draw."""
    params = LifecycleParams(stock_beta_human_capital=beta)
    econ = EconomicParams()
    n_periods = params.end_age - params.start_age

    rng = np.random.default_rng(seed)
    rate_shocks, stock_shocks = generate_correlated_shocks(
        n_periods, 1, econ.rho, rng
    )

    strategy = LDIStrategy()
    sim_result = simulate_with_strategy(
        strategy, params, econ, rate_shocks, stock_shocks,
    )

    # Convert to LifecycleResult for reuse of _plot_to_ax_* functions
    lifecycle_result = _sim_result_to_lifecycle_result(sim_result, params, econ)

    rebal_data = compute_rebalancing_data(sim_result, econ)

    return sim_result, lifecycle_result, rebal_data, params, econ


# =============================================================================
# Helper: save individual panel PNG
# =============================================================================

def _save_panel(fig, name):
    """Save a single-panel figure as PNG."""
    Path(PNG_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(PNG_DIR) / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  -> {path}")


def _export_and_print(plot_fn, name):
    """Export a panel as PNG using report_pages pattern, with print output."""
    path = _export_panel_as_png(plot_fn, name, PNG_DIR, figsize=(7, 4.5))
    print(f"  -> {path}")


# =============================================================================
# Page 1: The Market You Drew (2x2)
# =============================================================================

def _plot_market_interest_rate(ax, ages, rate_path, econ, ret_age):
    """Draw interest rate path panel."""
    ax.plot(ages, rate_path, color=COLORS['rate'], linewidth=2)
    ax.axhline(y=econ.r_bar, color='gray', linestyle='--', alpha=0.7,
               label=f'r\u0304 = {econ.r_bar:.1%}')
    ax.set_title("Interest Rate Path", fontweight='bold')
    ax.set_ylabel("Rate")
    ax.set_xlabel("Age")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax.legend(fontsize=9)
    ax.axvline(x=ret_age, color='gray', linestyle=':', alpha=0.5, linewidth=1)


def _plot_market_annual_stock_returns(ax, ages, stock_ret, ret_age):
    """Draw annual stock returns bar chart panel."""
    buy_color = COLORS['teal']
    sell_color = COLORS['orange']
    colors_bar = [buy_color if r >= 0 else sell_color for r in stock_ret]
    ax.bar(ages, stock_ret * 100, color=colors_bar, alpha=0.85, width=0.8)
    ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
    ax.set_title("Annual Stock Returns", fontweight='bold')
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Age")
    ax.axvline(x=ret_age, color='gray', linestyle=':', alpha=0.5, linewidth=1)


def _plot_market_cumulative(ax, ages, returns, color, title, drawdown_info_entry=None):
    """Draw cumulative return (growth of $1) panel with optional drawdown."""
    growth = np.cumprod(1 + returns)
    ax.plot(ages, growth, color=color, linewidth=2.5)
    ax.set_yscale('log')
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel("Growth of $1 (log)")
    ax.set_xlabel("Age")
    if drawdown_info_entry:
        peak, trough, dd = drawdown_info_entry
        ax.axvspan(ages[peak], ages[trough], color=color, alpha=0.12)
        ax.annotate(f'{dd:.0%}', xy=(ages[(peak + trough) // 2], growth[trough]),
                    fontsize=12, fontweight='bold', color=color, ha='center',
                    va='top', xytext=(0, -8), textcoords='offset points')


def create_market_page(sim_result, params, econ, export_png=True):
    """Page 1: interest rates, stock returns, cumulative stock & bond."""
    ages = sim_result.ages
    n = len(ages)
    rate_path = sim_result.interest_rates[:n]
    stock_ret = sim_result.stock_returns
    # Reconstruct bond returns for cumulative chart
    D = econ.bond_duration
    mu_bond = econ.mu_bond
    delta_r = np.diff(sim_result.interest_rates)
    bond_ret = sim_result.interest_rates[:-1] - D * delta_r + mu_bond
    ret_age = params.retirement_age

    stock_color = COLORS['stock']
    bond_color = COLORS['bond']

    # Compute drawdown info
    growth_stock = np.cumprod(1 + stock_ret)
    growth_bond = np.cumprod(1 + bond_ret)
    s_peak, s_trough, s_dd = _find_biggest_drawdown(growth_stock)
    b_peak, b_trough, b_dd = _find_biggest_drawdown(growth_bond)
    drawdown_info = {
        'stock': (s_peak, s_trough, s_dd),
        'bond': (b_peak, b_trough, b_dd),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("The Market You Drew", fontsize=18, fontweight='bold', y=0.98)

    _plot_market_interest_rate(axes[0, 0], ages, rate_path, econ, ret_age)
    _plot_market_annual_stock_returns(axes[0, 1], ages, stock_ret, ret_age)
    _plot_market_cumulative(axes[1, 0], ages, stock_ret, stock_color,
                            "Cumulative Stock Return (growth of $1)",
                            drawdown_info['stock'])
    axes[1, 0].axvline(x=ret_age, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    _plot_market_cumulative(axes[1, 1], ages, bond_ret, bond_color,
                            "Cumulative Bond Return (growth of $1)",
                            drawdown_info['bond'])
    axes[1, 1].axvline(x=ret_age, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    fig.text(
        0.5, 0.005,
        "Dotted line = retirement (age 65)  |  Shaded band = biggest drawdown",
        ha='center', fontsize=10, style='italic', color='gray',
    )
    plt.tight_layout(rect=[0, 0.025, 1, 0.95])

    # Export individual panels as standalone PNGs
    if export_png:
        Path(PNG_DIR).mkdir(parents=True, exist_ok=True)
        panel_fns = [
            ('market_interest_rate',
             lambda ax: _plot_market_interest_rate(ax, ages, rate_path, econ, ret_age)),
            ('market_annual_stock_returns',
             lambda ax: _plot_market_annual_stock_returns(ax, ages, stock_ret, ret_age)),
            ('market_cum_stock',
             lambda ax: (_plot_market_cumulative(ax, ages, stock_ret, stock_color,
                          "Cumulative Stock Return (growth of $1)", drawdown_info['stock']),
                         ax.axvline(x=ret_age, color='gray', linestyle=':', alpha=0.5, linewidth=1))),
            ('market_cum_bond',
             lambda ax: (_plot_market_cumulative(ax, ages, bond_ret, bond_color,
                          "Cumulative Bond Return (growth of $1)", drawdown_info['bond']),
                         ax.axvline(x=ret_age, color='gray', linestyle=':', alpha=0.5, linewidth=1))),
        ]
        for name, plot_fn in panel_fns:
            pfig, pax = plt.subplots(figsize=(7, 4.5))
            plot_fn(pax)
            pfig.tight_layout()
            _save_panel(pfig, name)
            plt.close(pfig)

    return fig, drawdown_info


# =============================================================================
# Page 2: Your Lifecycle Balance Sheet (4x2)
# =============================================================================

def create_balance_sheet_page(lifecycle_result, params, econ, export_png=True):
    """Page 2: earnings, cash flow, PV, HC composition, decomposition, allocation, consumption.

    Uses _plot_to_ax_* functions from report_pages.py for visual consistency
    with lifecycle_strategy.pdf.
    """
    ages = lifecycle_result.ages
    ret_age = params.retirement_age
    C = REPORT_COLORS
    xlabel = 'Age'

    fig = plt.figure(figsize=(14, 18))
    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.3,
                           top=0.96, bottom=0.03, left=0.08, right=0.97)
    fig.suptitle("Your Lifecycle Balance Sheet", fontsize=18, fontweight='bold', y=0.98)

    # Define all panels: (row, col, plot_fn, png_name)
    panels = [
        (0, 0, lambda ax: _plot_to_ax_income_expenses(ax, ages, lifecycle_result, C, xlabel, ret_age),
         'bs_earnings_expenses'),
        (0, 1, lambda ax: _plot_to_ax_earnings_vs_consumption(ax, ages, lifecycle_result, C, xlabel, ret_age),
         'bs_cash_flow'),
        (1, 0, lambda ax: _plot_to_ax_present_values(ax, ages, lifecycle_result, C, xlabel, ret_age),
         'bs_pv_earnings_expenses'),
        (1, 1, lambda ax: _plot_to_ax_hc_vs_fw(ax, ages, lifecycle_result, C, xlabel, ret_age),
         'bs_wealth_composition'),
        (2, 0, lambda ax: _plot_to_ax_hc_decomposition(ax, ages, lifecycle_result, C, xlabel, ret_age),
         'bs_hc_decomposition'),
        (2, 1, lambda ax: _plot_to_ax_net_hc_minus_expenses(ax, ages, lifecycle_result, C, xlabel, ret_age),
         'bs_net_hc_components'),
        (3, 0, lambda ax: _plot_to_ax_portfolio_allocation(ax, ages, lifecycle_result, C, xlabel, ret_age),
         'bs_portfolio_allocation'),
        (3, 1, lambda ax: _plot_to_ax_consumption_path(ax, ages, lifecycle_result, C, xlabel, ret_age),
         'bs_consumption'),
    ]

    for row, col, plot_fn, png_name in panels:
        ax = fig.add_subplot(gs[row, col])
        plot_fn(ax)

        if export_png:
            _export_and_print(plot_fn, png_name)

    return fig


# =============================================================================
# Page 3: Rebalancing & Outcomes (3x2)
# =============================================================================

def _plot_rebal_financial_wealth(ax, ages, fw, ret_age):
    """Draw financial wealth path panel."""
    ax.plot(ages, fw, color=COLORS['fw'], linewidth=2.5)
    ax.set_title("Financial Wealth ($k)", fontweight='bold')
    ax.set_ylabel("$ (000s)")
    ax.set_xlabel("Age")
    ax.axvline(x=ret_age, color='gray', linestyle=':', alpha=0.5)


def _plot_rebal_net_worth(ax, ages, net_worth, ret_age):
    """Draw net worth path panel."""
    ax.plot(ages, net_worth, color=COLORS['nw'], linewidth=2.5)
    ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
    ax.set_title("Net Worth (HC + FW \u2212 PV Expenses)", fontweight='bold')
    ax.set_ylabel("$ (000s)")
    ax.set_xlabel("Age")
    ax.axvline(x=ret_age, color='gray', linestyle=':', alpha=0.5)


def _plot_rebal_trades(ax, purchase_ages, purchase_pct, asset_color,
                       drawdown_entry, ages, title, ret_age):
    """Draw stock or bond rebalancing bar chart."""
    buy_color = COLORS['teal']
    sell_color = COLORS['orange']
    vals = purchase_pct * 100
    colors_bar = [buy_color if v >= 0 else sell_color for v in vals]
    ax.bar(purchase_ages, vals, color=colors_bar, alpha=0.85, width=0.8)
    ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
    if drawdown_entry:
        peak, trough, _ = drawdown_entry
        ax.axvspan(ages[peak], ages[trough], color=asset_color, alpha=0.08)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel("Purchase (%)")
    ax.set_xlabel("Age")
    ax.axvline(x=ret_age, color='gray', linestyle=':', alpha=0.5)


def create_rebalancing_page(sim_result, lifecycle_result, rebal_data, params, econ,
                            seed, export_png=True, drawdown_info=None):
    """Page 3: wealth, net worth, rebalancing bars, net FI PV, summary."""
    ages = sim_result.ages
    ret_age = params.retirement_age
    purchase_ages = ages[1:]

    fw = sim_result.financial_wealth
    net_worth = sim_result.net_worth

    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3,
                           top=0.95, bottom=0.05, left=0.08, right=0.97)
    fig.suptitle("Rebalancing & Outcomes", fontsize=18, fontweight='bold', y=0.98)

    # (0,0) Financial Wealth Path
    ax = fig.add_subplot(gs[0, 0])
    _plot_rebal_financial_wealth(ax, ages, fw, ret_age)

    # (0,1) Net Worth
    ax = fig.add_subplot(gs[0, 1])
    _plot_rebal_net_worth(ax, ages, net_worth, ret_age)

    # (1,0) Stock rebalancing
    ax = fig.add_subplot(gs[1, 0])
    _plot_rebal_trades(ax, purchase_ages, rebal_data['stock_purchase_pct'],
                       COLORS['stock'],
                       drawdown_info.get('stock') if drawdown_info else None,
                       ages, "Stock Rebalancing (% of portfolio)", ret_age)

    # (1,1) Bond rebalancing
    ax = fig.add_subplot(gs[1, 1])
    _plot_rebal_trades(ax, purchase_ages, rebal_data['bond_purchase_pct'],
                       COLORS['bond'],
                       drawdown_info.get('bond') if drawdown_info else None,
                       ages, "Bond Rebalancing (% of portfolio)", ret_age)

    # (2,0) Net Fixed Income PV — reuse from report_pages
    ax = fig.add_subplot(gs[2, 0])
    _plot_to_ax_net_fi_pv(ax, ages, lifecycle_result, econ, REPORT_COLORS, 'Age', ret_age)

    # (2,1) Summary Statistics
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')

    total_cons = np.sum(sim_result.consumption)
    terminal_fw = fw[-1]
    defaulted = bool(sim_result.defaulted)
    default_age_val = sim_result.default_age
    peak_fw = np.max(fw)
    peak_fw_age = ages[np.argmax(fw)]

    target_stock, target_bond, target_cash = compute_target_allocations(params, econ)

    summary_lines = [
        f"Seed: {seed}",
        f"Beta (HC): {params.stock_beta_human_capital:.1f}",
        "",
        f"Terminal Wealth: ${terminal_fw:,.0f}K",
        f"Peak Wealth: ${peak_fw:,.0f}K (age {peak_fw_age})",
        f"Total Consumption: ${total_cons:,.0f}K",
        "",
        f"Defaulted: {'YES' if defaulted else 'No'}",
    ]
    if defaulted:
        summary_lines.append(f"Default Age: {default_age_val:.0f}")

    summary_lines += [
        "",
        f"MV Targets: {target_stock*100:.0f}% / {target_bond*100:.0f}% / {target_cash*100:.0f}%",
        f"  (stock / bond / cash)",
    ]

    summary_text = "\n".join(summary_lines)
    ax.text(
        0.1, 0.95, summary_text,
        transform=ax.transAxes, fontsize=13, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#dee2e6'),
    )
    ax.set_title("Summary Statistics", fontweight='bold')

    fig.text(
        0.5, 0.01,
        "Teal = buying  |  Orange = selling  |  Dotted line = retirement (age 65)  |  Shaded band = biggest drawdown",
        ha='center', fontsize=10, style='italic', color='gray',
    )

    # Export individual panels as standalone PNGs
    if export_png:
        Path(PNG_DIR).mkdir(parents=True, exist_ok=True)
        stock_dd = drawdown_info.get('stock') if drawdown_info else None
        bond_dd = drawdown_info.get('bond') if drawdown_info else None

        rebal_panels = [
            ('rebal_financial_wealth',
             lambda ax: _plot_rebal_financial_wealth(ax, ages, fw, ret_age)),
            ('rebal_net_worth',
             lambda ax: _plot_rebal_net_worth(ax, ages, net_worth, ret_age)),
            ('rebal_stock_trades',
             lambda ax: _plot_rebal_trades(ax, purchase_ages, rebal_data['stock_purchase_pct'],
                                           COLORS['stock'], stock_dd, ages,
                                           "Stock Rebalancing (% of portfolio)", ret_age)),
            ('rebal_bond_trades',
             lambda ax: _plot_rebal_trades(ax, purchase_ages, rebal_data['bond_purchase_pct'],
                                           COLORS['bond'], bond_dd, ages,
                                           "Bond Rebalancing (% of portfolio)", ret_age)),
            ('rebal_net_fi',
             lambda ax: _plot_to_ax_net_fi_pv(ax, ages, lifecycle_result, econ,
                                               REPORT_COLORS, 'Age', ret_age)),
        ]
        for name, plot_fn in rebal_panels:
            _export_and_print(plot_fn, name)

    return fig


# =============================================================================
# Main
# =============================================================================

def generate_single_draw(seed=42, beta=0.0, output_path=None, export_png=True):
    """Generate the complete single-draw analysis PDF."""
    if output_path is None:
        output_path = "output/single_draw.pdf"

    print(f"Generating single draw analysis (seed={seed}, beta={beta})...")

    sim_result, lifecycle_result, rebal_data, params, econ = generate_single_draw_data(seed, beta)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        fig1, drawdown_info = create_market_page(sim_result, params, econ,
                                                 export_png=export_png)
        pdf.savefig(fig1)
        plt.close(fig1)

        fig2 = create_balance_sheet_page(lifecycle_result, params, econ,
                                         export_png=export_png)
        pdf.savefig(fig2)
        plt.close(fig2)

        fig3 = create_rebalancing_page(sim_result, lifecycle_result, rebal_data,
                                       params, econ, seed, export_png=export_png,
                                       drawdown_info=drawdown_info)
        pdf.savefig(fig3)
        plt.close(fig3)

    print(f"Saved PDF to {output_path}")
    if export_png:
        print(f"Saved PNGs to {PNG_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate single random draw lifecycle analysis"
    )
    parser.add_argument("--seed", type=int, default=61, help="Random seed (default: 61)")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="HC stock beta (default: 0.0)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output PDF path")
    parser.add_argument("--no-png", action="store_true",
                        help="Skip individual PNG export")
    args = parser.parse_args()

    generate_single_draw(
        seed=args.seed,
        beta=args.beta,
        output_path=args.output,
        export_png=not args.no_png,
    )
