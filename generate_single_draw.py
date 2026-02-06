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
    simulate_paths,
    simulate_with_strategy,
    generate_correlated_shocks,
)
from visualization import COLORS, apply_standard_style
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

    # Legacy engine — rich dict with HC decomposition, PV paths, etc.
    result = simulate_paths(
        params, econ, rate_shocks, stock_shocks,
        use_dynamic_revaluation=True,
    )

    # Strategy engine — for rebalancing computation
    strategy = LDIStrategy()
    sim_result = simulate_with_strategy(
        strategy, params, econ, rate_shocks, stock_shocks,
    )
    rebal_data = compute_rebalancing_data(sim_result, econ)

    return result, sim_result, rebal_data, params, econ


# =============================================================================
# Helper: save individual panel PNG
# =============================================================================

def _save_panel(fig, name, export_png):
    """Save a single-panel figure as PNG if export_png is True."""
    if not export_png:
        return
    Path(PNG_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(PNG_DIR) / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  -> {path}")


# =============================================================================
# Page 1: The Market You Drew (2x2)
# =============================================================================

def create_market_page(result, params, econ, export_png=True):
    """Page 1: interest rates, stock returns, cumulative stock & bond."""
    ages = np.arange(params.start_age, params.end_age)
    n = len(ages)
    rate_path = result['rate_paths'][0][:n]  # trim to n_periods
    stock_ret = result['stock_return_paths'][0]  # already n_periods-1
    bond_ret = result['bond_return_paths'][0]    # already n_periods-1
    ret_age = params.retirement_age

    stock_color = COLORS['stock']
    bond_color = COLORS['bond']
    buy_color = COLORS['teal']
    sell_color = COLORS['orange']

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("The Market You Drew", fontsize=18, fontweight='bold', y=0.98)

    # (0,0) Interest rate path
    ax = axes[0, 0]
    ax.plot(ages, rate_path, color=COLORS['rate'], linewidth=2)
    ax.axhline(y=econ.r_bar, color='gray', linestyle='--', alpha=0.7, label=f'r̄ = {econ.r_bar:.1%}')
    ax.set_title("Interest Rate Path", fontsize=13, fontweight='bold')
    ax.set_ylabel("Rate")
    ax.set_xlabel("Age")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax.legend(fontsize=9)

    # (0,1) Annual stock returns (bar chart)
    ax = axes[0, 1]
    colors_bar = [buy_color if r >= 0 else sell_color for r in stock_ret]
    ax.bar(ages, stock_ret * 100, color=colors_bar, alpha=0.85, width=0.8)
    ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
    ax.set_title("Annual Stock Returns", fontsize=13, fontweight='bold')
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Age")

    # (1,0) Cumulative stock return (growth of $1, log)
    ax = axes[1, 0]
    growth_stock = np.cumprod(1 + stock_ret)
    ax.plot(ages, growth_stock, color=stock_color, linewidth=2.5)
    ax.set_yscale('log')
    ax.set_title("Cumulative Stock Return (growth of $1)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Growth of $1 (log)")
    ax.set_xlabel("Age")

    # (1,1) Cumulative bond return (growth of $1, log)
    ax = axes[1, 1]
    growth_bond = np.cumprod(1 + bond_ret)
    ax.plot(ages, growth_bond, color=bond_color, linewidth=2.5)
    ax.set_yscale('log')
    ax.set_title("Cumulative Bond Return (growth of $1)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Growth of $1 (log)")
    ax.set_xlabel("Age")

    # Drawdown highlighting on cumulative panels
    s_peak, s_trough, s_dd = _find_biggest_drawdown(growth_stock)
    b_peak, b_trough, b_dd = _find_biggest_drawdown(growth_bond)

    axes[1, 0].axvspan(ages[s_peak], ages[s_trough], color=stock_color, alpha=0.12)
    axes[1, 0].annotate(f'{s_dd:.0%}', xy=(ages[(s_peak + s_trough) // 2], growth_stock[s_trough]),
                         fontsize=12, fontweight='bold', color=stock_color, ha='center',
                         va='top', xytext=(0, -8), textcoords='offset points')

    axes[1, 1].axvspan(ages[b_peak], ages[b_trough], color=bond_color, alpha=0.12)
    axes[1, 1].annotate(f'{b_dd:.0%}', xy=(ages[(b_peak + b_trough) // 2], growth_bond[b_trough]),
                         fontsize=12, fontweight='bold', color=bond_color, ha='center',
                         va='top', xytext=(0, -8), textcoords='offset points')

    # Retirement lines
    for ax in axes.flat:
        ax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    fig.text(
        0.5, 0.005,
        "Dashed line = retirement (age 65)  |  Shaded band = biggest drawdown",
        ha='center', fontsize=10, style='italic', color='gray',
    )
    plt.tight_layout(rect=[0, 0.025, 1, 0.95])

    # Drawdown info for standalone PNG export
    drawdown_info = {
        'stock': (s_peak, s_trough, s_dd),
        'bond': (b_peak, b_trough, b_dd),
    }

    # Export individual panels
    if export_png:
        panel_names = [
            'market_interest_rate', 'market_annual_stock_returns',
            'market_cum_stock', 'market_cum_bond',
        ]
        for i, name in enumerate(panel_names):
            pfig, pax = plt.subplots(figsize=(7, 4.5))
            src = axes.flat[i]
            # Re-draw each panel standalone
            _redraw_market_panel(pax, i, ages, rate_path, stock_ret, bond_ret,
                                 econ, ret_age, stock_color, bond_color,
                                 buy_color, sell_color, drawdown_info)
            pax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            pfig.tight_layout()
            _save_panel(pfig, name, True)
            plt.close(pfig)

    return fig, drawdown_info


def _redraw_market_panel(ax, idx, ages, rate_path, stock_ret, bond_ret,
                          econ, ret_age, stock_color, bond_color,
                          buy_color, sell_color, drawdown_info=None):
    """Re-draw a single market panel on a standalone axes."""
    if idx == 0:
        ax.plot(ages, rate_path, color=COLORS['rate'], linewidth=2)
        ax.axhline(y=econ.r_bar, color='gray', linestyle='--', alpha=0.7,
                    label=f'r̄ = {econ.r_bar:.1%}')
        ax.set_title("Interest Rate Path", fontsize=13, fontweight='bold')
        ax.set_ylabel("Rate")
        ax.set_xlabel("Age")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
        ax.legend(fontsize=9)
    elif idx == 1:
        colors_bar = [buy_color if r >= 0 else sell_color for r in stock_ret]
        ax.bar(ages, stock_ret * 100, color=colors_bar, alpha=0.85, width=0.8)
        ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
        ax.set_title("Annual Stock Returns", fontsize=13, fontweight='bold')
        ax.set_ylabel("Return (%)")
        ax.set_xlabel("Age")
    elif idx == 2:
        growth = np.cumprod(1 + stock_ret)
        ax.plot(ages, growth, color=stock_color, linewidth=2.5)
        ax.set_yscale('log')
        ax.set_title("Cumulative Stock Return (growth of $1)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Growth of $1 (log)")
        ax.set_xlabel("Age")
        if drawdown_info:
            s_peak, s_trough, s_dd = drawdown_info['stock']
            ax.axvspan(ages[s_peak], ages[s_trough], color=stock_color, alpha=0.12)
            ax.annotate(f'{s_dd:.0%}', xy=(ages[(s_peak + s_trough) // 2], growth[s_trough]),
                        fontsize=12, fontweight='bold', color=stock_color, ha='center',
                        va='top', xytext=(0, -8), textcoords='offset points')
    elif idx == 3:
        growth = np.cumprod(1 + bond_ret)
        ax.plot(ages, growth, color=bond_color, linewidth=2.5)
        ax.set_yscale('log')
        ax.set_title("Cumulative Bond Return (growth of $1)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Growth of $1 (log)")
        ax.set_xlabel("Age")
        if drawdown_info:
            b_peak, b_trough, b_dd = drawdown_info['bond']
            ax.axvspan(ages[b_peak], ages[b_trough], color=bond_color, alpha=0.12)
            ax.annotate(f'{b_dd:.0%}', xy=(ages[(b_peak + b_trough) // 2], growth[b_trough]),
                        fontsize=12, fontweight='bold', color=bond_color, ha='center',
                        va='top', xytext=(0, -8), textcoords='offset points')


# =============================================================================
# Page 2: Your Lifecycle Balance Sheet (4x2)
# =============================================================================

def create_balance_sheet_page(result, params, econ, export_png=True):
    """Page 2: earnings, cash flow, PV, HC composition, decomposition, allocation, consumption."""
    ages = np.arange(params.start_age, params.end_age)
    working_years = params.retirement_age - params.start_age
    ret_age = params.retirement_age

    # Extract single-path data (sim index 0)
    earnings = result['actual_earnings_paths'][0]
    expenses = result['expenses']
    hc = result['human_capital_paths'][0]
    pv_exp = result['pv_expenses_paths'][0]
    fw = result['financial_wealth_paths'][0]
    net_worth = result['net_worth_paths'][0]
    w_s = result['stock_weight_paths'][0]
    w_b = result['bond_weight_paths'][0]
    w_c = result['cash_weight_paths'][0]
    sub_cons = result['subsistence_consumption_paths'][0]
    var_cons = result['variable_consumption_paths'][0]
    # Use dynamic decomposition paths (vary with interest rates)
    hc_stock = result['hc_stock_paths'][0]
    hc_bond = result['hc_bond_paths'][0]
    hc_cash = result['hc_cash_paths'][0]
    exp_bond = result['exp_bond_paths'][0]
    exp_cash = result['exp_cash_paths'][0]

    fig = plt.figure(figsize=(14, 18))
    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.3,
                           top=0.96, bottom=0.03, left=0.08, right=0.97)
    fig.suptitle("Your Lifecycle Balance Sheet", fontsize=18, fontweight='bold', y=0.98)

    panel_info = []

    # (0,0) Realized Earnings vs Expenses
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ages, earnings, color=COLORS['earnings'], linewidth=2, label='Earnings')
    ax.plot(ages, expenses, color=COLORS['expenses'], linewidth=2, label='Expenses')
    ax.set_title("Realized Earnings vs Expenses", fontsize=13, fontweight='bold')
    ax.set_ylabel("$K / year")
    ax.set_xlabel("Age")
    ax.legend(fontsize=9)
    ax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)
    panel_info.append(('bs_earnings_expenses', ax))

    # (0,1) Cash Flow (savings / drawdown)
    ax = fig.add_subplot(gs[0, 1])
    cash_flow = earnings - result['total_consumption_paths'][0]
    pos = np.clip(cash_flow, 0, None)
    neg = np.clip(cash_flow, None, 0)
    ax.fill_between(ages, pos, 0, color=COLORS['savings'], alpha=0.6, label='Savings')
    ax.fill_between(ages, neg, 0, color=COLORS['drawdown'], alpha=0.6, label='Drawdown')
    ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
    ax.set_title("Cash Flow (Savings / Drawdown)", fontsize=13, fontweight='bold')
    ax.set_ylabel("$K / year")
    ax.set_xlabel("Age")
    ax.legend(fontsize=9)
    ax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)
    panel_info.append(('bs_cash_flow', ax))

    # (1,0) PV Earnings (HC) & PV Expenses
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ages, hc, color=COLORS['hc'], linewidth=2, label='PV Earnings (HC)')
    ax.plot(ages, pv_exp, color=COLORS['pv_expenses'], linewidth=2, label='PV Expenses')
    ax.set_title("PV Earnings (HC) & PV Expenses", fontsize=13, fontweight='bold')
    ax.set_ylabel("$K")
    ax.set_xlabel("Age")
    ax.legend(fontsize=9)
    ax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)
    panel_info.append(('bs_pv_earnings_expenses', ax))

    # (1,1) HC vs Financial Wealth (stacked area)
    ax = fig.add_subplot(gs[1, 1])
    ax.stackplot(ages, hc, fw,
                 labels=['Human Capital', 'Financial Wealth'],
                 colors=[COLORS['hc'], COLORS['fw']], alpha=0.8)
    ax.set_title("Total Wealth Composition", fontsize=13, fontweight='bold')
    ax.set_ylabel("$K")
    ax.set_xlabel("Age")
    ax.legend(loc='upper right', fontsize=9)
    ax.axvline(x=ret_age, color='white', linestyle='--', linewidth=2)
    panel_info.append(('bs_wealth_composition', ax))

    # (2,0) HC Decomposition (stock / bond / cash)
    ax = fig.add_subplot(gs[2, 0])
    ax.stackplot(ages, hc_stock, hc_bond, hc_cash,
                 labels=['Stock-like HC', 'Bond-like HC', 'Cash-like HC'],
                 colors=[COLORS['stock'], COLORS['bond'], COLORS['cash']],
                 alpha=0.8)
    ax.set_title("HC Decomposition", fontsize=13, fontweight='bold')
    ax.set_ylabel("$K")
    ax.set_xlabel("Age")
    ax.legend(loc='upper right', fontsize=9)
    ax.axvline(x=ret_age, color='white', linestyle='--', linewidth=2)
    panel_info.append(('bs_hc_decomposition', ax))

    # (2,1) Net HC minus Expenses components
    ax = fig.add_subplot(gs[2, 1])
    net_stock = hc_stock
    net_bond = hc_bond - exp_bond
    net_cash = hc_cash - exp_cash
    ax.plot(ages, net_stock, color=COLORS['stock'], linewidth=2, label='Net Stock')
    ax.plot(ages, net_bond, color=COLORS['bond'], linewidth=2, label='Net Bond')
    ax.plot(ages, net_cash, color=COLORS['cash'], linewidth=2, label='Net Cash')
    ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
    ax.set_title("Net HC − Expenses by Component", fontsize=13, fontweight='bold')
    ax.set_ylabel("$K")
    ax.set_xlabel("Age")
    ax.legend(fontsize=9)
    ax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)
    panel_info.append(('bs_net_hc_components', ax))

    # (3,0) Portfolio Allocation (stacked %)
    ax = fig.add_subplot(gs[3, 0])
    ax.stackplot(ages, w_s * 100, w_b * 100, w_c * 100,
                 labels=['Stocks', 'Bonds', 'Cash'],
                 colors=[COLORS['stock'], COLORS['bond'], COLORS['cash']],
                 alpha=0.8)
    ax.set_ylim(0, 100)
    ax.set_title("Portfolio Allocation", fontsize=13, fontweight='bold')
    ax.set_ylabel("Allocation (%)")
    ax.set_xlabel("Age")
    ax.legend(loc='upper right', fontsize=9)
    ax.axvline(x=ret_age, color='white', linestyle='--', linewidth=2)
    panel_info.append(('bs_portfolio_allocation', ax))

    # (3,1) Consumption Path (subsistence + variable)
    ax = fig.add_subplot(gs[3, 1])
    ax.stackplot(ages, sub_cons, var_cons,
                 labels=['Subsistence', 'Variable'],
                 colors=[COLORS['subsistence'], COLORS['variable']],
                 alpha=0.8)
    ax.set_title("Consumption Path", fontsize=13, fontweight='bold')
    ax.set_ylabel("$K / year")
    ax.set_xlabel("Age")
    ax.legend(loc='upper right', fontsize=9)
    ax.axvline(x=ret_age, color='white', linestyle='--', linewidth=2)
    panel_info.append(('bs_consumption', ax))

    # Export individual panels
    if export_png:
        for name, src_ax in panel_info:
            _export_subplot_as_png(src_ax, name, ages, ret_age, result, params, econ)

    return fig


def _export_subplot_as_png(src_ax, name, ages, ret_age, result, params, econ):
    """Export a balance-sheet panel as standalone PNG by re-creating it."""
    pfig, pax = plt.subplots(figsize=(7, 4.5))

    # Copy lines, collections, and titles from source axes
    title = src_ax.get_title()
    ylabel = src_ax.get_ylabel()
    xlabel = src_ax.get_xlabel()

    # Re-create content based on panel name
    _redraw_bs_panel(pax, name, ages, ret_age, result, params, econ)

    pax.set_title(title, fontsize=13, fontweight='bold')
    pax.set_ylabel(ylabel)
    pax.set_xlabel(xlabel)
    pfig.tight_layout()
    _save_panel(pfig, name, True)
    plt.close(pfig)


def _redraw_bs_panel(ax, name, ages, ret_age, result, params, econ):
    """Re-draw a specific balance sheet panel on standalone axes."""
    earnings = result['actual_earnings_paths'][0]
    expenses = result['expenses']
    hc = result['human_capital_paths'][0]
    pv_exp = result['pv_expenses_paths'][0]
    fw = result['financial_wealth_paths'][0]
    w_s = result['stock_weight_paths'][0]
    w_b = result['bond_weight_paths'][0]
    w_c = result['cash_weight_paths'][0]
    sub_cons = result['subsistence_consumption_paths'][0]
    var_cons = result['variable_consumption_paths'][0]
    # Use dynamic decomposition paths (vary with interest rates)
    hc_stock = result['hc_stock_paths'][0]
    hc_bond = result['hc_bond_paths'][0]
    hc_cash = result['hc_cash_paths'][0]
    exp_bond = result['exp_bond_paths'][0]
    exp_cash = result['exp_cash_paths'][0]
    total_cons = result['total_consumption_paths'][0]

    if name == 'bs_earnings_expenses':
        ax.plot(ages, earnings, color=COLORS['earnings'], linewidth=2, label='Earnings')
        ax.plot(ages, expenses, color=COLORS['expenses'], linewidth=2, label='Expenses')
        ax.legend(fontsize=9)
        ax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)
    elif name == 'bs_cash_flow':
        cf = earnings - total_cons
        pos = np.clip(cf, 0, None)
        neg = np.clip(cf, None, 0)
        ax.fill_between(ages, pos, 0, color=COLORS['savings'], alpha=0.6, label='Savings')
        ax.fill_between(ages, neg, 0, color=COLORS['drawdown'], alpha=0.6, label='Drawdown')
        ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=9)
        ax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)
    elif name == 'bs_pv_earnings_expenses':
        ax.plot(ages, hc, color=COLORS['hc'], linewidth=2, label='PV Earnings (HC)')
        ax.plot(ages, pv_exp, color=COLORS['pv_expenses'], linewidth=2, label='PV Expenses')
        ax.legend(fontsize=9)
        ax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)
    elif name == 'bs_wealth_composition':
        ax.stackplot(ages, hc, fw,
                     labels=['Human Capital', 'Financial Wealth'],
                     colors=[COLORS['hc'], COLORS['fw']], alpha=0.8)
        ax.legend(loc='upper right', fontsize=9)
        ax.axvline(x=ret_age, color='white', linestyle='--', linewidth=2)
    elif name == 'bs_hc_decomposition':
        ax.stackplot(ages, hc_stock, hc_bond, hc_cash,
                     labels=['Stock-like HC', 'Bond-like HC', 'Cash-like HC'],
                     colors=[COLORS['stock'], COLORS['bond'], COLORS['cash']],
                     alpha=0.8)
        ax.legend(loc='upper right', fontsize=9)
        ax.axvline(x=ret_age, color='white', linestyle='--', linewidth=2)
    elif name == 'bs_net_hc_components':
        ax.plot(ages, hc_stock, color=COLORS['stock'], linewidth=2, label='Net Stock')
        ax.plot(ages, hc_bond - exp_bond, color=COLORS['bond'], linewidth=2, label='Net Bond')
        ax.plot(ages, hc_cash - exp_cash, color=COLORS['cash'], linewidth=2, label='Net Cash')
        ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=9)
        ax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)
    elif name == 'bs_portfolio_allocation':
        ax.stackplot(ages, w_s * 100, w_b * 100, w_c * 100,
                     labels=['Stocks', 'Bonds', 'Cash'],
                     colors=[COLORS['stock'], COLORS['bond'], COLORS['cash']],
                     alpha=0.8)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', fontsize=9)
        ax.axvline(x=ret_age, color='white', linestyle='--', linewidth=2)
    elif name == 'bs_consumption':
        ax.stackplot(ages, sub_cons, var_cons,
                     labels=['Subsistence', 'Variable'],
                     colors=[COLORS['subsistence'], COLORS['variable']],
                     alpha=0.8)
        ax.legend(loc='upper right', fontsize=9)
        ax.axvline(x=ret_age, color='white', linestyle='--', linewidth=2)


# =============================================================================
# Page 3: Rebalancing & Outcomes (3x2)
# =============================================================================

def create_rebalancing_page(result, sim_result, rebal_data, params, econ,
                            seed, export_png=True, drawdown_info=None):
    """Page 3: wealth, net worth, rebalancing bars, net FI PV, summary."""
    ages = np.arange(params.start_age, params.end_age)
    ret_age = params.retirement_age
    purchase_ages = ages[1:]

    fw = result['financial_wealth_paths'][0]
    net_worth = result['net_worth_paths'][0]
    hc = result['human_capital_paths'][0]
    pv_exp = result['pv_expenses_paths'][0]

    buy_color = COLORS['teal']
    sell_color = COLORS['orange']

    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3,
                           top=0.95, bottom=0.05, left=0.08, right=0.97)
    fig.suptitle("Rebalancing & Outcomes", fontsize=18, fontweight='bold', y=0.98)

    panel_info = []

    # (0,0) Financial Wealth Path
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ages, fw, color=COLORS['fw'], linewidth=2.5)
    ax.set_title("Financial Wealth ($K)", fontsize=13, fontweight='bold')
    ax.set_ylabel("$K")
    ax.set_xlabel("Age")
    ax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)
    panel_info.append('rebal_financial_wealth')

    # (0,1) Net Worth (HC + FW - PV Expenses)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ages, net_worth, color=COLORS['nw'], linewidth=2.5)
    ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
    ax.set_title("Net Worth (HC + FW − PV Expenses)", fontsize=13, fontweight='bold')
    ax.set_ylabel("$K")
    ax.set_xlabel("Age")
    ax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)
    panel_info.append('rebal_net_worth')

    # (1,0) Stock rebalancing (% of portfolio)
    ax = fig.add_subplot(gs[1, 0])
    sp = rebal_data['stock_purchase_pct'] * 100
    colors_s = [buy_color if v >= 0 else sell_color for v in sp]
    ax.bar(purchase_ages, sp, color=colors_s, alpha=0.85, width=0.8)
    ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
    if drawdown_info:
        s_peak, s_trough, _ = drawdown_info['stock']
        ax.axvspan(ages[s_peak], ages[s_trough], color=COLORS['stock'], alpha=0.08)
    ax.set_title("Stock Rebalancing (% of portfolio)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Purchase (%)")
    ax.set_xlabel("Age")
    ax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)
    panel_info.append('rebal_stock_trades')

    # (1,1) Bond rebalancing (% of portfolio)
    ax = fig.add_subplot(gs[1, 1])
    bp = rebal_data['bond_purchase_pct'] * 100
    colors_b = [buy_color if v >= 0 else sell_color for v in bp]
    ax.bar(purchase_ages, bp, color=colors_b, alpha=0.85, width=0.8)
    ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
    if drawdown_info:
        b_peak, b_trough, _ = drawdown_info['bond']
        ax.axvspan(ages[b_peak], ages[b_trough], color=COLORS['bond'], alpha=0.08)
    ax.set_title("Bond Rebalancing (% of portfolio)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Purchase (%)")
    ax.set_xlabel("Age")
    ax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)
    panel_info.append('rebal_bond_trades')

    # (2,0) Net Fixed Income PV (bond hedge quality)
    ax = fig.add_subplot(gs[2, 0])
    # Net FI PV = bond holdings * FW + hc_bond - exp_bond
    w_b_path = result['bond_weight_paths'][0]
    w_c_path = result['cash_weight_paths'][0]
    fi_holdings = (w_b_path + w_c_path) * fw
    # Use dynamic decomposition paths (vary with interest rates)
    net_fi_pv = fi_holdings + result['hc_bond_paths'][0] + result['hc_cash_paths'][0] \
                - result['exp_bond_paths'][0] - result['exp_cash_paths'][0]
    ax.plot(ages, net_fi_pv, color=COLORS['bond'], linewidth=2.5)
    ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
    ax.set_title("Net Fixed Income Position", fontsize=13, fontweight='bold')
    ax.set_ylabel("$K")
    ax.set_xlabel("Age")
    ax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)
    panel_info.append('rebal_net_fi')

    # (2,1) Summary Statistics (text box)
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')

    total_cons = np.sum(result['total_consumption_paths'][0])
    terminal_fw = fw[-1]
    defaulted = bool(result['default_flags'][0])
    default_age_val = result['default_ages'][0]
    peak_fw = np.max(fw)
    peak_fw_age = ages[np.argmax(fw)]

    target_stock_pct = result['target_stock'] * 100
    target_bond_pct = result['target_bond'] * 100
    target_cash_pct = result['target_cash'] * 100

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
        f"MV Targets: {target_stock_pct:.0f}% / {target_bond_pct:.0f}% / {target_cash_pct:.0f}%",
        f"  (stock / bond / cash)",
    ]

    summary_text = "\n".join(summary_lines)
    ax.text(
        0.1, 0.95, summary_text,
        transform=ax.transAxes, fontsize=13, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#dee2e6'),
    )
    ax.set_title("Summary Statistics", fontsize=13, fontweight='bold')
    panel_info.append('rebal_summary')

    fig.text(
        0.5, 0.01,
        "Teal = buying  |  Orange = selling  |  Dashed line = retirement (age 65)  |  Shaded band = biggest drawdown",
        ha='center', fontsize=10, style='italic', color='gray',
    )

    # Export individual panels
    if export_png:
        _export_rebal_panels(result, sim_result, rebal_data, params, econ, seed, panel_info,
                             drawdown_info)

    return fig


def _export_rebal_panels(result, sim_result, rebal_data, params, econ, seed, panel_names,
                         drawdown_info=None):
    """Export rebalancing page panels as standalone PNGs."""
    ages = np.arange(params.start_age, params.end_age)
    ret_age = params.retirement_age
    purchase_ages = ages[1:]
    fw = result['financial_wealth_paths'][0]
    buy_color = COLORS['teal']
    sell_color = COLORS['orange']

    for name in panel_names:
        pfig, pax = plt.subplots(figsize=(7, 4.5))

        if name == 'rebal_financial_wealth':
            pax.plot(ages, fw, color=COLORS['fw'], linewidth=2.5)
            pax.set_title("Financial Wealth ($K)", fontsize=13, fontweight='bold')
            pax.set_ylabel("$K")
            pax.set_xlabel("Age")
            pax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)

        elif name == 'rebal_net_worth':
            nw = result['net_worth_paths'][0]
            pax.plot(ages, nw, color=COLORS['nw'], linewidth=2.5)
            pax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
            pax.set_title("Net Worth (HC + FW − PV Expenses)", fontsize=13, fontweight='bold')
            pax.set_ylabel("$K")
            pax.set_xlabel("Age")
            pax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)

        elif name == 'rebal_stock_trades':
            sp = rebal_data['stock_purchase_pct'] * 100
            colors_s = [buy_color if v >= 0 else sell_color for v in sp]
            pax.bar(purchase_ages, sp, color=colors_s, alpha=0.85, width=0.8)
            pax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
            if drawdown_info:
                s_peak, s_trough, _ = drawdown_info['stock']
                pax.axvspan(ages[s_peak], ages[s_trough], color=COLORS['stock'], alpha=0.08)
            pax.set_title("Stock Rebalancing (% of portfolio)", fontsize=13, fontweight='bold')
            pax.set_ylabel("Purchase (%)")
            pax.set_xlabel("Age")
            pax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)

        elif name == 'rebal_bond_trades':
            bp = rebal_data['bond_purchase_pct'] * 100
            colors_b = [buy_color if v >= 0 else sell_color for v in bp]
            pax.bar(purchase_ages, bp, color=colors_b, alpha=0.85, width=0.8)
            pax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
            if drawdown_info:
                b_peak, b_trough, _ = drawdown_info['bond']
                pax.axvspan(ages[b_peak], ages[b_trough], color=COLORS['bond'], alpha=0.08)
            pax.set_title("Bond Rebalancing (% of portfolio)", fontsize=13, fontweight='bold')
            pax.set_ylabel("Purchase (%)")
            pax.set_xlabel("Age")
            pax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)

        elif name == 'rebal_net_fi':
            w_b_path = result['bond_weight_paths'][0]
            w_c_path = result['cash_weight_paths'][0]
            fi_holdings = (w_b_path + w_c_path) * fw
            # Use dynamic decomposition paths (vary with interest rates)
            net_fi_pv = fi_holdings + result['hc_bond_paths'][0] + result['hc_cash_paths'][0] \
                        - result['exp_bond_paths'][0] - result['exp_cash_paths'][0]
            pax.plot(ages, net_fi_pv, color=COLORS['bond'], linewidth=2.5)
            pax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
            pax.set_title("Net Fixed Income Position", fontsize=13, fontweight='bold')
            pax.set_ylabel("$K")
            pax.set_xlabel("Age")
            pax.axvline(x=ret_age, color='gray', linestyle='--', alpha=0.5)

        elif name == 'rebal_summary':
            # Summary text panel — skip PNG export
            plt.close(pfig)
            continue

        pfig.tight_layout()
        _save_panel(pfig, name, True)
        plt.close(pfig)


# =============================================================================
# Main
# =============================================================================

def generate_single_draw(seed=42, beta=0.0, output_path=None, export_png=True):
    """Generate the complete single-draw analysis PDF."""
    if output_path is None:
        output_path = "output/single_draw.pdf"

    print(f"Generating single draw analysis (seed={seed}, beta={beta})...")

    result, sim_result, rebal_data, params, econ = generate_single_draw_data(seed, beta)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        fig1, drawdown_info = create_market_page(result, params, econ, export_png=export_png)
        pdf.savefig(fig1)
        plt.close(fig1)

        fig2 = create_balance_sheet_page(result, params, econ, export_png=export_png)
        pdf.savefig(fig2)
        plt.close(fig2)

        fig3 = create_rebalancing_page(result, sim_result, rebal_data,
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
