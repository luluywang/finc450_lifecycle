"""
PDF Report Page Creation Functions.

This module contains functions that create multi-panel figure pages for PDF reports.
These are higher-level layouts that combine multiple charts into a single page.

Single Code Path Architecture:
- Each panel has a dedicated _plot_to_ax_* function that draws to any axes
- create_base_case_page() uses these functions for both PDF grid and PNG export
- PNG export creates a standalone figure, calls the plotting function, saves, closes
- This ensures PDF and PNG outputs are always identical
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from core import LifecycleParams, EconomicParams, LifecycleResult, MonteCarloResult

from .helpers import apply_wealth_log_scale, save_panel_as_png


# Standard color palette for charts (colorblind-friendly: blue-orange palette)
REPORT_COLORS = {
    'earnings': '#0077B6',   # Teal-blue (was green)
    'expenses': '#E07A5F',   # Burnt orange (was red)
    'stock': '#F4A261',      # Coral (was blue)
    'bond': '#9b59b6',       # Purple (unchanged)
    'cash': '#f1c40f',       # Yellow (unchanged)
    'fw': '#457B9D',         # Blue (was green)
    'hc': '#e67e22',         # Orange (unchanged)
    'subsistence': '#95a5a6', # Gray (unchanged)
    'variable': '#2A9D8F',   # Teal (was red)
    'consumption': '#2A9D8F', # Teal (was red)
    'nw': '#9b59b6',         # Purple (unchanged)
    'rate': '#f39c12',       # Amber (unchanged)
    'optimal': '#1A759F',    # Deep blue (was green)
    'rot': '#E9C46A',        # Amber (was blue)
}


# =============================================================================
# Panel Plotting Functions (Single Source of Truth)
# =============================================================================
# Each function draws a specific panel to any axes object.
# These are used by both the PDF multi-panel grid and PNG individual export.

def _plot_to_ax_income_expenses(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot income and expenses panel."""
    ax.plot(x, result.earnings, color=COLORS['earnings'], linewidth=2, label='Earnings')
    ax.plot(x, result.expenses, color=COLORS['expenses'], linewidth=2, label='Expenses')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5, label='Retirement')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Income & Expenses ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)


def _plot_to_ax_cash_flow(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot cash flow panel."""
    savings = result.earnings - result.expenses
    ax.fill_between(x, 0, savings, where=savings >= 0, alpha=0.7, color=COLORS['earnings'], label='Savings')
    ax.fill_between(x, 0, savings, where=savings < 0, alpha=0.7, color=COLORS['expenses'], label='Drawdown')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Cash Flow: Earnings - Expenses ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)


def _plot_to_ax_earnings_vs_consumption(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot earnings vs total consumption with savings/drawdown fill."""
    ax.plot(x, result.earnings, color=COLORS['earnings'], linewidth=2, label='Earnings')
    ax.plot(x, result.total_consumption, color=COLORS['consumption'], linewidth=2, label='Consumption')
    diff = result.earnings - result.total_consumption
    ax.fill_between(x, 0, diff, where=diff >= 0, alpha=0.3, color=COLORS['earnings'], label='Savings')
    ax.fill_between(x, 0, diff, where=diff < 0, alpha=0.3, color=COLORS['expenses'], label='Drawdown')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Earnings vs Consumption ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)


def _plot_to_ax_present_values(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot present values panel."""
    ax.plot(x, result.pv_earnings, color=COLORS['earnings'], linewidth=2, label='PV Earnings')
    ax.plot(x, result.pv_expenses, color=COLORS['expenses'], linewidth=2, label='PV Expenses')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Present Values ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)


def _plot_to_ax_durations(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot durations panel."""
    ax.plot(x, result.duration_earnings, color=COLORS['earnings'], linewidth=2, label='Duration (Earnings)')
    ax.plot(x, result.duration_expenses, color=COLORS['expenses'], linewidth=2, label='Duration (Expenses)')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Years')
    ax.set_title('Durations (years)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)


def _plot_to_ax_hc_vs_fw(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot human capital vs financial wealth panel."""
    ax.fill_between(x, 0, result.financial_wealth, alpha=0.7, color=COLORS['fw'], label='Financial Wealth')
    ax.fill_between(x, result.financial_wealth, result.financial_wealth + result.human_capital,
                   alpha=0.7, color=COLORS['hc'], label='Human Capital')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Human Capital vs Financial Wealth ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)


def _plot_to_ax_net_wealth(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot net wealth = HC + FW - PV(expenses)."""
    ax.plot(x, result.human_capital + result.financial_wealth, color='black', linewidth=1.5,
            linestyle='--', label='Total Assets (HC+FW)')
    ax.plot(x, result.pv_expenses, color=COLORS['expenses'], linewidth=1.5,
            linestyle='--', label='PV Expenses')
    ax.fill_between(x, 0, result.net_worth, alpha=0.4, color=COLORS['nw'])
    ax.plot(x, result.net_worth, color=COLORS['nw'], linewidth=2, label='Net Worth')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Net Wealth: HC + FW − Expenses ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)


def _plot_to_ax_hc_decomposition(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot human capital decomposition panel."""
    ax.plot(x, result.hc_cash_component, color=COLORS['cash'], linewidth=2, label='HC Cash')
    ax.plot(x, result.hc_bond_component, color=COLORS['bond'], linewidth=2, label='HC Bond')
    ax.plot(x, result.hc_stock_component, color=COLORS['stock'], linewidth=2, label='HC Stock')
    ax.plot(x, result.human_capital, color='black', linewidth=1.5, linestyle='--', label='Total HC')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Human Capital Decomposition ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)


def _plot_to_ax_expense_decomposition(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot expense liability decomposition panel."""
    ax.plot(x, result.exp_cash_component, color=COLORS['cash'], linewidth=2, label='Expense Cash')
    ax.plot(x, result.exp_bond_component, color=COLORS['bond'], linewidth=2, label='Expense Bond')
    ax.plot(x, result.pv_expenses, color='black', linewidth=1.5, linestyle='--', label='Total Expenses')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Expense Liability Decomposition ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)


def _plot_to_ax_net_hc_minus_expenses(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot net HC minus expenses panel."""
    net_stock = result.hc_stock_component
    net_bond = result.hc_bond_component - result.exp_bond_component
    net_cash = result.hc_cash_component - result.exp_cash_component
    net_total = net_stock + net_bond + net_cash

    ax.plot(x, net_cash, color=COLORS['cash'], linewidth=2, label='Net Cash')
    ax.plot(x, net_bond, color=COLORS['bond'], linewidth=2, label='Net Bond')
    ax.plot(x, net_stock, color=COLORS['stock'], linewidth=2, label='Net Stock')
    ax.plot(x, net_total, color='black', linewidth=1.5, linestyle='--', label='Net Total')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Net HC minus Expenses ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)


def _plot_to_ax_consumption_path(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot consumption path panel."""
    ax.fill_between(x, 0, result.subsistence_consumption, alpha=0.7, color=COLORS['subsistence'], label='Subsistence')
    ax.fill_between(x, result.subsistence_consumption, result.total_consumption,
                   alpha=0.7, color=COLORS['variable'], label='Variable')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Consumption Path ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)


def _plot_to_ax_portfolio_allocation(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot portfolio allocation panel."""
    stock_pct = result.stock_weight_no_short * 100
    bond_pct = result.bond_weight_no_short * 100
    cash_pct = result.cash_weight_no_short * 100

    ax.fill_between(x, 0, cash_pct, alpha=0.7, color=COLORS['cash'], label='Cash')
    ax.fill_between(x, cash_pct, cash_pct + bond_pct, alpha=0.7, color=COLORS['bond'], label='Bonds')
    ax.fill_between(x, cash_pct + bond_pct, cash_pct + bond_pct + stock_pct,
                   alpha=0.7, color=COLORS['stock'], label='Stocks')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Allocation (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Portfolio Allocation (%)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)


def _plot_to_ax_net_fi_pv(ax, x, result, econ_params, COLORS, xlabel, retirement_x):
    """Plot net PV of fixed income exposures.

    Net FI PV = Bond_Holdings + HC_Bond_Component - Expense_Bond_Component
    - Zero = perfectly hedged (bond assets match bond liabilities)
    - Positive = net long bonds (overhedged)
    - Negative = net short bonds (underhedged)
    """
    bond_holdings = result.bond_weight_no_short * result.financial_wealth
    net_fi_pv = bond_holdings + result.hc_bond_component - result.exp_bond_component

    ax.plot(x, net_fi_pv, color=COLORS['bond'], linewidth=2)
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Net Fixed Income PV (Bonds + HC - Expenses)', fontweight='bold')
    ax.grid(True, alpha=0.3)


def _plot_to_ax_dv01(ax, x, result, econ_params, COLORS, xlabel, retirement_x):
    """Plot DV01 - dollar gain in net worth per 1pp rate drop.

    DV01 = (Asset_Dollar_Duration - Liability_Dollar_Duration) × 0.01

    Since hc_bond_component and exp_bond_component are already "bond-equivalent"
    amounts (i.e., they incorporate duration scaling as dur_X / bond_duration),
    we use bond_duration uniformly for all components:

    DV01 = bond_duration × (hc_bond_component + bond_holdings - exp_bond_component) × 0.01

    - Zero = perfectly hedged (no rate sensitivity)
    - Positive = net long duration (gains when rates drop)
    - Negative = net short duration (loses when rates drop)
    """
    bond_holdings = result.bond_weight_no_short * result.financial_wealth
    # All components are in bond-equivalent dollars, so use bond_duration uniformly
    total_bond_equiv_assets = result.hc_bond_component + bond_holdings
    total_bond_equiv_liabs = result.exp_bond_component
    dv01 = econ_params.bond_duration * (total_bond_equiv_assets - total_bond_equiv_liabs) * 0.01

    ax.plot(x, dv01, color=COLORS['bond'], linewidth=2)
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s) per pp')
    ax.set_title('Interest Rate Sensitivity ($ gain per 1pp rate drop)', fontweight='bold')
    ax.grid(True, alpha=0.3)


def _export_panel_as_png(plot_fn, panel_name, output_dir, figsize=(10, 6)):
    """Create a standalone figure, call plot function, save as PNG, close figure.

    Args:
        plot_fn: Callable that takes an axes and draws to it
        panel_name: Filename (without extension)
        output_dir: Output directory
        figsize: Figure size

    Returns:
        Path to saved file
    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(ax)
    path = save_panel_as_png(fig, panel_name, output_dir)
    plt.close(fig)
    return path


def create_base_case_page(
    result: 'LifecycleResult',
    params: 'LifecycleParams',
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (20, 32),
    use_years: bool = True,
    export_png: bool = False,
    png_output_dir: str = "output/teaching_panels",
    beta_str: str = "",
    is_first_beta: bool = False,
) -> plt.Figure:
    """
    Create Page 1: BASE CASE (Deterministic Median Path).

    Layout with 7 sections, 13 charts total (7x2 grid):
    - Row 0: Assumptions (Income & Expenses, Earnings vs Consumption)
    - Row 1: Forward-Looking Values (Present Values, Durations)
    - Row 2: Wealth Overview (HC vs FW, Net Wealth)
    - Row 3: Decompositions (HC Decomposition, Expense Decomposition)
    - Row 4: Net Position & Consumption (Net HC minus Expenses, Consumption Path)
    - Row 5: Portfolio & Hedging (Portfolio Allocation, Net FI PV)
    - Row 6: Interest Rate Sensitivity (DV01)

    Single Code Path: This function handles both PDF grid and PNG export.
    Each panel is drawn using a dedicated _plot_to_ax_* function.

    Args:
        result: LifecycleResult from compute_lifecycle_median_path
        params: LifecycleParams
        econ_params: EconomicParams (required for DV01 calculation)
        figsize: Figure size tuple (default taller for 7 rows)
        use_years: If True, x-axis shows years from career start
        export_png: If True, also export individual panels as PNG files
        png_output_dir: Directory for PNG files
        beta_str: Beta identifier for PNG filenames (e.g., "beta0p4")
        is_first_beta: If True, export beta-invariant panels (only for first beta)

    Returns:
        matplotlib Figure
    """
    # Create the multi-panel figure with 7x2 grid
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(7, 2, hspace=0.35, wspace=0.25)

    # Compute x-axis values
    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = result.ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    COLORS = REPORT_COLORS

    # Ensure PNG output directory exists
    if export_png:
        os.makedirs(png_output_dir, exist_ok=True)

    # =========================================================================
    # Define all panels with their grid position, plot function, and PNG config
    # =========================================================================
    # Format: (row, col, plot_fn, png_name_suffix, is_beta_invariant, needs_econ_params)
    panels = [
        # Row 0: Assumptions
        (0, 0, lambda ax: _plot_to_ax_income_expenses(ax, x, result, COLORS, xlabel, retirement_x),
         "income_expenses", True, False),
        (0, 1, lambda ax: _plot_to_ax_earnings_vs_consumption(ax, x, result, COLORS, xlabel, retirement_x),
         "earnings_vs_consumption", False, False),

        # Row 1: Forward-Looking Values
        (1, 0, lambda ax: _plot_to_ax_present_values(ax, x, result, COLORS, xlabel, retirement_x),
         "present_values", True, False),
        (1, 1, lambda ax: _plot_to_ax_durations(ax, x, result, COLORS, xlabel, retirement_x),
         "durations", True, False),

        # Row 2: Wealth Overview
        (2, 0, lambda ax: _plot_to_ax_hc_vs_fw(ax, x, result, COLORS, xlabel, retirement_x),
         "hc_vs_fw", False, False),
        (2, 1, lambda ax: _plot_to_ax_net_wealth(ax, x, result, COLORS, xlabel, retirement_x),
         "net_wealth", False, False),

        # Row 3: Decompositions
        (3, 0, lambda ax: _plot_to_ax_hc_decomposition(ax, x, result, COLORS, xlabel, retirement_x),
         "hc_decomposition", False, False),
        (3, 1, lambda ax: _plot_to_ax_expense_decomposition(ax, x, result, COLORS, xlabel, retirement_x),
         "expense_decomposition", True, False),

        # Row 4: Net Position & Consumption
        (4, 0, lambda ax: _plot_to_ax_net_hc_minus_expenses(ax, x, result, COLORS, xlabel, retirement_x),
         "net_hc_minus_expenses", False, False),
        (4, 1, lambda ax: _plot_to_ax_consumption_path(ax, x, result, COLORS, xlabel, retirement_x),
         "consumption_path", False, False),

        # Row 5: Portfolio & Hedging
        (5, 0, lambda ax: _plot_to_ax_portfolio_allocation(ax, x, result, COLORS, xlabel, retirement_x),
         "portfolio_allocation", False, False),
        (5, 1, lambda ax: _plot_to_ax_net_fi_pv(ax, x, result, econ_params, COLORS, xlabel, retirement_x),
         "net_fi_pv", False, True),

        # Row 6: Interest Rate Sensitivity
        (6, 0, lambda ax: _plot_to_ax_dv01(ax, x, result, econ_params, COLORS, xlabel, retirement_x),
         "dv01", False, True),
    ]

    # =========================================================================
    # Draw each panel to the grid AND optionally export as PNG
    # =========================================================================
    for row, col, plot_fn, png_suffix, is_beta_invariant, needs_econ in panels:
        # Draw to the multi-panel grid
        ax = fig.add_subplot(gs[row, col])
        plot_fn(ax)

        # Export as PNG if requested
        if export_png:
            # Determine whether to export this panel
            should_export = False
            if is_beta_invariant:
                # Beta-invariant panels: only export on first beta (no beta suffix)
                if is_first_beta:
                    should_export = True
                    panel_name = f"lifecycle_{png_suffix}"
            else:
                # Beta-dependent panels: always export with beta suffix
                should_export = True
                panel_name = f"lifecycle_{beta_str}_{png_suffix}"

            if should_export:
                _export_panel_as_png(plot_fn, panel_name, png_output_dir)

    fig.suptitle('PAGE 1: BASE CASE (Deterministic Median Path)', fontsize=16, fontweight='bold', y=0.995)
    return fig


def create_monte_carlo_page(
    mc_result: 'MonteCarloResult',
    params: 'LifecycleParams',
    econ_params: 'EconomicParams' = None,
    figsize: Tuple[int, int] = (20, 22),
    use_years: bool = True,
    percentiles: List[int] = None,
) -> plt.Figure:
    """
    Create Page 2: MONTE CARLO simulation results.

    Layout with 8 chart panels (4x2 grid):
    - Consumption Distribution (percentile lines)
    - Financial Wealth Distribution (percentile lines)
    - Net Worth Distribution (percentile lines)
    - Terminal Values Grid (text summary)
    - Savings Distribution (Earnings - Consumption)
    - Savings Rate (% of Earnings)
    - Cumulative Stock Returns (percentile bands)
    - Interest Rate Paths (percentile bands)

    Args:
        mc_result: MonteCarloResult from run_lifecycle_monte_carlo
        params: LifecycleParams
        econ_params: EconomicParams (optional)
        figsize: Figure size tuple
        use_years: If True, x-axis shows years from career start
        percentiles: List of percentiles to show (default: [5, 25, 50, 75, 95])

    Returns:
        matplotlib Figure
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    fig = plt.figure(figsize=(figsize[0], int(figsize[1] * 4 / 3)))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.25)

    n_sims = mc_result.financial_wealth_paths.shape[0]
    total_years = len(mc_result.ages)

    if use_years:
        x = np.arange(total_years)
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = mc_result.ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    COLORS = REPORT_COLORS

    # Compute percentiles
    consumption_pctls = np.percentile(mc_result.total_consumption_paths, percentiles, axis=0)
    fw_pctls = np.percentile(mc_result.financial_wealth_paths, percentiles, axis=0)

    net_worth_paths = (mc_result.human_capital_paths + mc_result.financial_wealth_paths -
                       mc_result.median_result.pv_expenses[np.newaxis, :])
    nw_pctls = np.percentile(net_worth_paths, percentiles, axis=0)

    stock_return_data = mc_result.stock_return_paths[:, :total_years]
    stock_cumulative = np.cumprod(1 + stock_return_data, axis=1)
    stock_pctls = np.percentile(stock_cumulative, percentiles, axis=0)

    rate_data = mc_result.interest_rate_paths[:, :total_years]
    rate_pctls = np.percentile(rate_data * 100, percentiles, axis=0)

    line_styles = {0: ':', 1: '--', 2: '-', 3: '--', 4: ':'}
    line_widths = {0: 1, 1: 1, 2: 2.5, 3: 1, 4: 1}

    # ===== Consumption Distribution =====
    ax = fig.add_subplot(gs[0, 0])
    for i, p in enumerate(percentiles):
        ax.plot(x, consumption_pctls[i], color=COLORS['consumption'],
               linestyle=line_styles[i], linewidth=line_widths[i],
               label=f'{p}th %ile' if p != 50 else 'Median')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Consumption Distribution ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    apply_wealth_log_scale(ax)

    # ===== Financial Wealth Distribution =====
    ax = fig.add_subplot(gs[0, 1])
    for i, p in enumerate(percentiles):
        ax.plot(x, fw_pctls[i], color=COLORS['fw'],
               linestyle=line_styles[i], linewidth=line_widths[i],
               label=f'{p}th %ile' if p != 50 else 'Median')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth Distribution ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    apply_wealth_log_scale(ax)

    # ===== Net Worth Distribution =====
    ax = fig.add_subplot(gs[1, 0])
    for i, p in enumerate(percentiles):
        ax.plot(x, nw_pctls[i], color=COLORS['nw'],
               linestyle=line_styles[i], linewidth=line_widths[i],
               label=f'{p}th %ile' if p != 50 else 'Median')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Net Worth Distribution (HC + FW - Expenses) ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    apply_wealth_log_scale(ax)

    # ===== Terminal Values Grid =====
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')

    terminal_fw = fw_pctls[:, -1]
    terminal_consumption = consumption_pctls[:, -1]
    depleted_count = np.sum(mc_result.financial_wealth_paths[:, -1] < 10)

    terminal_text = f"""
Terminal Values at Age {params.end_age - 1}
{'='*50}

Financial Wealth ($k):
  5th percentile:  ${terminal_fw[0]:>10,.0f}
  25th percentile: ${terminal_fw[1]:>10,.0f}
  Median:          ${terminal_fw[2]:>10,.0f}
  75th percentile: ${terminal_fw[3]:>10,.0f}
  95th percentile: ${terminal_fw[4]:>10,.0f}

Annual Consumption ($k):
  5th percentile:  ${terminal_consumption[0]:>10,.0f}
  25th percentile: ${terminal_consumption[1]:>10,.0f}
  Median:          ${terminal_consumption[2]:>10,.0f}
  75th percentile: ${terminal_consumption[3]:>10,.0f}
  95th percentile: ${terminal_consumption[4]:>10,.0f}

Runs depleted (FW < $10k): {depleted_count} of {n_sims}
Default Rate: {np.mean(mc_result.default_flags)*100:.1f}%
"""
    ax.text(0.1, 0.9, terminal_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    ax.set_title('Terminal Values Grid', fontweight='bold')

    # ===== Savings Distribution (Earnings - Consumption) =====
    ax = fig.add_subplot(gs[2, 0])
    savings_paths = mc_result.actual_earnings_paths - mc_result.total_consumption_paths
    savings_pctls = np.percentile(savings_paths, percentiles, axis=0)
    for i, p in enumerate(percentiles):
        ax.plot(x, savings_pctls[i], color=COLORS['earnings'],
               linestyle=line_styles[i], linewidth=line_widths[i],
               label=f'{p}th %ile' if p != 50 else 'Median')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Savings Distribution (Earnings - Consumption) ($k)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # ===== Savings as % of Earnings =====
    ax = fig.add_subplot(gs[2, 1])
    # Avoid division by zero: only compute ratio where earnings > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        savings_pct_paths = np.where(
            mc_result.actual_earnings_paths > 0,
            savings_paths / mc_result.actual_earnings_paths * 100,
            0.0
        )
    savings_pct_pctls = np.percentile(savings_pct_paths, percentiles, axis=0)
    for i, p in enumerate(percentiles):
        ax.plot(x, savings_pct_pctls[i], color=COLORS['earnings'],
               linestyle=line_styles[i], linewidth=line_widths[i],
               label=f'{p}th %ile' if p != 50 else 'Median')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Savings Rate (%)')
    ax.set_title('Savings Rate (% of Earnings)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)

    # ===== Cumulative Stock Returns =====
    ax = fig.add_subplot(gs[3, 0])
    log_stock_pctls = np.log(stock_pctls)

    ax.fill_between(x, log_stock_pctls[0], log_stock_pctls[1], alpha=0.15, color=COLORS['stock'])
    ax.fill_between(x, log_stock_pctls[1], log_stock_pctls[3], alpha=0.3, color=COLORS['stock'])
    ax.fill_between(x, log_stock_pctls[3], log_stock_pctls[4], alpha=0.15, color=COLORS['stock'])

    ax.plot(x, log_stock_pctls[2], color=COLORS['stock'], linewidth=2, label='Median')
    ax.plot(x, log_stock_pctls[1], color=COLORS['stock'], linewidth=1, linestyle='--', label='25th/75th')
    ax.plot(x, log_stock_pctls[3], color=COLORS['stock'], linewidth=1, linestyle='--')

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Log Cumulative Return')
    ax.set_title('Cumulative Stock Returns (Log Scale)', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)

    def log_to_pct(y, pos):
        return f'{np.exp(y)*100:.0f}%'
    ax.yaxis.set_major_formatter(plt.FuncFormatter(log_to_pct))

    # ===== Interest Rate Paths =====
    ax = fig.add_subplot(gs[3, 1])
    ax.fill_between(x, rate_pctls[0], rate_pctls[1], alpha=0.15, color=COLORS['rate'])
    ax.fill_between(x, rate_pctls[1], rate_pctls[3], alpha=0.3, color=COLORS['rate'])
    ax.fill_between(x, rate_pctls[3], rate_pctls[4], alpha=0.15, color=COLORS['rate'])

    ax.plot(x, rate_pctls[2], color=COLORS['rate'], linewidth=2, label='Median')
    ax.plot(x, rate_pctls[1], color=COLORS['rate'], linewidth=1, linestyle='--', label='25th/75th')
    ax.plot(x, rate_pctls[3], color=COLORS['rate'], linewidth=1, linestyle='--')

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Interest Rate (%)')
    ax.set_title('Interest Rate Paths (%)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(f'PAGE 2: MONTE CARLO ({n_sims} Runs)', fontsize=16, fontweight='bold', y=0.995)
    return fig


def create_scenario_page(
    scenario_type: str,
    params: 'LifecycleParams',
    econ_params: 'EconomicParams',
    figsize: Tuple[int, int] = (20, 18),
    use_years: bool = True,
    n_simulations: int = 50,
    random_seed: int = 42,
    rate_shock_years_before_retirement: int = 5,
    rate_shock_magnitude: float = -0.02,
) -> plt.Figure:
    """
    Create Teaching Scenario Pages (3a, 3b, 3c).

    Supports three scenario types:
    - 'normal': Optimal vs Rule of Thumb (normal market conditions)
    - 'sequenceRisk': Bad early returns stress test (-20% for 5 years at retirement)
    - 'rateShock': Interest rate shock at configurable age

    Each page shows:
    - Default Risk comparison
    - PV Consumption comparison (at time 0)
    - Financial Wealth percentile charts
    - Consumption percentile charts
    - Portfolio Allocation or market condition visualization

    Args:
        scenario_type: One of 'normal', 'sequenceRisk', 'rateShock'
        params: LifecycleParams
        econ_params: EconomicParams
        figsize: Figure size tuple
        use_years: If True, x-axis shows years from career start
        n_simulations: Number of Monte Carlo simulations
        random_seed: Random seed for reproducibility
        rate_shock_years_before_retirement: Years before retirement to apply rate shock (default: 5)
        rate_shock_magnitude: Magnitude of rate shock (default: -2%)

    Returns:
        matplotlib Figure
    """
    from core import (
        generate_correlated_shocks,
        simulate_interest_rates,
        simulate_stock_returns,
        run_strategy_comparison,
        compute_pv_consumption_realized,
    )

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    total_years = params.end_age - params.start_age
    working_years = params.retirement_age - params.start_age
    ages = np.arange(params.start_age, params.end_age)

    if use_years:
        x = np.arange(total_years)
        xlabel = 'Years from Career Start'
        retirement_x = working_years
    else:
        x = ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    bad_returns_early = scenario_type == 'sequenceRisk'

    rng = np.random.default_rng(random_seed)
    rate_shocks, stock_shocks = generate_correlated_shocks(
        total_years, n_simulations, econ_params.rho, rng
    )

    initial_rate = econ_params.r_bar
    rate_paths = simulate_interest_rates(
        initial_rate, total_years, n_simulations, econ_params, rate_shocks
    )

    if scenario_type == 'rateShock':
        shock_start = max(0, working_years - rate_shock_years_before_retirement)
        shock_end = working_years
        for sim in range(n_simulations):
            for t in range(shock_start, shock_end):
                rate_paths[sim, t] += rate_shock_magnitude

    stock_return_paths = simulate_stock_returns(rate_paths, econ_params, stock_shocks)

    if bad_returns_early:
        for sim in range(n_simulations):
            for t in range(working_years, min(working_years + 5, total_years)):
                stock_return_paths[sim, t] = -0.20

    comparison = run_strategy_comparison(
        params=params,
        econ_params=econ_params,
        n_simulations=n_simulations,
        random_seed=random_seed,
        bad_returns_early=bad_returns_early,
    )

    COLORS = REPORT_COLORS

    scenario_titles = {
        'normal': 'Normal Market Conditions',
        'sequenceRisk': 'Sequence Risk (Bad Early Returns)',
        'rateShock': f'Pre-Retirement Rate Shock ({rate_shock_years_before_retirement} years before retirement)',
    }

    # Define percentiles for comparison
    percentiles = [5, 25, 50, 75, 95]

    # Compute PV consumption for each simulation using realized rate paths
    if comparison.result_a.consumption.ndim == 1:
        # Single simulation
        optimal_pv_consumption = np.array([compute_pv_consumption_realized(comparison.result_a.consumption, comparison.result_a.interest_rates)])
        rot_pv_consumption = np.array([compute_pv_consumption_realized(comparison.result_b.consumption, comparison.result_b.interest_rates)])
    else:
        # Monte Carlo - compute PV for each simulation
        optimal_pv_consumption = np.array([
            compute_pv_consumption_realized(comparison.result_a.consumption[i], comparison.result_a.interest_rates[i])
            for i in range(comparison.n_sims)
        ])
        rot_pv_consumption = np.array([
            compute_pv_consumption_realized(comparison.result_b.consumption[i], comparison.result_b.interest_rates[i])
            for i in range(comparison.n_sims)
        ])

    # Compute PV consumption percentiles
    optimal_pv_consumption_percentiles = np.percentile(optimal_pv_consumption, percentiles)
    rot_pv_consumption_percentiles = np.percentile(rot_pv_consumption, percentiles)

    # Get wealth and consumption percentiles using the new API
    optimal_wealth_percentiles = comparison.wealth_percentiles('a', percentiles)
    rot_wealth_percentiles = comparison.wealth_percentiles('b', percentiles)
    optimal_consumption_percentiles = comparison.consumption_percentiles('a', percentiles)
    rot_consumption_percentiles = comparison.consumption_percentiles('b', percentiles)

    # ===== Default Risk Bar Chart =====
    ax = fig.add_subplot(gs[0, 0])
    strategies = ['Optimal\n(Variable)', 'Rule of Thumb\n(4% Rule)']
    default_rates = [comparison.default_rate('a') * 100, comparison.default_rate('b') * 100]
    colors = [COLORS['optimal'], COLORS['rot']]

    bars = ax.bar(strategies, default_rates, color=colors, alpha=0.8, edgecolor='black')
    for bar, rate in zip(bars, default_rates):
        height = bar.get_height()
        ax.annotate(f'{rate:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Default Rate (%)')
    ax.set_title('Default Risk Comparison', fontweight='bold')
    ax.set_ylim(0, max(default_rates) * 1.3 + 5)

    # ===== PV Consumption Comparison =====
    ax = fig.add_subplot(gs[0, 1])
    x_pos = np.arange(len(percentiles))
    width = 0.35

    ax.bar(x_pos - width/2, optimal_pv_consumption_percentiles,
          width, label='Optimal', color=COLORS['optimal'], alpha=0.8)
    ax.bar(x_pos + width/2, rot_pv_consumption_percentiles,
          width, label='Rule of Thumb', color=COLORS['rot'], alpha=0.8)

    ax.set_xlabel('Percentile')
    ax.set_ylabel('PV Consumption ($k)')
    ax.set_title('PV Consumption at Time 0', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{p}th' for p in percentiles])
    ax.legend(loc='upper left', fontsize=9)

    # ===== Financial Wealth Percentiles =====
    ax = fig.add_subplot(gs[1, 0])
    for i, p in enumerate(percentiles):
        if p == 50:
            ax.plot(x, optimal_wealth_percentiles[i], color=COLORS['optimal'],
                   linewidth=2, label='Optimal Median')
            ax.plot(x, rot_wealth_percentiles[i], color=COLORS['rot'],
                   linewidth=2, linestyle='--', label='RoT Median')
        elif p in [25, 75]:
            ax.plot(x, optimal_wealth_percentiles[i], color=COLORS['optimal'],
                   linewidth=1, alpha=0.6)
            ax.plot(x, rot_wealth_percentiles[i], color=COLORS['rot'],
                   linewidth=1, linestyle='--', alpha=0.6)

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth Percentiles', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)

    # ===== Consumption Percentiles =====
    ax = fig.add_subplot(gs[1, 1])
    for i, p in enumerate(percentiles):
        if p == 50:
            ax.plot(x, optimal_consumption_percentiles[i], color=COLORS['optimal'],
                   linewidth=2, label='Optimal Median')
            ax.plot(x, rot_consumption_percentiles[i], color=COLORS['rot'],
                   linewidth=2, linestyle='--', label='RoT Median')
        elif p in [25, 75]:
            ax.plot(x, optimal_consumption_percentiles[i], color=COLORS['optimal'],
                   linewidth=1, alpha=0.6)
            ax.plot(x, rot_consumption_percentiles[i], color=COLORS['rot'],
                   linewidth=1, linestyle='--', alpha=0.6)

    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Consumption Percentiles', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # ===== Scenario-specific panel =====
    if scenario_type == 'sequenceRisk':
        ax = fig.add_subplot(gs[2, 0])
        stock_data = stock_return_paths[:, :total_years]
        cumulative = np.cumprod(1 + stock_data, axis=1)
        pctls = np.percentile(cumulative, [5, 25, 50, 75, 95], axis=0)

        ax.fill_between(x, pctls[0], pctls[4], alpha=0.2, color='#3498db')
        ax.fill_between(x, pctls[1], pctls[3], alpha=0.3, color='#3498db')
        ax.plot(x, pctls[2], color='#3498db', linewidth=2, label='Median')
        ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)
        ax.axvspan(retirement_x, retirement_x + 5, alpha=0.2, color='red', label='Forced -20% Returns')

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Stock Return Paths (Showing Stress Period)', fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)

    elif scenario_type == 'rateShock':
        ax = fig.add_subplot(gs[2, 0])
        rate_data = rate_paths[:, :total_years] * 100
        pctls = np.percentile(rate_data, [5, 25, 50, 75, 95], axis=0)

        ax.fill_between(x, pctls[0], pctls[4], alpha=0.2, color='#f39c12')
        ax.fill_between(x, pctls[1], pctls[3], alpha=0.3, color='#f39c12')
        ax.plot(x, pctls[2], color='#f39c12', linewidth=2, label='Median')

        shock_start_x = working_years - rate_shock_years_before_retirement if use_years else params.retirement_age - rate_shock_years_before_retirement
        ax.axvspan(shock_start_x, retirement_x, alpha=0.2, color='red', label=f'Rate Shock Period ({rate_shock_magnitude*100:.0f}%/yr)')
        ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Interest Rate (%)')
        ax.set_title('Interest Rate Paths (Showing Shock)', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)

    else:  # normal
        ax = fig.add_subplot(gs[2, 0])
        # Get sample allocation from first simulation
        if comparison.result_b.stock_weight.ndim == 1:
            rot_stock_sample = comparison.result_b.stock_weight
            rot_bond_sample = comparison.result_b.bond_weight
        else:
            rot_stock_sample = comparison.result_b.stock_weight[0]
            rot_bond_sample = comparison.result_b.bond_weight[0]
        ax.plot(x, rot_stock_sample * 100, color=COLORS['rot'],
               linewidth=2, label='RoT Stocks')
        ax.plot(x, rot_bond_sample * 100, color='#9b59b6',
               linewidth=2, linestyle='--', label='RoT Bonds')
        ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Allocation (%)')
        ax.set_title('Rule of Thumb Glide Path', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 100)

    # ===== Summary Statistics =====
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')

    summary_text = f"""
Strategy Comparison Summary
{'='*50}

Scenario: {scenario_titles.get(scenario_type, scenario_type)}

Default Rates:
  Optimal (Variable):     {comparison.default_rate('a')*100:>6.1f}%
  Rule of Thumb (4%):     {comparison.default_rate('b')*100:>6.1f}%

Median Final Wealth ($k):
  Optimal:                ${comparison.median_final_wealth('a'):>10,.0f}
  Rule of Thumb:          ${comparison.median_final_wealth('b'):>10,.0f}

Median PV Consumption ($k):
  Optimal:                ${np.median(optimal_pv_consumption):>10,.0f}
  Rule of Thumb:          ${np.median(rot_pv_consumption):>10,.0f}

Simulations: {comparison.n_sims}
"""
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    page_num = {'normal': '3a', 'sequenceRisk': '3b', 'rateShock': '3c'}.get(scenario_type, '3')
    fig.suptitle(f'PAGE {page_num}: TEACHING SCENARIO - {scenario_titles.get(scenario_type, scenario_type)}',
                fontsize=16, fontweight='bold', y=0.995)

    return fig
