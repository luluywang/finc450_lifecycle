"""
Lifecycle Investment Strategy Report Generation

This module orchestrates PDF report generation by combining core simulation
functions with visualization components. All computation and plotting logic
has been factored out to the core and visualization modules.

Author: FINC 450 Life Cycle Investing
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Core computation imports
from core import (
    # Constants
    DEFAULT_RISKY_BETA,
    # Dataclasses
    EconomicParams,
    LifecycleParams,
    MonteCarloParams,
    # Economic functions
    compute_mv_optimal_allocation,
    # Simulation functions
    compute_lifecycle_median_path,
    compute_median_path_comparison,
    run_lifecycle_monte_carlo,
)

# Visualization imports
from visualization import (
    # Report page layouts
    create_base_case_page,
    create_monte_carlo_page,
    create_scenario_page,
    # Comparison plots
    create_median_path_comparison_figure,
    create_allocation_comparison_page,
    # Sensitivity plots
    create_beta_comparison_figure,
    create_gamma_comparison_figure,
    create_initial_wealth_comparison_figure,
    create_equity_premium_comparison_figure,
    create_volatility_comparison_figure,
    # PNG export utility
    save_panel_as_png,
    REPORT_COLORS,
)


def _save_base_case_panels(
    result,
    params: LifecycleParams,
    beta: float,
    use_years: bool = True,
    output_dir: str = "output/teaching_panels",
    is_first_beta: bool = False,
) -> list:
    """
    Create and save individual PNG panels for a base case page.

    This extracts each of the 10 panels from the base case layout and saves them
    individually at 300 DPI for PowerPoint integration.

    Beta-invariant panels (income_expenses, cash_flow, present_values, durations,
    expense_decomposition) are only saved once when is_first_beta=True.

    Args:
        result: LifecycleResult from compute_lifecycle_median_path
        params: LifecycleParams
        beta: Stock beta value for naming
        use_years: If True, x-axis shows years from career start
        output_dir: Directory to save PNG files
        is_first_beta: If True, save beta-invariant panels (call with True for first beta only)

    Returns:
        List of saved file paths
    """
    saved_paths = []
    COLORS = REPORT_COLORS

    # Compute x-axis values
    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = result.ages
        xlabel = 'Age'
        retirement_x = params.retirement_age

    beta_str = f"beta{beta}".replace(".", "p")

    # Beta-invariant panels: only save once (no beta suffix)
    beta_invariant_panels = [
        ("income_expenses", lambda ax: _plot_income_expenses(ax, x, result, COLORS, xlabel, retirement_x)),
        ("cash_flow", lambda ax: _plot_cash_flow(ax, x, result, COLORS, xlabel, retirement_x)),
        ("present_values", lambda ax: _plot_present_values(ax, x, result, COLORS, xlabel, retirement_x)),
        ("durations", lambda ax: _plot_durations(ax, x, result, COLORS, xlabel, retirement_x)),
        ("expense_decomposition", lambda ax: _plot_expense_decomposition(ax, x, result, COLORS, xlabel, retirement_x)),
    ]

    # Beta-dependent panels: save for each beta value
    beta_dependent_panels = [
        ("hc_vs_fw", lambda ax: _plot_hc_vs_fw(ax, x, result, COLORS, xlabel, retirement_x)),
        ("hc_decomposition", lambda ax: _plot_hc_decomposition(ax, x, result, COLORS, xlabel, retirement_x)),
        ("net_hc_minus_expenses", lambda ax: _plot_net_hc_minus_expenses(ax, x, result, COLORS, xlabel, retirement_x)),
        ("consumption_path", lambda ax: _plot_consumption_path(ax, x, result, COLORS, xlabel, retirement_x)),
        ("portfolio_allocation", lambda ax: _plot_portfolio_allocation(ax, x, result, COLORS, xlabel, retirement_x)),
    ]

    # Save beta-invariant panels only on first call
    if is_first_beta:
        for name_suffix, plot_fn in beta_invariant_panels:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_fn(ax)
            panel_name = f"lifecycle_{name_suffix}"
            path = save_panel_as_png(fig, panel_name, output_dir)
            saved_paths.append(path)
            plt.close(fig)

    # Always save beta-dependent panels with beta suffix
    for name_suffix, plot_fn in beta_dependent_panels:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_fn(ax)
        panel_name = f"lifecycle_{beta_str}_{name_suffix}"
        path = save_panel_as_png(fig, panel_name, output_dir)
        saved_paths.append(path)
        plt.close(fig)

    return saved_paths


def _plot_income_expenses(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot income and expenses panel."""
    ax.plot(x, result.earnings, color=COLORS['earnings'], linewidth=2, label='Earnings')
    ax.plot(x, result.expenses, color=COLORS['expenses'], linewidth=2, label='Expenses')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5, label='Retirement')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Income & Expenses ($k)')
    ax.legend(loc='upper right', fontsize=8)


def _plot_cash_flow(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot cash flow panel."""
    savings = result.earnings - result.expenses
    ax.fill_between(x, 0, savings, where=savings >= 0, alpha=0.7, color=COLORS['earnings'], label='Savings')
    ax.fill_between(x, 0, savings, where=savings < 0, alpha=0.7, color=COLORS['expenses'], label='Drawdown')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Cash Flow: Earnings - Expenses ($k)')
    ax.legend(loc='upper right', fontsize=8)


def _plot_present_values(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot present values panel."""
    ax.plot(x, result.pv_earnings, color=COLORS['earnings'], linewidth=2, label='PV Earnings')
    ax.plot(x, result.pv_expenses, color=COLORS['expenses'], linewidth=2, label='PV Expenses')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Present Values ($k)')
    ax.legend(loc='upper right', fontsize=8)


def _plot_durations(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot durations panel."""
    ax.plot(x, result.duration_earnings, color=COLORS['earnings'], linewidth=2, label='Duration (Earnings)')
    ax.plot(x, result.duration_expenses, color=COLORS['expenses'], linewidth=2, label='Duration (Expenses)')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Years')
    ax.set_title('Durations (years)')
    ax.legend(loc='upper right', fontsize=8)


def _plot_hc_vs_fw(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot human capital vs financial wealth panel."""
    ax.fill_between(x, 0, result.financial_wealth, alpha=0.7, color=COLORS['fw'], label='Financial Wealth')
    ax.fill_between(x, result.financial_wealth, result.financial_wealth + result.human_capital,
                   alpha=0.7, color=COLORS['hc'], label='Human Capital')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Human Capital vs Financial Wealth ($k)')
    ax.legend(loc='upper right', fontsize=8)


def _plot_hc_decomposition(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot human capital decomposition panel."""
    ax.plot(x, result.hc_cash_component, color=COLORS['cash'], linewidth=2, label='HC Cash')
    ax.plot(x, result.hc_bond_component, color=COLORS['bond'], linewidth=2, label='HC Bond')
    ax.plot(x, result.hc_stock_component, color=COLORS['stock'], linewidth=2, label='HC Stock')
    ax.plot(x, result.human_capital, color='black', linewidth=1.5, linestyle='--', label='Total HC')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Human Capital Decomposition ($k)')
    ax.legend(loc='upper right', fontsize=8)


def _plot_expense_decomposition(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot expense liability decomposition panel."""
    ax.plot(x, result.exp_cash_component, color=COLORS['cash'], linewidth=2, label='Expense Cash')
    ax.plot(x, result.exp_bond_component, color=COLORS['bond'], linewidth=2, label='Expense Bond')
    ax.plot(x, result.pv_expenses, color='black', linewidth=1.5, linestyle='--', label='Total Expenses')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Expense Liability Decomposition ($k)')
    ax.legend(loc='upper right', fontsize=8)


def _plot_net_hc_minus_expenses(ax, x, result, COLORS, xlabel, retirement_x):
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
    ax.set_title('Net HC minus Expenses ($k)')
    ax.legend(loc='upper right', fontsize=8)


def _plot_consumption_path(ax, x, result, COLORS, xlabel, retirement_x):
    """Plot consumption path panel."""
    ax.fill_between(x, 0, result.subsistence_consumption, alpha=0.7, color=COLORS['subsistence'], label='Subsistence')
    ax.fill_between(x, result.subsistence_consumption, result.total_consumption,
                   alpha=0.7, color=COLORS['variable'], label='Variable')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Consumption Path ($k)')
    ax.legend(loc='upper right', fontsize=8)


def _plot_portfolio_allocation(ax, x, result, COLORS, xlabel, retirement_x):
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
    ax.set_title('Portfolio Allocation (%)')
    ax.legend(loc='upper right', fontsize=8)


def _save_beta_comparison_panels(
    beta_values: list,
    base_params: LifecycleParams,
    econ_params: EconomicParams,
    use_years: bool = True,
    output_dir: str = "output/teaching_panels",
) -> list:
    """
    Create and save individual PNG panels for the beta comparison page.

    This extracts each of the 6 panels from the beta comparison layout.

    Returns:
        List of saved file paths
    """
    from core import compute_lifecycle_median_path

    saved_paths = []

    # Compute results for each beta value
    results = {}
    for beta in beta_values:
        params = LifecycleParams(
            start_age=base_params.start_age,
            retirement_age=base_params.retirement_age,
            end_age=base_params.end_age,
            initial_earnings=base_params.initial_earnings,
            earnings_growth=base_params.earnings_growth,
            earnings_hump_age=base_params.earnings_hump_age,
            earnings_decline=base_params.earnings_decline,
            base_expenses=base_params.base_expenses,
            expense_growth=base_params.expense_growth,
            retirement_expenses=base_params.retirement_expenses,
            stock_beta_human_capital=beta,
            gamma=base_params.gamma,
            target_stock_allocation=base_params.target_stock_allocation,
            target_bond_allocation=base_params.target_bond_allocation,
            risk_free_rate=base_params.risk_free_rate,
            equity_premium=base_params.equity_premium,
            initial_wealth=base_params.initial_wealth,
        )
        results[beta] = compute_lifecycle_median_path(params, econ_params)

    # Colors for different beta values
    beta_colors = ['#1A759F', '#E9C46A', '#2A9D8F']  # blue, amber, teal

    # Get x-axis values
    result_0 = results[beta_values[0]]
    if use_years:
        x = np.arange(len(result_0.ages))
        xlabel = 'Years from Career Start'
    else:
        x = result_0.ages
        xlabel = 'Age'

    retirement_x = base_params.retirement_age - base_params.start_age if use_years else base_params.retirement_age

    # Compute shared y-axis range for HC decomposition charts
    hc_min = float('inf')
    hc_max = float('-inf')
    for beta in beta_values:
        for component in [results[beta].hc_stock_component,
                          results[beta].hc_bond_component,
                          results[beta].hc_cash_component]:
            hc_min = min(hc_min, np.min(component))
            hc_max = max(hc_max, np.max(component))
    hc_range = hc_max - hc_min
    hc_ylim = (hc_min - 0.05 * hc_range, hc_max + 0.05 * hc_range)

    panel_configs = [
        ("stock_weight_by_beta", "stock_weight_no_short", "Stock Weight by Beta", "Weight", (-0.05, 1.15)),
        ("bond_weight_by_beta", "bond_weight_no_short", "Bond Weight by Beta", "Weight", (-0.05, 1.15)),
        ("cash_weight_by_beta", "cash_weight_no_short", "Cash Weight by Beta", "Weight", (-0.05, 1.15)),
        ("hc_stock_by_beta", "hc_stock_component", "Stock Component of Human Capital", "$ (000s)", hc_ylim),
        ("hc_bond_by_beta", "hc_bond_component", "Bond Component of Human Capital", "$ (000s)", hc_ylim),
        ("hc_cash_by_beta", "hc_cash_component", "Cash Component of Human Capital", "$ (000s)", hc_ylim),
    ]

    for name_suffix, attr_name, title, ylabel, ylim in panel_configs:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, beta in enumerate(beta_values):
            ax.plot(x, getattr(results[beta], attr_name), color=beta_colors[i],
                    linewidth=2, label=f'Beta = {beta}')
        ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        if ylim[1] <= 1.5:  # For weight charts
            ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(ylim)

        panel_name = f"lifecycle_beta_comparison_{name_suffix}"
        path = save_panel_as_png(fig, panel_name, output_dir)
        saved_paths.append(path)
        plt.close(fig)

    return saved_paths


def _save_allocation_comparison_panels(
    comparison_beta0,
    comparison_beta_risky,
    params_beta0: LifecycleParams,
    params_beta_risky: LifecycleParams,
    use_years: bool = True,
    output_dir: str = "output/teaching_panels",
) -> list:
    """
    Create and save individual PNG panels for the allocation comparison page.

    This extracts each of the 4 panels from the allocation comparison layout.

    Returns:
        List of saved file paths
    """
    saved_paths = []

    risky_beta = params_beta_risky.stock_beta_human_capital

    if use_years:
        x = np.arange(len(comparison_beta0.ages))
        xlabel = 'Years from Career Start'
        retirement_x = params_beta0.retirement_age - params_beta0.start_age
    else:
        x = comparison_beta0.ages
        xlabel = 'Age'
        retirement_x = params_beta0.retirement_age

    stock_color = '#F4A261'
    bond_color = '#9b59b6'
    cash_color = '#95a5a6'

    # Panel 1: LDI Allocation (Beta=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ldi_b0 = comparison_beta0.result_a
    ax.stackplot(x,
                 ldi_b0.stock_weight * 100,
                 ldi_b0.bond_weight * 100,
                 ldi_b0.cash_weight * 100,
                 labels=['Stocks', 'Bonds', 'Cash'],
                 colors=[stock_color, bond_color, cash_color],
                 alpha=0.8)
    ax.axvline(x=retirement_x, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Allocation (%)')
    ax.set_title('LDI Strategy (Beta = 0, Bond-like HC)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 100)
    path = save_panel_as_png(fig, "lifecycle_allocation_ldi_beta0", output_dir)
    saved_paths.append(path)
    plt.close(fig)

    # Panel 2: LDI Allocation (Beta=risky)
    fig, ax = plt.subplots(figsize=(10, 6))
    ldi_risky = comparison_beta_risky.result_a
    ax.stackplot(x,
                 ldi_risky.stock_weight * 100,
                 ldi_risky.bond_weight * 100,
                 ldi_risky.cash_weight * 100,
                 labels=['Stocks', 'Bonds', 'Cash'],
                 colors=[stock_color, bond_color, cash_color],
                 alpha=0.8)
    ax.axvline(x=retirement_x, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Allocation (%)')
    ax.set_title(f'LDI Strategy (Beta = {risky_beta}, Risky HC)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 100)
    path = save_panel_as_png(fig, f"lifecycle_allocation_ldi_beta{risky_beta}".replace(".", "p"), output_dir)
    saved_paths.append(path)
    plt.close(fig)

    # Panel 3: RoT Allocation (Beta=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    rot_b0 = comparison_beta0.result_b
    ax.stackplot(x,
                 rot_b0.stock_weight * 100,
                 rot_b0.bond_weight * 100,
                 rot_b0.cash_weight * 100,
                 labels=['Stocks', 'Bonds', 'Cash'],
                 colors=[stock_color, bond_color, cash_color],
                 alpha=0.8)
    ax.axvline(x=retirement_x, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Allocation (%)')
    ax.set_title('Rule-of-Thumb: (100-Age)% Stock')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 100)
    path = save_panel_as_png(fig, "lifecycle_allocation_rot", output_dir)
    saved_paths.append(path)
    plt.close(fig)

    # Panel 4: Summary text (skip - no meaningful visual for teaching)
    # Instead, create a comparison chart showing both strategies side by side
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, ldi_b0.stock_weight * 100, color='#1A759F', linewidth=2.5, label='LDI (Beta=0)')
    ax.plot(x, ldi_risky.stock_weight * 100, color='#2A9D8F', linewidth=2.5, linestyle='--', label=f'LDI (Beta={risky_beta})')
    ax.plot(x, rot_b0.stock_weight * 100, color='#E9C46A', linewidth=2.5, linestyle=':', label='Rule-of-Thumb')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.5, label='Retirement')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Stock Allocation (%)')
    ax.set_title('Stock Allocation Comparison: LDI vs Rule-of-Thumb')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 100)
    path = save_panel_as_png(fig, "lifecycle_allocation_stock_comparison", output_dir)
    saved_paths.append(path)
    plt.close(fig)

    return saved_paths


def generate_lifecycle_pdf(
    output_path: str = 'output/lifecycle_strategy.pdf',
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    include_legacy_pages: bool = False,
    use_years: bool = True,
    rot_savings_rate: float = 0.15,
    rot_target_duration: float = 6.0,
    rot_withdrawal_rate: float = 0.04,
    export_png: bool = True,
    png_output_dir: str = "output/teaching_panels",
) -> str:
    """
    Generate a PDF report showing lifecycle investment strategy.

    STRUCTURE:
    - Pages 1-3: Deterministic Median Path for Beta = 0.0, DEFAULT_RISKY_BETA, 1.0
    - Page 4: Effect of Stock Beta comparison
    - Page 5: Portfolio Allocation Comparison (LDI vs RoT for Beta=0 and Beta=risky)

    Args:
        output_path: Path for output PDF file
        params: Lifecycle parameters (uses defaults if None)
        econ_params: Economic parameters (uses defaults if None)
        include_legacy_pages: If True, include sensitivity and scenario pages
        use_years: If True, x-axis shows years from career start
        rot_savings_rate: Rule-of-Thumb savings rate (default 15%)
        rot_target_duration: Rule-of-Thumb FI target duration (default 6)
        rot_withdrawal_rate: Rule-of-Thumb withdrawal rate (default 4%)
        export_png: If True, export individual panels as PNG files (default True)
        png_output_dir: Directory for PNG files (default 'output/teaching_panels')

    Returns:
        Path to generated PDF file
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    beta_values = [0.0, DEFAULT_RISKY_BETA, 1.0]

    with PdfPages(output_path) as pdf:
        # ====================================================================
        # PAGES 1-3: DETERMINISTIC MEDIAN PATH for each Beta
        # ====================================================================
        for page_num, beta in enumerate(beta_values, start=1):
            print(f"Generating Page {page_num}: Deterministic Path (Beta = {beta})...")

            beta_params = LifecycleParams(
                start_age=params.start_age,
                retirement_age=params.retirement_age,
                end_age=params.end_age,
                initial_earnings=params.initial_earnings,
                earnings_growth=params.earnings_growth,
                earnings_hump_age=params.earnings_hump_age,
                earnings_decline=params.earnings_decline,
                base_expenses=params.base_expenses,
                expense_growth=params.expense_growth,
                retirement_expenses=params.retirement_expenses,
                stock_beta_human_capital=beta,
                gamma=params.gamma,
                target_stock_allocation=params.target_stock_allocation,
                target_bond_allocation=params.target_bond_allocation,
                risk_free_rate=params.risk_free_rate,
                equity_premium=params.equity_premium,
                initial_wealth=params.initial_wealth,
                consumption_boost=params.consumption_boost,
            )
            result = compute_lifecycle_median_path(beta_params, econ_params)

            fig = create_base_case_page(
                result=result,
                params=beta_params,
                econ_params=econ_params,
                figsize=(20, 24),
                use_years=use_years
            )
            fig.suptitle(f'PAGE {page_num}: DETERMINISTIC MEDIAN PATH (Beta = {beta})',
                        fontsize=16, fontweight='bold', y=0.995)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Export individual panels as PNG
            if export_png:
                _save_base_case_panels(
                    result=result,
                    params=beta_params,
                    beta=beta,
                    use_years=use_years,
                    output_dir=png_output_dir,
                    is_first_beta=(page_num == 1),
                )

        # ====================================================================
        # PAGE 4: BETA COMPARISON
        # ====================================================================
        print("Generating Page 4: Beta Comparison...")
        fig = create_beta_comparison_figure(
            beta_values=beta_values,
            base_params=params,
            econ_params=econ_params,
            figsize=(16, 10),
            use_years=use_years
        )
        fig.suptitle('Effect of Stock Beta on Portfolio Allocation & Human Capital',
                    fontsize=14, fontweight='bold', y=1.02)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Export individual beta comparison panels as PNG
        if export_png:
            _save_beta_comparison_panels(
                beta_values=beta_values,
                base_params=params,
                econ_params=econ_params,
                use_years=use_years,
                output_dir=png_output_dir,
            )

        # ====================================================================
        # PAGE 5: PORTFOLIO ALLOCATION COMPARISON (Beta = 0 vs Beta = DEFAULT_RISKY_BETA)
        # ====================================================================
        print("Generating Page 5: Portfolio Allocation Comparison...")

        # Beta = 0 comparison
        median_comparison_beta0 = compute_median_path_comparison(
            params=params,
            econ_params=econ_params,
            rot_savings_rate=rot_savings_rate,
            rot_target_duration=rot_target_duration,
            rot_withdrawal_rate=rot_withdrawal_rate,
        )

        # Beta = DEFAULT_RISKY_BETA comparison
        params_beta03 = LifecycleParams(
            start_age=params.start_age,
            retirement_age=params.retirement_age,
            end_age=params.end_age,
            initial_earnings=params.initial_earnings,
            earnings_growth=params.earnings_growth,
            earnings_hump_age=params.earnings_hump_age,
            earnings_decline=params.earnings_decline,
            base_expenses=params.base_expenses,
            expense_growth=params.expense_growth,
            retirement_expenses=params.retirement_expenses,
            stock_beta_human_capital=DEFAULT_RISKY_BETA,
            gamma=params.gamma,
            target_stock_allocation=params.target_stock_allocation,
            target_bond_allocation=params.target_bond_allocation,
            risk_free_rate=params.risk_free_rate,
            equity_premium=params.equity_premium,
            initial_wealth=params.initial_wealth,
            consumption_boost=params.consumption_boost,
        )
        median_comparison_beta03 = compute_median_path_comparison(
            params=params_beta03,
            econ_params=econ_params,
            rot_savings_rate=rot_savings_rate,
            rot_target_duration=rot_target_duration,
            rot_withdrawal_rate=rot_withdrawal_rate,
        )

        fig = create_allocation_comparison_page(
            comparison_beta0=median_comparison_beta0,
            comparison_beta_risky=median_comparison_beta03,
            params_beta0=params,
            params_beta_risky=params_beta03,
            figsize=(16, 10),
            use_years=use_years,
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Export individual allocation comparison panels as PNG
        if export_png:
            _save_allocation_comparison_panels(
                comparison_beta0=median_comparison_beta0,
                comparison_beta_risky=median_comparison_beta03,
                params_beta0=params,
                params_beta_risky=params_beta03,
                use_years=use_years,
                output_dir=png_output_dir,
            )

        # ====================================================================
        # LEGACY PAGES (optional)
        # ====================================================================
        if include_legacy_pages:
            print("Generating legacy comparison pages...")

            # Gamma comparison
            fig = create_gamma_comparison_figure(
                gamma_values=[1.0, 2.0, 4.0, 8.0],
                base_params=params,
                econ_params=econ_params,
                figsize=(16, 10),
                use_years=use_years
            )
            fig.suptitle('Effect of Risk Aversion on Lifecycle Strategy',
                        fontsize=14, fontweight='bold', y=1.02)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Initial Wealth comparison
            fig = create_initial_wealth_comparison_figure(
                wealth_values=[-50, 0, 50, 200],
                base_params=params,
                econ_params=econ_params,
                figsize=(16, 10),
                use_years=use_years
            )
            fig.suptitle('Effect of Initial Wealth on Lifecycle Strategy',
                        fontsize=14, fontweight='bold', y=1.02)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Equity Premium comparison
            fig = create_equity_premium_comparison_figure(
                premium_values=[0.02, 0.04, 0.06, 0.08],
                base_params=params,
                econ_params=econ_params,
                figsize=(16, 10),
                use_years=use_years
            )
            fig.suptitle('Effect of Equity Risk Premium on Lifecycle Strategy',
                        fontsize=14, fontweight='bold', y=1.02)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Volatility comparison
            fig = create_volatility_comparison_figure(
                volatility_values=[0.12, 0.18, 0.24, 0.30],
                base_params=params,
                econ_params=econ_params,
                figsize=(16, 10),
                use_years=use_years
            )
            fig.suptitle('Effect of Stock Volatility on Lifecycle Strategy',
                        fontsize=14, fontweight='bold', y=1.02)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Monte Carlo page
            print("Generating Monte Carlo page...")
            mc_params = MonteCarloParams(n_simulations=50, random_seed=42)
            mc_result = run_lifecycle_monte_carlo(params, econ_params, mc_params)
            fig = create_monte_carlo_page(
                mc_result=mc_result,
                params=params,
                econ_params=econ_params,
                figsize=(20, 22),
                use_years=use_years
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Teaching scenarios
            for scenario_type in ['normal', 'sequenceRisk', 'rateShock']:
                print(f"Generating {scenario_type} scenario page...")
                fig = create_scenario_page(
                    scenario_type=scenario_type,
                    params=params,
                    econ_params=econ_params,
                    figsize=(20, 18),
                    use_years=use_years,
                    n_simulations=50,
                    random_seed=42,
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        # Summary page with parameters
        _add_summary_page(pdf, params, econ_params)

    # Print PNG export summary
    if export_png:
        import glob
        png_files = glob.glob(os.path.join(png_output_dir, "lifecycle_*.png"))
        print(f"Exported {len(png_files)} PNG panels to {png_output_dir}/")

    return output_path


def _add_summary_page(pdf, params: LifecycleParams, econ_params: EconomicParams):
    """Add a summary page with parameter values to the PDF."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    if params.gamma > 0:
        mv_stock, mv_bond, mv_cash = compute_mv_optimal_allocation(
            mu_stock=econ_params.mu_excess,
            mu_bond=econ_params.mu_bond,
            sigma_s=econ_params.sigma_s,
            sigma_r=econ_params.sigma_r,
            rho=econ_params.rho,
            duration=econ_params.bond_duration,
            gamma=params.gamma
        )
        allocation_source = "Mean-Variance Optimization"
    else:
        mv_stock = params.target_stock_allocation
        mv_bond = params.target_bond_allocation
        mv_cash = 1 - mv_stock - mv_bond
        allocation_source = "Fixed Targets"

    summary_text = f"""
Lifecycle Investment Strategy Parameters
========================================

Age Parameters:
  - Career Start: {params.start_age}
  - Retirement Age: {params.retirement_age}
  - Planning Horizon: {params.end_age}

Income Parameters:
  - Initial Earnings: ${params.initial_earnings:,.0f}k
  - Earnings Growth: {params.earnings_growth*100:.1f}%

Expense Parameters:
  - Base Expenses: ${params.base_expenses:,.0f}k
  - Retirement Expenses: ${params.retirement_expenses:,.0f}k

Initial Wealth: ${params.initial_wealth:,.0f}k

Economic Parameters:
  - Risk-Free Rate: {econ_params.r_bar*100:.1f}%
  - Equity Premium: {econ_params.mu_excess*100:.1f}%
  - Stock Volatility: {econ_params.sigma_s*100:.0f}%
  - Risk Aversion (gamma): {params.gamma:.1f}

Human Capital:
  - Stock Beta: {params.stock_beta_human_capital:.2f}
  - Bond Duration: {econ_params.bond_duration:.1f} years

Target Allocation ({allocation_source}):
  - Stocks: {mv_stock*100:.1f}%
  - Bonds: {mv_bond*100:.1f}%
  - Cash: {mv_cash*100:.1f}%
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# Default instances for CLI - single source of truth
_DEFAULT_PARAMS = LifecycleParams()
_DEFAULT_ECON = EconomicParams()


def main(
    output_path: str = 'output/lifecycle_strategy.pdf',
    start_age: int = None,
    retirement_age: int = None,
    end_age: int = None,
    initial_earnings: float = None,
    stock_beta_hc: float = None,
    bond_duration: float = None,
    gamma: float = None,
    mu_excess: float = None,
    bond_sharpe: float = None,
    sigma_s: float = None,
    sigma_r: float = None,
    rho: float = None,
    r_bar: float = None,
    consumption_share: float = None,
    consumption_boost: float = None,
    initial_wealth: float = None,
    include_scenarios: bool = False,
    use_years: bool = True,
    verbose: bool = True,
    rot_savings_rate: float = 0.15,
    rot_target_duration: float = 6.0,
    rot_withdrawal_rate: float = 0.04,
):
    """
    Generate lifecycle strategy PDF with configurable parameters.

    Args:
        output_path: Path for output PDF file
        start_age: Age at career start
        retirement_age: Age at retirement
        end_age: Planning horizon end
        initial_earnings: Starting annual earnings in $000s
        stock_beta_hc: Beta of human capital to stocks
        bond_duration: Bond duration for MV optimization (years)
        gamma: Risk aversion coefficient (0 = use fixed targets)
        mu_excess: Equity risk premium
        bond_sharpe: Bond Sharpe ratio
        sigma_s: Stock return volatility
        sigma_r: Interest rate shock volatility
        rho: Correlation between rate and stock shocks
        r_bar: Long-run real risk-free rate
        consumption_share: Share of net worth consumed above subsistence
        consumption_boost: Boost above median return for consumption rate
        initial_wealth: Initial financial wealth in $000s
        include_scenarios: If True, include legacy scenario pages
        use_years: If True, x-axis shows years from start
        verbose: If True, print progress
        rot_savings_rate: Rule-of-Thumb savings rate
        rot_target_duration: Rule-of-Thumb target duration
        rot_withdrawal_rate: Rule-of-Thumb withdrawal rate
    """
    # Use dataclass defaults for any None values
    start_age = start_age if start_age is not None else _DEFAULT_PARAMS.start_age
    retirement_age = retirement_age if retirement_age is not None else _DEFAULT_PARAMS.retirement_age
    end_age = end_age if end_age is not None else _DEFAULT_PARAMS.end_age
    initial_earnings = initial_earnings if initial_earnings is not None else _DEFAULT_PARAMS.initial_earnings
    stock_beta_hc = stock_beta_hc if stock_beta_hc is not None else _DEFAULT_PARAMS.stock_beta_human_capital
    gamma = gamma if gamma is not None else _DEFAULT_PARAMS.gamma
    consumption_share = consumption_share if consumption_share is not None else _DEFAULT_PARAMS.consumption_share
    consumption_boost = consumption_boost if consumption_boost is not None else _DEFAULT_PARAMS.consumption_boost
    initial_wealth = initial_wealth if initial_wealth is not None else _DEFAULT_PARAMS.initial_wealth
    r_bar = r_bar if r_bar is not None else _DEFAULT_ECON.r_bar
    mu_excess = mu_excess if mu_excess is not None else _DEFAULT_ECON.mu_excess
    bond_sharpe = bond_sharpe if bond_sharpe is not None else _DEFAULT_ECON.bond_sharpe
    sigma_s = sigma_s if sigma_s is not None else _DEFAULT_ECON.sigma_s
    sigma_r = sigma_r if sigma_r is not None else _DEFAULT_ECON.sigma_r
    rho = rho if rho is not None else _DEFAULT_ECON.rho
    bond_duration = bond_duration if bond_duration is not None else _DEFAULT_ECON.bond_duration

    if verbose:
        print("Computing lifecycle investment strategy...")

    econ_params = EconomicParams(
        r_bar=r_bar,
        mu_excess=mu_excess,
        bond_sharpe=bond_sharpe,
        sigma_s=sigma_s,
        sigma_r=sigma_r,
        rho=rho,
        bond_duration=bond_duration,
    )

    if gamma > 0:
        opt_stock, opt_bond, opt_cash = compute_mv_optimal_allocation(
            mu_stock=mu_excess,
            mu_bond=econ_params.mu_bond,
            sigma_s=sigma_s,
            sigma_r=sigma_r,
            rho=rho,
            duration=bond_duration,
            gamma=gamma
        )
        if verbose:
            print(f"MV Optimal Allocation (gamma={gamma}): "
                  f"Stocks={opt_stock:.1%}, Bonds={opt_bond:.1%}, Cash={opt_cash:.1%}")
    else:
        opt_stock, opt_bond, opt_cash = 0.60, 0.30, 0.10

    params = LifecycleParams(
        start_age=start_age,
        retirement_age=retirement_age,
        end_age=end_age,
        initial_earnings=initial_earnings,
        stock_beta_human_capital=stock_beta_hc,
        gamma=gamma,
        target_stock_allocation=opt_stock,
        target_bond_allocation=opt_bond,
        consumption_share=consumption_share,
        consumption_boost=consumption_boost,
        initial_wealth=initial_wealth,
        risk_free_rate=r_bar,
        equity_premium=mu_excess,
    )

    output = generate_lifecycle_pdf(
        output_path=output_path,
        params=params,
        econ_params=econ_params,
        include_legacy_pages=include_scenarios,
        use_years=use_years,
        rot_savings_rate=rot_savings_rate,
        rot_target_duration=rot_target_duration,
        rot_withdrawal_rate=rot_withdrawal_rate,
    )

    if verbose:
        print(f"PDF generated: {output}")
        result = compute_lifecycle_median_path(params, econ_params)
        print("\nKey Statistics at Selected Ages:")
        print("-" * 70)
        print(f"{'Age':>5} {'Earnings':>12} {'Human Cap':>12} {'Fin Wealth':>12} {'Stock Wt':>10}")
        print("-" * 70)
        for age_idx in [0, 10, 20, 30, 39, 45, 55]:
            if age_idx < len(result.ages):
                age = result.ages[age_idx]
                print(f"{age:>5} {result.earnings[age_idx]:>12,.0f} {result.human_capital[age_idx]:>12,.0f} "
                      f"{result.financial_wealth[age_idx]:>12,.0f} {result.stock_weight_no_short[age_idx]:>10.1%}")

    return output


if __name__ == '__main__':
    import argparse

    # Use dataclass defaults for help text
    _p, _e = _DEFAULT_PARAMS, _DEFAULT_ECON

    parser = argparse.ArgumentParser(
        description='Generate lifecycle investment strategy PDF'
    )
    parser.add_argument('-o', '--output', default='output/lifecycle_strategy.pdf',
                       help='Output PDF file path')
    parser.add_argument('--start-age', type=int,
                       help=f'Age at career start (default: {_p.start_age})')
    parser.add_argument('--retirement-age', type=int,
                       help=f'Retirement age (default: {_p.retirement_age})')
    parser.add_argument('--end-age', type=int,
                       help=f'Planning horizon end (default: {_p.end_age})')
    parser.add_argument('--initial-earnings', type=float,
                       help=f'Initial earnings in $000s (default: {_p.initial_earnings})')
    parser.add_argument('--stock-beta', type=float,
                       help=f'Stock beta of human capital (default: {_p.stock_beta_human_capital})')
    parser.add_argument('--bond-duration', type=float,
                       help=f'Bond duration for MV optimization in years (default: {_e.bond_duration})')
    parser.add_argument('--gamma', type=float,
                       help=f'Risk aversion for MV optimization (default: {_p.gamma})')
    parser.add_argument('--mu-excess', type=float,
                       help=f'Equity risk premium (default: {_e.mu_excess})')
    parser.add_argument('--bond-sharpe', type=float,
                       help=f'Bond Sharpe ratio (default: {_e.bond_sharpe})')
    parser.add_argument('--sigma', type=float,
                       help=f'Stock return volatility (default: {_e.sigma_s})')
    parser.add_argument('--sigma-r', type=float,
                       help=f'Interest rate shock volatility (default: {_e.sigma_r})')
    parser.add_argument('--rho', type=float,
                       help=f'Correlation between rate and stock shocks (default: {_e.rho})')
    parser.add_argument('--r-bar', type=float,
                       help=f'Long-run real risk-free rate (default: {_e.r_bar})')
    parser.add_argument('--consumption-share', type=float,
                       help=f'Share of net worth consumed above subsistence (default: {_p.consumption_share})')
    parser.add_argument('--consumption-boost', type=float,
                       help=f'Boost above median return for consumption rate (default: {_p.consumption_boost})')
    parser.add_argument('--initial-wealth', type=float,
                       help=f'Initial financial wealth in $000s (default: {_p.initial_wealth})')
    parser.add_argument('--use-age', action='store_true',
                       help='Use age instead of years from start on x-axis')
    parser.add_argument('--include-legacy', action='store_true',
                       help='Include legacy sensitivity and scenario pages')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress output messages')
    parser.add_argument('--rot-savings-rate', type=float, default=0.15,
                       help='Rule-of-Thumb savings rate (default: 0.15)')
    parser.add_argument('--rot-target-duration', type=float, default=6.0,
                       help='Rule-of-Thumb target duration (default: 6)')
    parser.add_argument('--rot-withdrawal-rate', type=float, default=0.04,
                       help='Rule-of-Thumb withdrawal rate (default: 0.04)')

    args = parser.parse_args()

    main(
        output_path=args.output,
        start_age=args.start_age,
        retirement_age=args.retirement_age,
        end_age=args.end_age,
        initial_earnings=args.initial_earnings,
        stock_beta_hc=args.stock_beta,
        bond_duration=args.bond_duration,
        gamma=args.gamma,
        mu_excess=args.mu_excess,
        bond_sharpe=args.bond_sharpe,
        sigma_s=args.sigma,
        sigma_r=args.sigma_r,
        rho=args.rho,
        r_bar=args.r_bar,
        consumption_share=args.consumption_share,
        consumption_boost=args.consumption_boost,
        initial_wealth=args.initial_wealth,
        include_scenarios=args.include_legacy,
        use_years=not args.use_age,
        verbose=not args.quiet,
        rot_savings_rate=args.rot_savings_rate,
        rot_target_duration=args.rot_target_duration,
        rot_withdrawal_rate=args.rot_withdrawal_rate,
    )
