"""
Generate individual figure files for FINC450 Lifecycle Investing lecture.

These figures are designed to tell the story of lifecycle investing:
1. The lifecycle problem: income vs expenses mismatch
2. Human capital as an implicit asset
3. The "gauges" framework: what to track beyond the retirement account
4. Why portfolio allocation changes over life

Usage:
    python generate_lecture_figures.py [--output-dir figures/] [--format png]
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from lifecycle_strategy import (
    LifecycleParams,
    LifecycleResult,
    compute_lifecycle_median_path,
    compute_mv_optimal_allocation,
)
from retirement_simulation import EconomicParams


# Consistent style for all figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
})

# Color scheme
COLORS = {
    'earnings': '#27ae60',   # Green
    'expenses': '#e74c3c',   # Red
    'savings': '#27ae60',    # Green (same as earnings)
    'drawdown': '#e74c3c',   # Red (same as expenses)
    'hc': '#e67e22',         # Orange - Human Capital
    'fw': '#2ecc71',         # Light Green - Financial Wealth
    'tw': '#3498db',         # Blue - Total Wealth
    'stock': '#3498db',      # Blue
    'bond': '#9b59b6',       # Purple
    'cash': '#f1c40f',       # Yellow
    'subsistence': '#95a5a6', # Gray
    'variable': '#e74c3c',   # Red
    'pv_earnings': '#27ae60',
    'pv_expenses': '#e74c3c',
}


def setup_figure(figsize=(10, 6)):
    """Create a figure with consistent styling."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def add_retirement_line(ax, retirement_x, use_years=True):
    """Add a vertical line at retirement."""
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.7,
               label='Retirement', linewidth=1.5)


def get_x_axis(result, params, use_years=True):
    """Get x-axis values and labels."""
    if use_years:
        x = np.arange(len(result.ages))
        xlabel = 'Years from Career Start'
        retirement_x = params.retirement_age - params.start_age
    else:
        x = result.ages
        xlabel = 'Age'
        retirement_x = params.retirement_age
    return x, xlabel, retirement_x


# =============================================================================
# FIGURE 1: The Lifecycle Problem - Income vs Expenses
# =============================================================================

def fig_income_expenses(result: LifecycleResult, params: LifecycleParams,
                        use_years: bool = True, figsize=(10, 6)) -> plt.Figure:
    """
    The fundamental lifecycle problem: income and expenses don't match over time.

    Key insight: You earn during working years but need to consume throughout life.
    This mismatch is THE problem that lifecycle finance solves.
    """
    fig, ax = setup_figure(figsize)
    x, xlabel, retirement_x = get_x_axis(result, params, use_years)

    # Plot income and expenses on same chart
    ax.plot(x, result.earnings, color=COLORS['earnings'], linewidth=2.5,
            label='Labor Income')
    ax.plot(x, result.expenses, color=COLORS['expenses'], linewidth=2.5,
            label='Subsistence Expenses')

    # Fill the gap to show the problem
    ax.fill_between(x, result.expenses, result.earnings,
                    where=result.earnings >= result.expenses,
                    alpha=0.2, color=COLORS['savings'], label='Surplus (save)')
    ax.fill_between(x, result.expenses, result.earnings,
                    where=result.earnings < result.expenses,
                    alpha=0.2, color=COLORS['drawdown'], label='Deficit (draw)')

    add_retirement_line(ax, retirement_x, use_years)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s per year)')
    ax.set_title('The Lifecycle Problem: Income vs. Expenses')
    ax.legend(loc='upper right')
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 2: Cash Flow Over Life
# =============================================================================

def fig_cash_flow(result: LifecycleResult, params: LifecycleParams,
                  use_years: bool = True, figsize=(10, 6)) -> plt.Figure:
    """
    Net cash flow: savings during working years, drawdown in retirement.

    Key insight: The pattern of cash flows determines the investment problem.
    """
    fig, ax = setup_figure(figsize)
    x, xlabel, retirement_x = get_x_axis(result, params, use_years)

    cash_flow = result.earnings - result.expenses

    ax.fill_between(x, 0, cash_flow, where=cash_flow >= 0,
                    alpha=0.7, color=COLORS['savings'], label='Savings')
    ax.fill_between(x, 0, cash_flow, where=cash_flow < 0,
                    alpha=0.7, color=COLORS['drawdown'], label='Drawdown')
    ax.plot(x, cash_flow, color='black', linewidth=1.5)

    add_retirement_line(ax, retirement_x, use_years)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s per year)')
    ax.set_title('Net Cash Flow: Earnings minus Expenses')
    ax.legend(loc='upper right')
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 3: Present Values - The Forward-Looking View
# =============================================================================

def fig_present_values(result: LifecycleResult, params: LifecycleParams,
                       use_years: bool = True, figsize=(10, 6)) -> plt.Figure:
    """
    Present values of future earnings and expenses.

    Key insight: Finance thinks in present values. Human capital is the PV of
    future earnings - it's an asset you own but can't see in your brokerage account.
    """
    fig, ax = setup_figure(figsize)
    x, xlabel, retirement_x = get_x_axis(result, params, use_years)

    ax.plot(x, result.pv_earnings, color=COLORS['pv_earnings'], linewidth=2.5,
            label='PV(Future Earnings) = Human Capital')
    ax.plot(x, result.pv_expenses, color=COLORS['pv_expenses'], linewidth=2.5,
            label='PV(Future Expenses) = Liability')

    # Net human capital
    net_hc = result.pv_earnings - result.pv_expenses
    ax.plot(x, net_hc, color=COLORS['tw'], linewidth=2, linestyle='--',
            label='Net Human Capital')

    add_retirement_line(ax, retirement_x, use_years)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Present Values: Your Hidden Balance Sheet')
    ax.legend(loc='upper right')
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 4: Human Capital vs Financial Wealth
# =============================================================================

def fig_wealth_composition(result: LifecycleResult, params: LifecycleParams,
                           use_years: bool = True, figsize=(10, 6)) -> plt.Figure:
    """
    Total wealth = Human Capital + Financial Wealth.

    Key insight: Early in life, most of your wealth is human capital.
    This is WHY young people should hold more stocks in their financial portfolio.
    """
    fig, ax = setup_figure(figsize)
    x, xlabel, retirement_x = get_x_axis(result, params, use_years)

    # Stacked area showing composition
    ax.fill_between(x, 0, result.financial_wealth,
                    alpha=0.8, color=COLORS['fw'], label='Financial Wealth')
    ax.fill_between(x, result.financial_wealth,
                    result.financial_wealth + result.human_capital,
                    alpha=0.8, color=COLORS['hc'], label='Human Capital')

    # Total wealth line
    ax.plot(x, result.total_wealth, color='black', linewidth=2,
            linestyle='--', label='Total Wealth')

    add_retirement_line(ax, retirement_x, use_years)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Total Wealth = Human Capital + Financial Wealth')
    ax.legend(loc='upper right')
    ax.set_xlim(x[0] - 1, x[-1] + 1)
    ax.set_ylim(0, None)

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 5: The Gauges - What to Track
# =============================================================================

def fig_gauges_dashboard(result: LifecycleResult, params: LifecycleParams,
                         use_years: bool = True, figsize=(14, 10)) -> plt.Figure:
    """
    The "dashboard" of gauges to track for retirement planning.

    Key insight: It's not just about your 401(k) balance. You need to track:
    - Human Capital (your future earning power)
    - Financial Wealth (your savings)
    - Expense Liability (what you owe your future self)
    - Net Worth (assets minus liabilities)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    x, xlabel, retirement_x = get_x_axis(result, params, use_years)

    # Gauge 1: Human Capital
    ax = axes[0, 0]
    ax.fill_between(x, 0, result.human_capital, alpha=0.7, color=COLORS['hc'])
    ax.plot(x, result.human_capital, color=COLORS['hc'], linewidth=2)
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Gauge 1: Human Capital\n(PV of Future Earnings)', fontweight='bold')
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    # Gauge 2: Financial Wealth
    ax = axes[0, 1]
    ax.fill_between(x, 0, result.financial_wealth, alpha=0.7, color=COLORS['fw'])
    ax.plot(x, result.financial_wealth, color=COLORS['fw'], linewidth=2)
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Gauge 2: Financial Wealth\n(Your Savings/Investments)', fontweight='bold')
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    # Gauge 3: Expense Liability
    ax = axes[1, 0]
    ax.fill_between(x, 0, result.pv_expenses, alpha=0.7, color=COLORS['expenses'])
    ax.plot(x, result.pv_expenses, color=COLORS['expenses'], linewidth=2)
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Gauge 3: Expense Liability\n(PV of Future Spending Needs)', fontweight='bold')
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    # Gauge 4: Net Worth
    ax = axes[1, 1]
    net_worth = result.human_capital + result.financial_wealth - result.pv_expenses
    ax.fill_between(x, 0, net_worth, where=net_worth >= 0, alpha=0.7, color=COLORS['tw'])
    ax.fill_between(x, 0, net_worth, where=net_worth < 0, alpha=0.7, color=COLORS['expenses'])
    ax.plot(x, net_worth, color='black', linewidth=2)
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Gauge 4: Net Worth\n(HC + FW - Expenses)', fontweight='bold')
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    fig.suptitle('The Four Gauges of Lifecycle Finance', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 6: Why Allocation Changes - The Theory
# =============================================================================

def fig_allocation_theory(result: LifecycleResult, params: LifecycleParams,
                          use_years: bool = True, figsize=(12, 8)) -> plt.Figure:
    """
    The theoretical basis for changing allocation over life.

    Key insight: Human capital is like a bond (stable income stream).
    To maintain target risk, financial portfolio must adjust as HC depletes.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    x, xlabel, retirement_x = get_x_axis(result, params, use_years)

    # Panel 1: Human Capital as % of Total Wealth
    ax = axes[0, 0]
    hc_share = result.human_capital / np.maximum(result.total_wealth, 1) * 100
    fw_share = result.financial_wealth / np.maximum(result.total_wealth, 1) * 100

    ax.fill_between(x, 0, fw_share, alpha=0.8, color=COLORS['fw'], label='Financial Wealth %')
    ax.fill_between(x, fw_share, 100, alpha=0.8, color=COLORS['hc'], label='Human Capital %')
    ax.axvline(x=retirement_x, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('% of Total Wealth')
    ax.set_title('Wealth Composition Over Life')
    ax.legend(loc='right')
    ax.set_ylim(0, 100)
    ax.set_xlim(x[0], x[-1])

    # Panel 2: HC Decomposition (bond-like vs stock-like)
    ax = axes[0, 1]
    ax.plot(x, result.hc_bond_component + result.hc_cash_component,
            color=COLORS['bond'], linewidth=2, label='Bond-like HC')
    ax.plot(x, result.hc_stock_component,
            color=COLORS['stock'], linewidth=2, label='Stock-like HC')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title(f'Human Capital Decomposition\n(Beta = {params.stock_beta_human_capital})')
    ax.legend(loc='upper right')
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    # Panel 3: Financial Portfolio Allocation
    ax = axes[1, 0]
    ax.fill_between(x, 0, result.cash_weight_no_short * 100,
                    alpha=0.8, color=COLORS['cash'], label='Cash')
    ax.fill_between(x, result.cash_weight_no_short * 100,
                    (result.cash_weight_no_short + result.bond_weight_no_short) * 100,
                    alpha=0.8, color=COLORS['bond'], label='Bonds')
    ax.fill_between(x, (result.cash_weight_no_short + result.bond_weight_no_short) * 100, 100,
                    alpha=0.8, color=COLORS['stock'], label='Stocks')
    ax.axvline(x=retirement_x, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Portfolio Weight (%)')
    ax.set_title('Financial Portfolio Allocation')
    ax.legend(loc='right')
    ax.set_ylim(0, 100)
    ax.set_xlim(x[0], x[-1])

    # Panel 4: Stock allocation explanation
    ax = axes[1, 1]
    ax.plot(x, result.stock_weight_no_short * 100, color=COLORS['stock'],
            linewidth=2.5, label='Stock Weight in Financial Portfolio')
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Stock Allocation (%)')
    ax.set_title('The Equity Glide Path')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    # Add annotation
    ax.annotate('High stocks early:\nHC is bond-like',
                xy=(5, 95), fontsize=10, ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.annotate('Lower stocks later:\nHC depleted',
                xy=(50, 60), fontsize=10, ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle('Why Portfolio Allocation Changes Over Life', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 7: Duration Matching Intuition
# =============================================================================

def fig_duration_matching(result: LifecycleResult, params: LifecycleParams,
                          use_years: bool = True, figsize=(10, 6)) -> plt.Figure:
    """
    Duration of assets and liabilities.

    Key insight: Duration tells you interest rate sensitivity.
    Matching durations hedges interest rate risk.
    """
    fig, ax = setup_figure(figsize)
    x, xlabel, retirement_x = get_x_axis(result, params, use_years)

    ax.plot(x, result.duration_earnings, color=COLORS['pv_earnings'],
            linewidth=2.5, label='Duration of Future Earnings')
    ax.plot(x, result.duration_expenses, color=COLORS['pv_expenses'],
            linewidth=2.5, label='Duration of Future Expenses')

    add_retirement_line(ax, retirement_x, use_years)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Duration (years)')
    ax.set_title('Duration: Interest Rate Sensitivity of Your Balance Sheet')
    ax.legend(loc='upper right')
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 8: Consumption Path
# =============================================================================

def fig_consumption_path(result: LifecycleResult, params: LifecycleParams,
                         use_years: bool = True, figsize=(10, 6)) -> plt.Figure:
    """
    Optimal consumption over the lifecycle.

    Key insight: Consumption smoothing is optimal. Variable consumption
    adjusts to wealth, providing automatic adjustment to market conditions.
    """
    fig, ax = setup_figure(figsize)
    x, xlabel, retirement_x = get_x_axis(result, params, use_years)

    ax.fill_between(x, 0, result.subsistence_consumption,
                    alpha=0.7, color=COLORS['subsistence'], label='Subsistence')
    ax.fill_between(x, result.subsistence_consumption, result.total_consumption,
                    alpha=0.7, color=COLORS['variable'], label='Variable (wealth-based)')
    ax.plot(x, result.total_consumption, color='black', linewidth=1.5)

    add_retirement_line(ax, retirement_x, use_years)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s per year)')
    ax.set_title('Optimal Consumption: Subsistence + Variable')
    ax.legend(loc='upper right')
    ax.set_xlim(x[0] - 1, x[-1] + 1)
    ax.set_ylim(0, None)

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 9: Beta Comparison
# =============================================================================

def fig_beta_comparison(params: LifecycleParams, econ_params: EconomicParams,
                        use_years: bool = True, figsize=(12, 5)) -> plt.Figure:
    """
    How labor income risk (beta) affects portfolio allocation.

    Key insight: If your job is risky (high beta), hold LESS stock.
    Professor (beta=0) vs Tech entrepreneur (beta=1).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    betas = [0.0, 0.5, 1.0]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    labels = ['Beta=0 (Professor)', 'Beta=0.5 (Manager)', 'Beta=1.0 (Entrepreneur)']

    results = []
    for beta in betas:
        beta_params = LifecycleParams(
            start_age=params.start_age,
            retirement_age=params.retirement_age,
            end_age=params.end_age,
            initial_earnings=params.initial_earnings,
            stock_beta_human_capital=beta,
            gamma=params.gamma,
            initial_wealth=params.initial_wealth,
        )
        results.append(compute_lifecycle_median_path(beta_params, econ_params))

    x, xlabel, retirement_x = get_x_axis(results[0], params, use_years)

    # Panel 1: Stock allocation
    ax = axes[0]
    for i, (result, color, label) in enumerate(zip(results, colors, labels)):
        ax.plot(x, result.stock_weight_no_short * 100, color=color,
                linewidth=2.5, label=label)
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Stock Allocation (%)')
    ax.set_title('Stock Allocation by Labor Income Beta')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    # Panel 2: Human capital
    ax = axes[1]
    for i, (result, color, label) in enumerate(zip(results, colors, labels)):
        ax.plot(x, result.human_capital, color=color, linewidth=2.5, label=label)
    ax.axvline(x=retirement_x, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$ (000s)')
    ax.set_title('Human Capital by Labor Income Beta')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    fig.suptitle('Effect of Labor Income Risk on Portfolio Choice', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# Main: Generate All Figures
# =============================================================================

def generate_all_figures(
    output_dir: str = 'figures',
    format: str = 'png',
    dpi: int = 150,
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    use_years: bool = True,
):
    """
    Generate all lecture figures and save to individual files.

    Args:
        output_dir: Directory to save figures
        format: File format ('png', 'pdf', 'svg')
        dpi: Resolution for raster formats
        params: Lifecycle parameters
        econ_params: Economic parameters
        use_years: If True, x-axis shows years from career start
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Compute base case result
    print("Computing lifecycle solution...")
    result = compute_lifecycle_median_path(params, econ_params)

    # Define all figures to generate
    figures = [
        ('01_income_expenses', fig_income_expenses,
         "The lifecycle problem: income vs expenses"),
        ('02_cash_flow', fig_cash_flow,
         "Net cash flow over life"),
        ('03_present_values', fig_present_values,
         "Present values: human capital and expense liability"),
        ('04_wealth_composition', fig_wealth_composition,
         "Total wealth = Human Capital + Financial Wealth"),
        ('05_gauges_dashboard', fig_gauges_dashboard,
         "The four gauges of lifecycle finance"),
        ('06_allocation_theory', fig_allocation_theory,
         "Why portfolio allocation changes over life"),
        ('07_duration_matching', fig_duration_matching,
         "Duration: interest rate sensitivity"),
        ('08_consumption_path', fig_consumption_path,
         "Optimal consumption path"),
    ]

    # Generate each figure
    print(f"\nGenerating {len(figures)} figures...")
    for name, func, description in figures:
        print(f"  {name}: {description}")
        fig = func(result, params, use_years=use_years)
        filepath = output_path / f"{name}.{format}"
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    # Generate beta comparison (needs special handling)
    print("  09_beta_comparison: Effect of labor income risk")
    fig = fig_beta_comparison(params, econ_params, use_years=use_years)
    filepath = output_path / f"09_beta_comparison.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\nAll figures saved to: {output_path.absolute()}")
    print(f"Format: {format}, DPI: {dpi}")

    # Print suggested Beamer usage
    print("\n" + "="*60)
    print("BEAMER USAGE:")
    print("="*60)
    print(r"""
\begin{frame}{The Lifecycle Problem}
  \includegraphics[width=\textwidth]{figures/01_income_expenses.png}
\end{frame}

\begin{frame}{The Four Gauges of Lifecycle Finance}
  \includegraphics[width=\textwidth]{figures/05_gauges_dashboard.png}
\end{frame}

\begin{frame}{Why Allocation Changes Over Life}
  \includegraphics[width=\textwidth]{figures/06_allocation_theory.png}
\end{frame}
""")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate lecture figures for lifecycle investing'
    )
    parser.add_argument('--output-dir', default='figures',
                        help='Output directory for figures')
    parser.add_argument('--format', default='png', choices=['png', 'pdf', 'svg'],
                        help='Output format')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for raster formats')
    parser.add_argument('--use-age', action='store_true',
                        help='Use age instead of years on x-axis')

    args = parser.parse_args()

    generate_all_figures(
        output_dir=args.output_dir,
        format=args.format,
        dpi=args.dpi,
        use_years=not args.use_age,
    )
