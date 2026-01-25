"""
Lifecycle Investment Strategy Report Generation

This module orchestrates PDF report generation by combining core simulation
functions with visualization components. All computation and plotting logic
has been factored out to the core and visualization modules.

Author: FINC 450 Life Cycle Investing
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Core computation imports
from core import (
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
    run_strategy_comparison,
)

# Visualization imports
from visualization import (
    # Report page layouts
    create_base_case_page,
    create_monte_carlo_page,
    create_scenario_page,
    # Comparison plots
    create_strategy_comparison_figure,
    create_median_path_comparison_figure,
    # Sensitivity plots
    create_beta_comparison_figure,
    create_gamma_comparison_figure,
    create_initial_wealth_comparison_figure,
    create_equity_premium_comparison_figure,
    create_volatility_comparison_figure,
)


def generate_lifecycle_pdf(
    output_path: str = 'lifecycle_strategy.pdf',
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    include_legacy_pages: bool = False,
    use_years: bool = True,
    rot_savings_rate: float = 0.15,
    rot_target_duration: float = 6.0,
    rot_withdrawal_rate: float = 0.04,
) -> str:
    """
    Generate a PDF report showing lifecycle investment strategy.

    STRUCTURE:
    - Pages 1-3: Deterministic Median Path for Beta = 0.0, 0.5, 1.0
    - Page 4: Effect of Stock Beta comparison
    - Page 5: LDI vs Rule-of-Thumb Median Path Comparison
    - Page 6: Monte Carlo Strategy Comparison

    Args:
        output_path: Path for output PDF file
        params: Lifecycle parameters (uses defaults if None)
        econ_params: Economic parameters (uses defaults if None)
        include_legacy_pages: If True, include sensitivity and scenario pages
        use_years: If True, x-axis shows years from career start
        rot_savings_rate: Rule-of-Thumb savings rate (default 15%)
        rot_target_duration: Rule-of-Thumb FI target duration (default 6)
        rot_withdrawal_rate: Rule-of-Thumb withdrawal rate (default 4%)

    Returns:
        Path to generated PDF file
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    beta_values = [0.0, 0.5, 1.0]

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

        # ====================================================================
        # PAGE 5: LDI vs RULE-OF-THUMB MEDIAN PATH COMPARISON
        # ====================================================================
        print("Generating Page 5: LDI vs Rule-of-Thumb Median Path Comparison...")
        median_comparison = compute_median_path_comparison(
            params=params,
            econ_params=econ_params,
            rot_savings_rate=rot_savings_rate,
            rot_target_duration=rot_target_duration,
            rot_withdrawal_rate=rot_withdrawal_rate,
        )
        fig = create_median_path_comparison_figure(
            comparison_result=median_comparison,
            params=params,
            econ_params=econ_params,
            figsize=(18, 14),
            use_years=use_years,
        )
        fig.suptitle('PAGE 5: LDI vs Rule-of-Thumb (Deterministic Median Path)',
                    fontsize=16, fontweight='bold', y=1.02)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ====================================================================
        # PAGE 6: MONTE CARLO STRATEGY COMPARISON
        # ====================================================================
        print("Generating Page 6: Monte Carlo Strategy Comparison...")
        mc_comparison = run_strategy_comparison(
            params=params,
            econ_params=econ_params,
            n_simulations=100,
            random_seed=42,
            rot_savings_rate=rot_savings_rate,
            rot_target_duration=rot_target_duration,
            rot_withdrawal_rate=rot_withdrawal_rate,
        )
        fig = create_strategy_comparison_figure(
            comparison_result=mc_comparison,
            params=params,
            figsize=(18, 12),
            use_years=use_years,
        )
        fig.suptitle('PAGE 6: LDI vs Rule-of-Thumb (Monte Carlo Comparison)',
                    fontsize=16, fontweight='bold', y=1.02)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

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


def main(
    output_path: str = 'lifecycle_strategy.pdf',
    start_age: int = 25,
    retirement_age: int = 65,
    end_age: int = 85,
    initial_earnings: float = 150,
    stock_beta_hc: float = 0.0,
    bond_duration: float = 20.0,
    gamma: float = 2.0,
    mu_excess: float = 0.04,
    bond_sharpe: float = 0.0,
    sigma_s: float = 0.18,
    sigma_r: float = 0.006,
    rho: float = 0.0,
    r_bar: float = 0.02,
    consumption_share: float = 0.05,
    consumption_boost: float = 0.0,
    initial_wealth: float = 100.0,
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

    parser = argparse.ArgumentParser(
        description='Generate lifecycle investment strategy PDF'
    )
    parser.add_argument('-o', '--output', default='lifecycle_strategy.pdf',
                       help='Output PDF file path')
    parser.add_argument('--start-age', type=int, default=25,
                       help='Age at career start (default: 25)')
    parser.add_argument('--retirement-age', type=int, default=65,
                       help='Retirement age (default: 65)')
    parser.add_argument('--end-age', type=int, default=85,
                       help='Planning horizon end (default: 85)')
    parser.add_argument('--initial-earnings', type=float, default=150,
                       help='Initial earnings in $000s (default: 150)')
    parser.add_argument('--stock-beta', type=float, default=0.0,
                       help='Stock beta of human capital (default: 0.0)')
    parser.add_argument('--bond-duration', type=float, default=20.0,
                       help='Bond duration for MV optimization in years (default: 20)')
    parser.add_argument('--gamma', type=float, default=2.0,
                       help='Risk aversion for MV optimization (default: 2.0)')
    parser.add_argument('--mu-excess', type=float, default=0.04,
                       help='Equity risk premium (default: 0.04)')
    parser.add_argument('--bond-sharpe', type=float, default=0.0,
                       help='Bond Sharpe ratio (default: 0.0)')
    parser.add_argument('--sigma', type=float, default=0.18,
                       help='Stock return volatility (default: 0.18)')
    parser.add_argument('--sigma-r', type=float, default=0.006,
                       help='Interest rate shock volatility (default: 0.006)')
    parser.add_argument('--rho', type=float, default=0.0,
                       help='Correlation between rate and stock shocks (default: 0.0)')
    parser.add_argument('--r-bar', type=float, default=0.02,
                       help='Long-run real risk-free rate (default: 0.02)')
    parser.add_argument('--consumption-share', type=float, default=0.05,
                       help='Share of net worth consumed above subsistence (default: 0.05)')
    parser.add_argument('--consumption-boost', type=float, default=0.0,
                       help='Boost above median return for consumption rate (default: 0.0)')
    parser.add_argument('--initial-wealth', type=float, default=100,
                       help='Initial financial wealth in $000s (default: 100)')
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
