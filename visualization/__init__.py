"""
Visualization module for lifecycle investment analysis.

This module consolidates all matplotlib visualization code for the lifecycle
investment strategy project, providing a clean separation from business logic.

Submodules:
- styles: Color schemes, fonts, and style constants
- helpers: Common plotting utilities
- lifecycle_plots: Individual lifecycle charts (earnings, wealth, allocations)
- monte_carlo_plots: Fan charts, distributions, teaching scenarios
- comparison_plots: Strategy comparison visualizations
- sensitivity_plots: Parameter sensitivity analysis charts
"""

# Import styles and helpers
from .styles import (
    COLORS,
    STRATEGY_COLORS,
    apply_standard_style,
    get_percentile_line_styles,
)

from .helpers import (
    apply_wealth_log_scale,
    setup_figure,
    add_retirement_line,
    get_x_axis,
    add_zero_line,
    format_currency_axis,
    set_standard_xlim,
    add_legend,
    # New DRY helpers
    plot_fan_chart,
    plot_allocation_stack,
    setup_comparison_axes,
    compute_x_axis_from_ages,
    add_standard_chart_elements,
)

# Import lifecycle plots
from .lifecycle_plots import (
    plot_earnings_expenses_profile,
    plot_forward_present_values,
    plot_durations,
    plot_human_vs_financial_wealth,
    plot_hc_decomposition,
    plot_target_financial_holdings,
    plot_portfolio_shares,
    plot_total_wealth_holdings,
    plot_consumption_dollars,
    plot_consumption_breakdown,
    create_lifecycle_figure,
)

# Import Monte Carlo plots
from .monte_carlo_plots import (
    create_monte_carlo_fan_chart,
    create_monte_carlo_detailed_view,
    plot_wealth_paths_spaghetti,
    plot_final_wealth_distribution,
    create_teaching_scenarios_figure,
    create_sequence_of_returns_figure,
)

# Import comparison plots
from .comparison_plots import (
    create_optimal_vs_4pct_rule_comparison,
    create_strategy_comparison_figure,
    create_median_path_comparison_figure,
    create_allocation_comparison_page,
    plot_duration_matching_intuition,
    plot_strategy_comparison_bars,
)

# Import report page layouts
from .report_pages import (
    create_base_case_page,
    create_monte_carlo_page,
    create_scenario_page,
    REPORT_COLORS,
)

# Import sensitivity plots
from .sensitivity_plots import (
    create_beta_comparison_figure,
    create_gamma_comparison_figure,
    create_initial_wealth_comparison_figure,
    create_consumption_boost_comparison_figure,
    create_equity_premium_comparison_figure,
    create_income_comparison_figure,
    create_volatility_comparison_figure,
)

# Import human capital plots
from .human_capital_plots import (
    create_hc_stock_sensitivity_figure,
)

__all__ = [
    # Styles
    'COLORS',
    'STRATEGY_COLORS',
    'apply_standard_style',
    'get_percentile_line_styles',

    # Helpers
    'apply_wealth_log_scale',
    'setup_figure',
    'add_retirement_line',
    'get_x_axis',
    'add_zero_line',
    'format_currency_axis',
    'set_standard_xlim',
    'add_legend',
    # New DRY helpers
    'plot_fan_chart',
    'plot_allocation_stack',
    'setup_comparison_axes',
    'compute_x_axis_from_ages',
    'add_standard_chart_elements',

    # Lifecycle plots
    'plot_earnings_expenses_profile',
    'plot_forward_present_values',
    'plot_durations',
    'plot_human_vs_financial_wealth',
    'plot_hc_decomposition',
    'plot_target_financial_holdings',
    'plot_portfolio_shares',
    'plot_total_wealth_holdings',
    'plot_consumption_dollars',
    'plot_consumption_breakdown',
    'create_lifecycle_figure',

    # Monte Carlo plots
    'create_monte_carlo_fan_chart',
    'create_monte_carlo_detailed_view',
    'plot_wealth_paths_spaghetti',
    'plot_final_wealth_distribution',
    'create_teaching_scenarios_figure',
    'create_sequence_of_returns_figure',

    # Comparison plots
    'create_optimal_vs_4pct_rule_comparison',
    'create_strategy_comparison_figure',
    'create_median_path_comparison_figure',
    'create_allocation_comparison_page',
    'plot_duration_matching_intuition',
    'plot_strategy_comparison_bars',

    # Report pages
    'create_base_case_page',
    'create_monte_carlo_page',
    'create_scenario_page',
    'REPORT_COLORS',

    # Sensitivity plots
    'create_beta_comparison_figure',
    'create_gamma_comparison_figure',
    'create_initial_wealth_comparison_figure',
    'create_consumption_boost_comparison_figure',
    'create_equity_premium_comparison_figure',
    'create_income_comparison_figure',
    'create_volatility_comparison_figure',

    # Human capital plots
    'create_hc_stock_sensitivity_figure',
]
