"""
Centralized style definitions for lifecycle visualization.

This module provides consistent colors, fonts, and styles across all plots.
"""

import matplotlib.pyplot as plt

# Set consistent style for all figures
plt.style.use('seaborn-v0_8-whitegrid')

# Main color scheme
COLORS = {
    # Primary colors
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',

    # Semantic colors
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

    # Strategy comparison colors
    'optimal': '#2ecc71',    # Green
    'rot': '#3498db',        # Blue (Rule of Thumb)
    'color_4pct': '#e74c3c', # Red

    # Rate/return colors
    'rate': '#f39c12',
    'nw': '#9b59b6',
    'consumption': '#e74c3c',
}

# Strategy comparison color list
STRATEGY_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def apply_standard_style():
    """Apply standard matplotlib style settings."""
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
    })


def get_percentile_line_styles():
    """Get line styles for percentile charts."""
    return {
        'styles': {0: ':', 1: '--', 2: '-', 3: '--', 4: ':'},  # 5th, 25th, 50th, 75th, 95th
        'widths': {0: 1, 1: 1, 2: 2.5, 3: 1, 4: 1},
    }
