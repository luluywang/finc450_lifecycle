"""
Centralized style definitions for lifecycle visualization.

This module provides consistent colors, fonts, and styles across all plots.
"""

import matplotlib.pyplot as plt

# Set consistent style for all figures
plt.style.use('seaborn-v0_8-whitegrid')

# Main color scheme (colorblind-friendly: blue-orange palette)
COLORS = {
    # Primary colors (colorblind-safe)
    'blue': '#1A759F',
    'orange': '#E07A5F',
    'teal': '#2A9D8F',
    'amber': '#E9C46A',

    # Legacy aliases for backward compatibility (map to colorblind-safe colors)
    'green': '#2A9D8F',  # Maps to teal (colorblind-safe alternative to green)
    'red': '#E07A5F',    # Maps to burnt orange (colorblind-safe alternative to red)

    # Semantic colors - Income/Expense flows (blue=positive, orange=negative)
    'earnings': '#0077B6',   # Teal-blue (was green)
    'expenses': '#E07A5F',   # Burnt orange (was red)
    'savings': '#0077B6',    # Match earnings
    'drawdown': '#E07A5F',   # Match expenses
    'hc': '#e67e22',         # Orange - Human Capital (unchanged)
    'fw': '#457B9D',         # Blue - Financial Wealth (was green)
    'tw': '#1D3557',         # Dark blue - Total Wealth
    'stock': '#F4A261',      # Coral (was blue/red)
    'bond': '#9b59b6',       # Purple (unchanged)
    'cash': '#f1c40f',       # Yellow (unchanged)
    'subsistence': '#95a5a6', # Gray (unchanged)
    'variable': '#2A9D8F',   # Teal (was red)
    'pv_earnings': '#0077B6',
    'pv_expenses': '#E07A5F',

    # Strategy comparison colors
    'optimal': '#1A759F',    # Deep blue (was green)
    'rot': '#E9C46A',        # Amber (was blue)
    'color_4pct': '#E9C46A', # Amber (was red)

    # Rate/return colors
    'rate': '#f39c12',
    'nw': '#9b59b6',
    'consumption': '#2A9D8F',

    # Market conditions
    'bull': '#264653',       # Deep teal (was green)
    'bear': '#BC6C25',       # Rust (was red)
}

# Strategy comparison color list (colorblind-safe)
STRATEGY_COLORS = ['#1A759F', '#E9C46A', '#2A9D8F', '#BC6C25']


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
