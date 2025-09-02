#!/usr/bin/env python3
"""
Centralized plotting style configuration for apertus memorization analysis.
Ensures consistent styling across all plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_style():
    """
    Set the default plotting style for all memorization analysis plots.
    Uses Computer Modern font for publication-ready figures.
    """
    # Set matplotlib font to Computer Modern with proper mathtext
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman', 'cmr10']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.formatter.use_mathtext'] = True  # Use mathtext for better minus sign rendering
    plt.rcParams['text.usetex'] = False  # Don't require LaTeX installation
    
    # Figure settings
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    
    # Set seaborn style
    sns.set_style("white")
    

# Standard figure sizes
FIGURE_SIZES = {
    'single': (12, 8),
    'double': (24, 8),
    'triple': (40, 10),
    'square': (10, 10),
}

# Standard font sizes
FONT_SIZES = {
    'title': 24,
    'subtitle': 20,
    'axis_label': 26,
    'axis_label_bold': {'size': 26, 'weight': 'bold'},
    'tick': 20,
    'annotation': 18,
    'legend': 18,
    'colorbar_label': 24,
    'colorbar_label_bold': {'size': 24, 'weight': 'bold'},
}

# Standard color maps
COLORMAPS = {
    'memorization': 'YlOrRd',  # Yellow-Orange-Red for memorization intensity
    'difference': 'RdBu_r',    # Red-Blue diverging for differences
    'ttr': 'viridis',          # Viridis for TTR distributions
}

# Heatmap settings
HEATMAP_SETTINGS = {
    'linewidths': 0.5,
    'linecolor': 'lightgray',
    'annot': True,
    'fmt': '.3f',
    'square': False,
}


def apply_heatmap_style(ax, title=None, xlabel=None, ylabel=None):
    """
    Apply consistent styling to a heatmap axis.
    
    Args:
        ax: Matplotlib axis object
        title: Title for the heatmap
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    if title:
        ax.set_title(title, fontsize=FONT_SIZES['title'], pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZES['axis_label'], fontweight='bold')
    
    # Set tick label sizes
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=FONT_SIZES['tick'])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=FONT_SIZES['tick'])


def style_colorbar(fig, axes):
    """
    Apply consistent styling to all colorbars in a figure.
    
    Args:
        fig: Matplotlib figure object
        axes: Array of axes (to identify which are colorbars)
    """
    for ax in fig.axes:
        # Check if this is a colorbar axis
        if ax not in axes.flat if hasattr(axes, 'flat') else [axes]:
            # This is a colorbar
            ax.yaxis.label.set_fontsize(FONT_SIZES['colorbar_label'])
            ax.tick_params(axis='y', labelsize=FONT_SIZES['tick'])
            ax.yaxis.label.set_fontweight('bold')