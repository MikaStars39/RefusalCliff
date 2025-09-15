import torch
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Dict, Any
import json
import numpy as np

# Set matplotlib style to use scienceplots retro
import scienceplots
plt.style.use(['science', 'no-latex', 'retro'])

# Override specific settings to maintain our preferences
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 7,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'axes.facecolor': '#f8f8f8',  # Light gray background for plot area
    'figure.facecolor': 'white',  # White background for figure
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'axes.spines.top': True,  # Show top spine
    'axes.spines.right': True,  # Show right spine
    'xtick.direction': 'out',  # Ticks point outward
    'ytick.direction': 'out',  # Ticks point outward
    'xtick.minor.visible': False,  # Hide minor x ticks
    'ytick.minor.visible': False,  # Hide minor y ticks
})

def plot_bar_chart_from_dict(
    data: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Bar Chart",
    xlabel: str = "Models",
    ylabel: str = "Values",
    figsize: tuple = (8, 6),
    colors: Optional[List[str]] = None,
    bar_width: float = 0.3,  # Keep bars thin
    show_values: bool = True,
) -> None:
    """
    Plot bar chart directly from a dictionary with academic style.
    
    Args:
        data: Dictionary with labels as keys and values as values
        save_path: Optional path to save the plot
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size tuple
        colors: Optional list of colors for bars
        bar_width: Width of bars (0.0 to 1.0)
        show_values: Whether to show values on top of bars
    """
    labels = list(data.keys())
    values = list(data.values())
    
    if not values or not labels:
        raise ValueError("Both values and labels must be provided")
    
    # Use scienceplots retro color palette
    if colors is None:
        # Get retro colors from matplotlib's current color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Create figure with programmer styling
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')  # White background for figure
    ax.set_facecolor('#f8f8f8')  # Light gray background for plot area
    
    # Add grid first so it appears behind bars
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#dddddd', axis='y', zorder=0)
    
    # Create bar positions with reduced spacing
    x_pos = np.arange(len(labels)) * 0.5 # Reduce spacing between bars
    
    # Create color list with gradient and special color for last bar
    bar_colors = []
    for i in range(len(values)):
        if i == len(values) - 1:  # Last bar is dark gray
            bar_colors.append('#404040')  # Dark gray
        else:
            # Use colors from the palette in order
            bar_colors.append(colors[i % len(colors)])
    
    # Create standard rectangular bars
    bars = ax.bar(x_pos, values, 
                  width=bar_width,
                  color=bar_colors,
                  alpha=0.85,
                  edgecolor='none',
                  zorder=3)
    
    # Add value labels on top of bars if requested
    if show_values:
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value*100:.1f}%',  # Convert to percentage format
                   ha='center', va='bottom',
                   fontsize=8, color='#333333',
                   fontweight='normal',
                   zorder=5)
    
    # Configure plot with programmer styling
    ax.set_title(title, fontsize=12, fontweight='bold', color='#333333', pad=10)
    ax.set_ylabel("Attack Successful Rate", fontsize=10, fontweight='normal', color='#333333')
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20 if len(max(labels, key=len)) > 8 else 0, 
                       ha='right' if len(max(labels, key=len)) > 8 else 'center')  # Font size controlled by rcParams
    
    # Set y-axis limits to 0-100% (0.0-1.0 in decimal)
    ax.set_ylim(0, 1.0)
    
    # Format y-axis as percentages
    from matplotlib.ticker import FuncFormatter
    def percent_formatter(x, pos):
        return f'{x*100:.0f}%'
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    
    # Customize ticks with light colors
    ax.tick_params(axis='x', which='major', labelsize=7, width=0.8, 
                   color='#cccccc', labelcolor='#666666')
    ax.tick_params(axis='y', which='major', labelsize=10, width=0.8, 
                   color='#cccccc', labelcolor='#666666')
    
    # Set light colored spines for all four sides
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['top'].set_color('#cccccc')
    ax.spines['right'].set_color('#cccccc')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['top'].set_linewidth(0.8)
    ax.spines['right'].set_linewidth(0.8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        # Ensure PDF extension
        if save_path and not save_path.lower().endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', format='pdf')
        print(f"Bar chart saved to: {save_path}")
    
    plt.show()

def plot_grouped_bar_chart(
    data_path: str,  # Path to JSON file containing grouped data
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    colors: Optional[List[str]] = None,
    bar_width: float = 0.35,
    show_values: bool = True,
) -> None:
    """
    Plot grouped bar chart with academic style.
    
    Args:
        data_path: Path to JSON file containing grouped bar chart data
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        colors: Optional list of colors for bar groups
        bar_width: Width of individual bars
        show_values: Whether to show values on top of bars
    """
    # Load configuration from JSON file
    with open(data_path, "r") as f:
        config = json.load(f)
    
    categories = config["categories"]  # x-axis categories
    groups = config["groups"]  # dictionary with group_name: values
    title = config.get("title", "Grouped Bar Chart")
    xlabel = config.get("xlabel", "Categories")
    ylabel = config.get("ylabel", "Values")
    
    if not groups or not categories:
        raise ValueError("Both groups and categories must be provided")
    
    # Academic color palette
    if colors is None:
        colors = [
            '#8B4513', '#B22222', '#228B22', '#4169E1', '#9932CC',
            '#FF8C00', '#DC143C', '#2F4F4F', '#8B008B', '#556B2F'
        ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f8f8')
    
    # Calculate positions
    x = np.arange(len(categories))
    group_names = list(groups.keys())
    n_groups = len(group_names)
    
    # Plot bars for each group
    for i, group_name in enumerate(group_names):
        values = groups[group_name]
        offset = (i - n_groups/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width,
                     label=group_name,
                     color=colors[i % len(colors)],
                     alpha=0.8,
                     edgecolor='white',
                     linewidth=1.0)
        
        # Add value labels if requested
        if show_values:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(max(groups.values())) * 0.01,
                       f'{value:.2f}' if isinstance(value, float) else str(value),
                       ha='center', va='bottom',
                       fontsize=7, color='#333333')
    
    # Configure plot
    ax.set_xlabel(xlabel, fontsize=10, fontweight='normal', color='#333333')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='normal', color='#333333')
    ax.set_title(title, fontsize=12, fontweight='bold', color='#333333', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    
    # Configure legend
    legend = ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor='#dddddd',
        facecolor='#f8f8f8',
        framealpha=0.9,
        fontsize=10,
        loc='upper right'
    )
    legend.get_frame().set_linewidth(0.8)
    
    # Styling
    ax.tick_params(axis='x', which='major', labelsize=7, width=0.8,
                   color='#cccccc', labelcolor='#666666')
    ax.tick_params(axis='y', which='major', labelsize=10, width=0.8,
                   color='#cccccc', labelcolor='#666666')
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='#dddddd', axis='y')
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['top'].set_color('#cccccc')
    ax.spines['right'].set_color('#cccccc')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['top'].set_linewidth(0.8)
    ax.spines['right'].set_linewidth(0.8)
    
    plt.tight_layout()
    
    if save_path:
        # Ensure PDF extension
        if save_path and not save_path.lower().endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none', format='pdf')
        print(f"Grouped bar chart saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Data to plot
    data = {
        "R1-LLaMA-8B": 0.672,
        "R1-Qwen-7B": 0.525,
        "Hermes-14B": 0.500,
        "Skywork-OR1-7B": 0.425,
        "R1-Qwen-14B": 0.225,
        "QwQ-32B": 0.21,
        "Qwen3-4B": 0.025,
        "Qwen3-30B": 0.025,
        "Phi4": 0.015,
        "LLaMA-8B-it": 0.02,
        "Qwen2.5-7B-it": 0.02,
    }
    
    # Create output directory
    os.makedirs("outputs/fig", exist_ok=True)
    
    # Plot the bar chart using the data dictionary
    plot_bar_chart_from_dict(
        data=data,
        save_path="outputs/fig/model_comparison.pdf",
        figsize=(6, 4),
        title="ASR on Advbench",
    ) 