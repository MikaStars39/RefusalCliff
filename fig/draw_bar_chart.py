import torch
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Dict, Any
import json
import numpy as np

# Set matplotlib style to match programmer aesthetics
plt.style.use('default')  # Reset to default first
plt.rcParams.update({
    'font.family': 'monospace',
    'font.monospace': ['Consolas', 'DejaVu Sans Mono', 'Courier New', 'monospace'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': '#f8f8f8',  # Light gray background for plot area
    'figure.facecolor': 'white',  # White background for figure
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'grid.alpha': 0.4,
    'axes.edgecolor': '#cccccc',  # Light gray for axes
    'xtick.color': '#666666',  # Light gray for tick labels
    'ytick.color': '#666666',
    'text.color': '#333333',  # Dark gray for text
    'axes.labelcolor': '#333333',
    'text.usetex': False,
})

def plot_bar_chart(
    data_path: str,  # Path to JSON file containing data
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
    colors: Optional[List[str]] = None,
    bar_width: float = 0.6,
    show_values: bool = True,
) -> None:
    """
    Plot bar chart with academic style matching the programmer aesthetic.
    
    Args:
        data_path: Path to JSON file containing bar chart data
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        colors: Optional list of colors for bars
        bar_width: Width of bars (0.0 to 1.0)
        show_values: Whether to show values on top of bars
    """
    # Load configuration from JSON file
    with open(data_path, "r") as f:
        config = json.load(f)
    
    labels = config["labels"]
    values = config["values"]
    title = config.get("title", "Bar Chart")
    xlabel = config.get("xlabel", "Categories")
    ylabel = config.get("ylabel", "Values")
    
    if not values or not labels:
        raise ValueError("Both values and labels must be provided")
    
    if len(values) != len(labels):
        raise ValueError("Values and labels must have the same length")
    
    # Academic color palette - sophisticated and distinguishable colors
    if colors is None:
        colors = [
            '#8B4513',  # Saddle Brown
            '#B22222',  # Fire Brick Red
            '#228B22',  # Forest Green
            '#4169E1',  # Royal Blue
            '#9932CC',  # Dark Orchid
            '#FF8C00',  # Dark Orange
            '#DC143C',  # Crimson
            '#2F4F4F',  # Dark Slate Gray
            '#8B008B',  # Dark Magenta
            '#556B2F'   # Dark Olive Green
        ]
    
    # Create figure with programmer styling
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')  # White background for figure
    ax.set_facecolor('#f8f8f8')  # Light gray background for plot area
    
    # Create bar positions
    x_pos = np.arange(len(labels))
    
    # Create bars with colors
    bars = ax.bar(x_pos, values, 
                  width=bar_width,
                  color=[colors[i % len(colors)] for i in range(len(values))],
                  alpha=0.8,
                  edgecolor='white',
                  linewidth=1.0)
    
    # Add value labels on top of bars if requested
    if show_values:
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                   f'{value:.2f}' if isinstance(value, float) else str(value),
                   ha='center', va='bottom',
                   fontsize=10, color='#333333',
                   fontweight='normal')
    
    # Configure plot with programmer styling
    ax.set_xlabel(xlabel, fontsize=16, fontweight='normal', color='#333333')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='normal', color='#333333')
    ax.set_title(title, fontsize=18, fontweight='normal', color='#333333', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45 if len(max(labels, key=len)) > 8 else 0, 
                       ha='right' if len(max(labels, key=len)) > 8 else 'center')
    
    # Set y-axis limits with some padding
    y_min = min(0, min(values) * 1.1 if min(values) < 0 else 0)
    y_max = max(values) * 1.1
    ax.set_ylim(y_min, y_max)
    
    # Customize ticks with light colors
    ax.tick_params(axis='both', which='major', labelsize=12, width=0.8, 
                   color='#cccccc', labelcolor='#666666')
    
    # Add grid with subtle styling
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='#dddddd', axis='y')
    
    # Set light colored spines
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
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
                       fontsize=9, color='#333333')
    
    # Configure plot
    ax.set_xlabel(xlabel, fontsize=16, fontweight='normal', color='#333333')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='normal', color='#333333')
    ax.set_title(title, fontsize=18, fontweight='normal', color='#333333', pad=20)
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
    ax.tick_params(axis='both', which='major', labelsize=12, width=0.8,
                   color='#cccccc', labelcolor='#666666')
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='#dddddd', axis='y')
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Grouped bar chart saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Example usage - create sample data files and plots
    
    # Sample simple bar chart data
    simple_data = {
        "labels": ["Method A", "Method B", "Method C", "Method D", "Method E"],
        "values": [0.85, 0.92, 0.78, 0.89, 0.94],
        "title": "Model Performance Comparison",
        "xlabel": "Methods",
        "ylabel": "Accuracy"
    }
    
    # Sample grouped bar chart data
    grouped_data = {
        "categories": ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4"],
        "groups": {
            "Model A": [0.85, 0.78, 0.92, 0.88],
            "Model B": [0.89, 0.82, 0.94, 0.91],
            "Model C": [0.82, 0.85, 0.89, 0.86]
        },
        "title": "Multi-Model Performance Comparison",
        "xlabel": "Datasets",
        "ylabel": "Performance Score"
    }
    
    # Create sample data files
    os.makedirs("outputs/fig", exist_ok=True)
    
    with open("outputs/fig/bar_data.json", "w") as f:
        json.dump(simple_data, f, indent=2)
    
    with open("outputs/fig/grouped_bar_data.json", "w") as f:
        json.dump(grouped_data, f, indent=2)
    
    # Plot examples
    plot_bar_chart(
        data_path="outputs/fig/bar_data.json",
        save_path="outputs/fig/simple_bar_chart.png",
        figsize=(8, 6)
    )
    
    plot_grouped_bar_chart(
        data_path="outputs/fig/grouped_bar_data.json",
        save_path="outputs/fig/grouped_bar_chart.png",
        figsize=(10, 6)
    ) 