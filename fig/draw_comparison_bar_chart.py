import torch
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Dict, Any, Tuple
import json
import numpy as np

# Set matplotlib style to use scienceplots retro
import scienceplots
plt.style.use(['science', 'no-latex', 'retro'])

# Override specific settings to maintain our preferences
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Georgia', 'Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'axes.facecolor': '#f8f8f8',  # Light gray background for plot area
    'figure.facecolor': 'white',  # White background for figure
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'axes.spines.top': False,  # Hide top spine
    'axes.spines.right': False,  # Hide right spine
    'xtick.direction': 'out',  # Ticks point outward
    'ytick.direction': 'out',  # Ticks point outward
    'xtick.minor.visible': False,  # Hide minor x ticks
    'ytick.minor.visible': False,  # Hide minor y ticks
})

def plot_comparison_bar_chart(
    data: Dict[str, Tuple[float, ...]],  # model_name: (value1, value2, value3, ...)
    save_path: Optional[str] = None,
    title: str = "Model Performance Comparison",
    xlabel: str = "",
    ylabel: str = "Values",
    figsize: tuple = (10, 6),
    bar_width: float = 0.15,
    group_spacing: float = 0.1,
    legend_labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
) -> None:
    """
    Plot comparison bar chart with multiple bars per model grouped together.
    
    Args:
        data: Dictionary with model names as keys and tuples of values
        save_path: Optional path to save the plot
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size tuple
        bar_width: Width of individual bars
        group_spacing: Spacing between model groups
        legend_labels: Labels for legend (if None, uses "Condition 1", "Condition 2", etc.)
        colors: List of colors for each condition (if None, uses default colors)
    """
    if not data:
        raise ValueError("Data must be provided")
    
    labels = list(data.keys())
    # Get the number of values per model (all models should have the same number)
    n_values = len(list(data.values())[0])
    
    # Create figure with academic styling
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')  # White background for figure
    ax.set_facecolor('white')  # White background for plot area
    
    # Add grid first so it appears behind bars
    ax.grid(True, alpha=0.9, linestyle='-', linewidth=0.5, color='#dddddd', axis='y', zorder=0)
    
    # Define consistent colors for each condition (position in tuple)
    if colors is None:
        condition_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    else:
        condition_colors = colors
    
    # Create legend labels if not provided
    if legend_labels is None:
        legend_labels = [f'Condition {i+1}' for i in range(n_values)]
    
    # Calculate positions for grouped bars
    n_models = len(labels)
    total_width = n_values * bar_width
    group_centers = np.arange(n_models) * (total_width + group_spacing)
    
    all_bars = []
    
    # Create bars for each condition
    for condition_idx in range(n_values):
        condition_bars = []
        for model_idx, model_name in enumerate(labels):
            value = data[model_name][condition_idx]
            x_pos = group_centers[model_idx] + condition_idx * bar_width - total_width/2 + bar_width/2
            
            bar = ax.bar(x_pos, value, 
                        width=bar_width,
                        color=condition_colors[condition_idx % len(condition_colors)],
                        alpha=0.9,  # No transparency
                        edgecolor='black',  # Black edges
                        linewidth=0.8,  # Edge line width
                        zorder=3)
            condition_bars.extend(bar)
        all_bars.append(condition_bars)
    
    # Create legend entries with square markers
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, 
                 facecolor=condition_colors[i % len(condition_colors)], 
                 alpha=0.9,  # Match bar transparency
                 edgecolor='black',  # Match bar edges
                 linewidth=0.8,  # Match bar edge width
                 label=legend_labels[i])
        for i in range(n_values)
    ]
    
    # Configure plot with strictly black text
    if xlabel:  # Only set xlabel if not empty
        ax.set_xlabel(xlabel, fontsize=10, fontweight='normal', color='black')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='normal', color='black')
    ax.set_title(title, fontsize=12, fontweight='normal', color='black', pad=10)
    
    # Set x-axis labels at group centers (horizontal, no rotation)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=10, color='black')
    
    # Set y-axis limits with some padding
    all_values = [val for values in data.values() for val in values]
    max_val = max(all_values)
    ax.set_ylim(0, 0.7)
    
    # Format y-axis based on data range
    if max_val <= 1.0:  # Assume percentage values
        from matplotlib.ticker import FuncFormatter
        def percent_formatter(x, pos):
            return f'{x*100:.0f}%'
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    
    # Configure legend at bottom with no border
    legend = ax.legend(
        handles=legend_elements,
        frameon=False,  # Remove frame/border
        fontsize=10,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.2),  # Move up slightly
        ncol=min(n_values, 5),  # Max 4 columns
        handlelength=1.0,  # Make squares more compact
        handletextpad=0.5,  # Reduce space between square and text
        columnspacing=0.5  # Reduce space between columns
    )
    # Set legend text to black
    for text in legend.get_texts():
        text.set_color('black')
    
    # Customize ticks with black text, remove top and right ticks
    ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, 
                   color='#cccccc', labelcolor='black')
    ax.tick_params(top=False, right=False)  # Remove top and right ticks
    
    # Set black colored spines for left and bottom sides only
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_visible(False)  # Hide top spine
    ax.spines['right'].set_visible(False)  # Hide right spine
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Adjust layout to accommodate bottom legend
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        # Ensure PDF extension
        if save_path and not save_path.lower().endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', format='pdf')
        print(f"Comparison bar chart saved to: {save_path}")
    
    plt.show()

def plot_comparison_from_json(
    data_path: str,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot comparison bar chart from JSON configuration file.
    
    Args:
        data_path: Path to JSON file containing comparison data
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    # Load configuration from JSON file
    with open(data_path, "r") as f:
        config = json.load(f)
    
    data = config["data"]  # Dictionary with model_name: [value1, value2, ...]
    title = config.get("title", "Model Performance Comparison")
    xlabel = config.get("xlabel", "")
    ylabel = config.get("ylabel", "Values")
    legend_labels = config.get("legend_labels", None)
    colors = config.get("colors", None)
    
    # Convert list format to tuple format
    formatted_data = {model: tuple(values) for model, values in data.items()}
    
    plot_comparison_bar_chart(
        data=formatted_data,
        save_path=save_path,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        legend_labels=legend_labels,
        colors=colors
    )

if __name__ == "__main__":
    # Example data - each model has multiple values
    data = {
        "Baseline": (0.34, 0.35, 0.19, 0.16),
        "Top 1%": (0.215, 0.23, 0.085, 0.15),
        "Top 3%": (0.16, 0.22, 0.07, 0.085),
        "Random 3%": (0.34, 0.31, 0.20, 0.18),
    }

    data = {
        "Baseline": (0.49, 0.4234, 0.3643, 0.4976),
        "Top 3%": (0.542, 0.4893, 0.4008, 0.56),
        "Top 10%": (0.66, 0.5405, 0.5616, 0.63),
        "Random 10%": (0.4976, 0.4406, 0.29, 0.48),
    }

    # data = {
    #     "Baseline": (0.195, 0.23, 0.19, 0.175),
    #     "Top 1%": (0.175, 0.225, 0.085, 0.12),
    #     "Top 3%": (0.135, 0.215, 0.1, 0.11),
    #     "Random 3%": (0.190, 0.21, 0.20, 0.18),
    # }
    
    # Create output directory
    os.makedirs("outputs/fig", exist_ok=True)
    
    # Plot the comparison bar chart
    plot_comparison_bar_chart(
        data=data,
        save_path="outputs/fig/model_comparison_improvement_prober.pdf",
        title="Refusal Score",
        ylabel="Refusal Score",
        figsize=(4.5, 4),
        legend_labels=["R1-7B", "OR1-7B", "R1-8B", "R1-14B"],
        colors=["#FF99CC", "#7EA6E0", "#67AB9F", "#97D077"]  # Custom colors: blue, red, green
    )