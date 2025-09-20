import torch
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Dict, Any
import json
import numpy as np

# Set matplotlib style to use scienceplots science + notebook styles with retro colors
import scienceplots
plt.style.use(['science', 'notebook', 'retro'])

# Override specific settings to maintain our preferences
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Georgia', 'Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif'],
    'xtick.labelsize': 7,
    'ytick.labelsize': 12,
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
    colors: Optional[Dict[str, str]] = None,  # Now accepts dict mapping labels to colors
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
        colors: Optional dict mapping labels to colors, or None for default colors
        bar_width: Width of bars (0.0 to 1.0)
        show_values: Whether to show values on top of bars
    """
    labels = list(data.keys())
    values = list(data.values())
    
    if not values or not labels:
        raise ValueError("Both values and labels must be provided")
    
    # Use scienceplots retro color palette
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Create figure with programmer styling
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')  # White background for figure
    ax.set_facecolor('#f8f8f8')  # Light gray background for plot area
    
    # Add grid first so it appears behind bars
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#dddddd', axis='y', zorder=0)
    
    # Create bar positions with reduced spacing
    x_pos = np.arange(len(labels)) * 0.5 # Reduce spacing between bars
    
    # Create color list - use custom colors if provided, otherwise default
    bar_colors = []
    for i, label in enumerate(labels):
        if colors and label in colors:
            # Use custom color for this specific label
            bar_colors.append(colors[label])
        elif i == len(values) - 1:  # Last bar is dark gray if no custom color
            bar_colors.append('#404040')  # Dark gray
        else:
            # Use default colors from the palette in order
            bar_colors.append(default_colors[i % len(default_colors)])
    
    # Create standard rectangular bars
    bars = ax.bar(x_pos, values, 
                  width=bar_width,
                  color=bar_colors,
                  alpha=0.85,
                  edgecolor='#888888',
                  linewidth=0.8,
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
    ax.set_title(title, fontsize=10, fontweight='normal', color='#333333', pad=10)
    ax.set_ylabel("Attack Successful Rate", fontsize=10, fontweight='normal', color='#333333', labelpad=0)
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha='right')  # Font size controlled by rcParams
    
    # Set y-axis limits to 0-100% (0.0-1.0 in decimal)
    ax.set_ylim(0, 0.8)
    
    # Format y-axis as percentages
    from matplotlib.ticker import FuncFormatter
    def percent_formatter(x, pos):
        return f'{x*100:.0f}%'
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    
    # Customize ticks with light colors
    ax.tick_params(axis='x', which='major', labelsize=9, width=0.8, 
                   color='#cccccc', labelcolor='#666666')
    ax.tick_params(axis='y', which='major', labelsize=7, width=0.8, 
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
    ax.set_title(title, fontsize=10, fontweight='normal', color='#333333', pad=10)
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
        "OR1-7B": 0.35192307692307695,        # Skywork-OR1-7B
        "R1-7B": 0.3403846153846154,                         # RealSafe-R1-7B
        "R1-8B": 0.19038461538461537,         # DeepSeek-R1-Distill-Llama-8B
        "Hermes-14B": 0.15384615384615385,    # Hermes-4-14B
        "R1-14B": 0.14423076923076922,        # DeepSeek-R1-Distill-Qwen-14B
        "Phi4-mini": 0.175,                   # Phi-4-mini-reasoning
        "Phi4": 0.07884615384615384,   
        "QwQ-32B": 0.019230769230769232,      # QwQ-32B                     # Phi-4-reasoning
        "Qwen3-4B": 0.0019230769230769232,    # Qwen3-4B-Thinking-2507
        "Qwen3-30B": 0.0010230769230769232,
        "Realsafe-7B": 0.0,                   # RealSafe-R1-7B
        "Realsafe-8B": 0.0,                   # RealSafe-R1-8B
        # "LLaMA-8B": missing in all.sh, skip
        # "Qwen2.5-7B": missing in all.sh, skip
    }

    data = {
        "R1-8B": 0.329,
        "R1-7B": 0.396,
        "Hermes-14B": 0.36,
        "OR1-7B": 0.38,
        "R1-14B": 0.44,
        "QwQ-32B": 0.16,
        "Qwen3-4B": 0.015,
        "Qwen3-30B": 0.025,
        "Phi4": 0.015,
        "Phi4-mini": 0.175,                   # Phi-4-mini-reasoning
        "Realsafe-7B": 0.0,                   # RealSafe-R1-7B
        "Realsafe-8B": 0.0,                   # RealSafe-R1-8B
    }

    # rank by ASR rate
    data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
    
    # Create output directory
    os.makedirs("outputs/fig", exist_ok=True)
    
    # Custom colors for specific models (scienceplots retro color palette)
    retro_colors = ["#B5739D", "#7EA6E0", "#67AB9F", "#97D077", "#FFD966", "#FFB570", "#EA6B66"]
    
    custom_colors = {
        "OR1-7B": "#B5739D",      # Retro Purple
        "R1-7B": "#7EA6E0",       # Retro Blue
        "R1-8B": "#67AB9F",       # Retro Teal
        "Hermes-14B": "#97D077",  # Retro Green
        "R1-14B": "#FFD966",      # Retro Yellow
        "Phi4-mini": "#FFB570",   # Retro Orange
        "QwQ-32B": "#EA6B66",     # Retro Red
        "Phi4": "#B5739D",        # Retro Purple (repeat)
        "Qwen3-4B": "#7EA6E0",    # Retro Blue (repeat)
        "Qwen3-30B": "#67AB9F",   # Retro Teal (repeat)
        "Realsafe-7B": "#696969", # Dim Gray
        "Realsafe-8B": "#778899", # Light Slate Gray
    }

    custom_colors = {
        "OR1-7B": "#EA6B66",     # Retro Red
        "R1-7B": "#FFB570",   # Retro Orange
        "R1-8B": "#FFB570",   # Retro Orange
        "Hermes-14B": "#EA6B66",     # Retro Red
        "R1-14B": "#FFB570",   # Retro Orange
        "Phi4-mini": "#FFB570",   # Retro Orange
        "QwQ-32B": "#EA6B66",     # Retro Red
        "Phi4":"#EA6B66",     # Retro Red
        "Qwen3-4B": "#97D077",  # Retro Green
        "Qwen3-30B": "#97D077",  # Retro Green
        "Realsafe-7B": "#97D077",  # Retro Green
        "Realsafe-8B": "#97D077",  # Retro Green
    }
    
    # Plot the bar chart using the data dictionary
    plot_bar_chart_from_dict(
        data=data,
        save_path="outputs/fig/model_comparison_wj.pdf",
        figsize=(6, 3),
        title="WildJailbreak",
        colors=custom_colors,  # Pass custom colors
    ) 