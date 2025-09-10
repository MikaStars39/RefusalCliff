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
    'font.family': 'monospace',
    'font.monospace': ['Consolas', 'DejaVu Sans Mono', 'Courier New', 'monospace'],
    'font.size': 8,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 8,  # Smaller x-axis tick labels
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'axes.facecolor': '#f8f8f8',  # Light gray background for plot area
    'figure.facecolor': 'white',  # White background for figure
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'axes.spines.top': False,  # Remove top spine
    'axes.spines.right': False,  # Remove right spine
    'xtick.direction': 'out',  # Ticks point outward
    'ytick.direction': 'out',  # Ticks point outward
    'xtick.minor.visible': False,  # Hide minor x ticks
    'ytick.minor.visible': False,  # Hide minor y ticks
})

def plot_comparison_bar_chart(
    data: Dict[str, Tuple[float, float]],  # model_name: (before_value, after_value)
    save_path: Optional[str] = None,
    title: str = "Model Performance Comparison",
    xlabel: str = "Models",
    ylabel: str = "Values",
    figsize: tuple = (10, 6),
    bar_width: float = 0.35,
    show_values: bool = True,
    show_improvement: bool = True,
) -> None:
    """
    Plot comparison bar chart with dual bars per model showing improvement.
    
    Args:
        data: Dictionary with model names as keys and (before, after) tuples as values
        save_path: Optional path to save the plot
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size tuple
        bar_width: Width of individual bars
        show_values: Whether to show values on top of bars
        show_improvement: Whether to show improvement percentage above right bars
    """
    if not data:
        raise ValueError("Data must be provided")
    
    labels = list(data.keys())
    before_values = [data[label][0] for label in labels]
    after_values = [data[label][1] for label in labels]
    
    # Create figure with academic styling
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')  # White background for figure
    ax.set_facecolor('#f8f8f8')  # Light gray background for plot area
    
    # Add grid first so it appears behind bars
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#dddddd', axis='y', zorder=0)
    
    # Create bar positions
    x_pos = np.arange(len(labels))
    
    # Define colors - left bars are lighter (grayer), right bars are darker
    left_color = '#a0a0a0'  # Light gray for before values
    right_color = '#404040'  # Dark gray for after values
    
    # Create bars
    bars_before = ax.bar(x_pos - bar_width/2, before_values, 
                        width=bar_width,
                        color=left_color,
                        alpha=0.85,
                        edgecolor='none',
                        label='Before',
                        zorder=3)
    
    bars_after = ax.bar(x_pos + bar_width/2, after_values, 
                       width=bar_width,
                       color=right_color,
                       alpha=0.85,
                       edgecolor='none',
                       label='After',
                       zorder=3)
    
    # Add value labels on top of bars if requested
    if show_values:
        for i, (bar_before, bar_after, before_val, after_val) in enumerate(zip(bars_before, bars_after, before_values, after_values)):
            # Label for before bar
            height_before = bar_before.get_height()
            ax.text(bar_before.get_x() + bar_before.get_width()/2., height_before + max(max(before_values), max(after_values)) * 0.01,
                   f'{before_val*100:.1f}%' if before_val < 1 else f'{before_val:.1f}',
                   ha='center', va='bottom',
                   fontsize=9, color='#555555',
                   fontweight='normal',
                   zorder=5)
            
            # Label for after bar
            height_after = bar_after.get_height()
            ax.text(bar_after.get_x() + bar_after.get_width()/2., height_after + max(max(before_values), max(after_values)) * 0.01,
                   f'{after_val*100:.1f}%' if after_val < 1 else f'{after_val:.1f}',
                   ha='center', va='bottom',
                   fontsize=9, color='#333333',
                   fontweight='normal',
                   zorder=5)
    
    # Add improvement percentage labels above right bars
    if show_improvement:
        for i, (bar_after, before_val, after_val) in enumerate(zip(bars_after, before_values, after_values)):
            improvement = abs(after_val - before_val)
            if before_val < 1:  # Assume percentage values
                improvement_text = f'+{improvement*100:.1f}%'
            else:
                improvement_text = f'+{improvement:.1f}'
            
            height_after = bar_after.get_height()
            ax.text(bar_after.get_x() + bar_after.get_width()/2., 
                   height_after + max(max(before_values), max(after_values)) * 0.05,
                   improvement_text,
                   ha='center', va='bottom',
                   fontsize=10, color='#d62728',  # Red color for improvement
                   fontweight='bold',
                   zorder=5)
    
    # Configure plot
    ax.set_xlabel(xlabel, fontsize=12, fontweight='normal', color='#333333')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='normal', color='#333333')
    ax.set_title(title, fontsize=14, fontweight='normal', color='#333333', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20 if len(max(labels, key=len)) > 8 else 0, 
                       ha='right' if len(max(labels, key=len)) > 8 else 'center',
                       fontsize=10)
    
    # Set y-axis limits with some padding for improvement labels
    max_val = max(max(before_values), max(after_values))
    ax.set_ylim(0, max_val * 1.15)
    
    # Format y-axis based on data range
    if max_val <= 1.0:  # Assume percentage values
        from matplotlib.ticker import FuncFormatter
        def percent_formatter(x, pos):
            return f'{x*100:.0f}%'
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    
    # Configure legend
    legend = ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor='#dddddd',
        facecolor='#f8f8f8',
        framealpha=0.9,
        fontsize=10,
        loc='upper left'
    )
    legend.get_frame().set_linewidth(0.8)
    
    # Customize ticks with light colors
    ax.tick_params(axis='x', which='major', labelsize=10, width=0.8, 
                   color='#cccccc', labelcolor='#666666')
    ax.tick_params(axis='y', which='major', labelsize=12, width=0.8, 
                   color='#cccccc', labelcolor='#666666')
    
    # Set light colored spines
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Adjust layout
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
    
    data = config["data"]  # Dictionary with model_name: [before, after]
    title = config.get("title", "Model Performance Comparison")
    xlabel = config.get("xlabel", "Models")
    ylabel = config.get("ylabel", "Values")
    
    # Convert list format to tuple format
    formatted_data = {model: tuple(values) for model, values in data.items()}
    
    plot_comparison_bar_chart(
        data=formatted_data,
        save_path=save_path,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize
    )

if __name__ == "__main__":
    # Example data - each model has (before, after) values
    data = {
        "R1-LLaMA-8B": (0.30, 0.60),
        "R1-Qwen-7B": (0.25, 0.52),
        "Hermes-14B": (0.20, 0.50),
        "Skywork-OR1-7B": (0.15, 0.42),
        "R1-Qwen-14B": (0.10, 0.22),
        "QwQ-32B": (0.005, 0.01),
        "Qwen3-Thinking-4B": (0.01, 0.025),
        "LLaMA-8B-it": (0.008, 0.02),
    }
    
    # Create output directory
    os.makedirs("outputs/fig", exist_ok=True)
    
    # Plot the comparison bar chart
    plot_comparison_bar_chart(
        data=data,
        save_path="outputs/fig/model_comparison_improvement.pdf",
        title="Model Performance: Before vs After Optimization",
        ylabel="Attack Success Rate",
        figsize=(10, 6)
    ) 