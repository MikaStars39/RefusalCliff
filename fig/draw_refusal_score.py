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

def plot_multiple_prober_results(
    pt_paths: str,  # Changed to string for JSON file path
    save_path: Optional[str] = None,
    figsize: tuple = (6, 6),
    colors: Optional[List[str]] = None,
    linewidth: float = 1.0,
) -> None:
    """
    Plot multiple prober result .pt files on a single figure with academic style.
    
    Args:
        pt_paths: Path to JSON file containing pt_paths and labels
        title: Plot title
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        colors: Optional list of colors for each plot line
    """
    # Load configuration from JSON file
    with open(pt_paths, "r") as f:
        config = json.load(f)
    
    labels = config["labels"]
    pt_file_paths = config["pt_paths"]
    
    if not pt_file_paths:
        raise ValueError("At least one .pt file path must be provided")
    
    # Academic color palette - sophisticated and distinguishable colors
    if colors is None:
        colors = [
            '#8B4513',  # Saddle Brown (similar to the chart)
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
    
    # Create x-axis (assuming normalized to 100 points)
    x_axis = np.arange(100)
    
    for i, (pt_path, label) in enumerate(zip(pt_file_paths, labels)):
        if not os.path.exists(pt_path):
            print(f"Warning: File not found - {pt_path}")
            continue
            
        try:
            # Load the .pt file
            results = torch.load(pt_path, map_location='cpu')
            
            # Handle different result formats
            if isinstance(results, dict):
                # If multiple item types in one file, plot all of them
                for j, (item_type, result_tensor) in enumerate(results.items()):
                    color_idx = (i * len(results) + j) % len(colors)
                    base_color = colors[color_idx]
                    
                    # Plot the first 90% with lighter color
                    ax.plot(x_axis[:90], result_tensor[:90].numpy(), 
                           color=base_color, 
                           linewidth=linewidth,
                           alpha=0.4)  # Lighter alpha for first 90%
                    
                    # Plot the last 10% with normal color
                    ax.plot(x_axis[89:], result_tensor[89:].numpy(), 
                           label=f"{label}", 
                           color=base_color, 
                           linewidth=linewidth,
                           alpha=0.8)  # Normal alpha for last 10%
                           
            elif isinstance(results, torch.Tensor):
                # Single tensor result
                if results.dim() == 1 and len(results) == 100:
                    # 1D tensor with 100 points
                    color = colors[i % len(colors)]
                    
                    # Plot the first 90% with lighter color
                    ax.plot(x_axis[:90], results[:90].numpy(), 
                           color=color, 
                           linewidth=linewidth,
                           alpha=0.4)  # Lighter alpha for first 90%
                    
                    # Plot the last 10% with normal color
                    ax.plot(x_axis[89:], results[89:].numpy(), 
                           label=label, 
                           color=color, 
                           linewidth=linewidth,
                           alpha=0.8)  # Normal alpha for last 10%
                           
                elif results.dim() == 2:
                    # 2D tensor (multiple sequences or layer results)
                    if results.shape[1] == 100:
                        # Each row is a sequence
                        for j, seq in enumerate(results):
                            color_idx = (i + j) % len(colors)
                            base_color = colors[color_idx]
                            
                            # Plot the first 90% with lighter color
                            ax.plot(x_axis[:90], seq[:90].numpy(), 
                                   color=base_color, 
                                   linewidth=linewidth,
                                   alpha=0.4)  # Lighter alpha for first 90%
                            
                            # Plot the last 10% with normal color
                            ax.plot(x_axis[89:], seq[89:].numpy(), 
                                   label=f"{label}", 
                                   color=base_color, 
                                   linewidth=linewidth,
                                   alpha=0.8)  # Normal alpha for last 10%
                    else:
                        print(f"Warning: Unexpected tensor shape {results.shape} in {pt_path}")
                else:
                    print(f"Warning: Unexpected tensor format in {pt_path}")
            else:
                print(f"Warning: Unexpected data type {type(results)} in {pt_path}")
                
        except Exception as e:
            print(f"Error loading {pt_path}: {e}")
            continue
    
    # Configure plot with programmer styling
    ax.set_xlabel('Normalized Position', fontsize=16, fontweight='normal', color='#333333')
    ax.set_ylabel('Refusal Score', fontsize=16, fontweight='normal', color='#333333')
    
    # Add grayer background for the last 10% of x-axis (right 10%)
    ax.axvspan(90, 105, alpha=0.2, color='#c8c8c8', zorder=0)  # More subtle background
    
    # Set axis limits and ticks
    ax.set_xlim(0, 105)  # Changed from 110 to 105
    ax.set_ylim(0, 1)
    
    # Add large dots for the last point (position 99, index 99) for each plotted line
    for i, (pt_path, label) in enumerate(zip(pt_file_paths, labels)):
        if not os.path.exists(pt_path):
            continue
            
        try:
            # Load the .pt file
            results = torch.load(pt_path, map_location='cpu')
            
            # Handle different result formats to get the last point
            if isinstance(results, dict):
                for j, (item_type, result_tensor) in enumerate(results.items()):
                    color_idx = (i * len(results) + j) % len(colors)
                    if len(result_tensor) >= 100:
                        ax.scatter(99, result_tensor[99].item(), 
                                 color=colors[color_idx], s=30, zorder=5, alpha=0.9)
            elif isinstance(results, torch.Tensor):
                if results.dim() == 1 and len(results) >= 100:
                    color = colors[i % len(colors)]
                    ax.scatter(99, results[99].item(), 
                             color=color, s=30, zorder=5, alpha=0.9)
                elif results.dim() == 2 and results.shape[1] >= 100:
                    for j, seq in enumerate(results):
                        color_idx = (i + j) % len(colors)
                        ax.scatter(99, seq[99].item(), 
                                 color=colors[color_idx], s=30, zorder=5, alpha=0.9)
        except Exception as e:
            continue
    
    # Customize ticks with light colors
    ax.tick_params(axis='both', which='major', labelsize=12, width=0.8, 
                   color='#cccccc', labelcolor='#666666')
    ax.set_xticks(np.arange(0, 105, 20))  # This will show 0, 20, 40, 60, 80, 100
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    
    # Add grid with subtle styling (this will include vertical line at 100)
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='#dddddd')
    
    # Configure legend with light styling in top-left corner
    legend = ax.legend(
        frameon=True, 
        fancybox=False, 
        edgecolor='#dddddd',  # Light gray border
        facecolor='#f8f8f8',  # Light gray background
        framealpha=0.9,
        fontsize=10,
        loc='upper left'  # Position in top-left corner
    )
    legend.get_frame().set_linewidth(0.8)
    
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
        print(f"Plot saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_multiple_prober_results(
        pt_paths="outputs/fig/pt_files.json",
        save_path="outputs/fig/refusal_score.png",
        linewidth=1.5,
        figsize=(6, 4),
    )