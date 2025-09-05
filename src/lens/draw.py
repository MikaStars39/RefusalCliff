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
    title: str = "Prober Results Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (6, 6),
    colors: Optional[List[str]] = None
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
                    ax.plot(x_axis, result_tensor.numpy(), 
                           label=f"{label}_{item_type}", 
                           color=colors[color_idx], 
                           linewidth=2.5,
                           alpha=0.8)
            elif isinstance(results, torch.Tensor):
                # Single tensor result
                if results.dim() == 1 and len(results) == 100:
                    # 1D tensor with 100 points
                    color = colors[i % len(colors)]
                    ax.plot(x_axis, results.numpy(), 
                           label=label, 
                           color=color, 
                           linewidth=2.5,
                           alpha=0.8)
                elif results.dim() == 2:
                    # 2D tensor (multiple sequences or layer results)
                    if results.shape[1] == 100:
                        # Each row is a sequence
                        for j, seq in enumerate(results):
                            color_idx = (i + j) % len(colors)
                            ax.plot(x_axis, seq.numpy(), 
                                   label=f"{label}_seq{j}", 
                                   color=colors[color_idx], 
                                   linewidth=2.5,
                                   alpha=0.8)
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
    
    # Set axis limits and ticks
    ax.set_xlim(0, 99)
    ax.set_ylim(0, 1)
    
    # Customize ticks with light colors
    ax.tick_params(axis='both', which='major', labelsize=12, width=0.8, 
                   color='#cccccc', labelcolor='#666666')
    ax.set_xticks(np.arange(0, 100, 20))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    
    # Add grid with subtle styling
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