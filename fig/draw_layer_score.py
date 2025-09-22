import torch
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Dict, Any
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Set matplotlib style to use scienceplots retro
import scienceplots
plt.style.use(['science', 'no-latex', 'retro'])

# Override specific settings to maintain our preferences
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Georgia', 'Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif'],
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'axes.facecolor': 'white',  # White background for plot area
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

def plot_multiple_prober_results(
    pt_paths: str,  # Changed to string for JSON file path
    save_path: Optional[str] = None,
    figsize: tuple = (6, 4),
    start_color: str = "#1f77b4",  # Blue color (default matplotlib blue)
    end_color: str = "#808080",    # Gray color
    linewidth: float = 1.0,
    title: str = "Refusal Score Analysis",
) -> None:
    """
    Plot multiple prober result .pt files on a single figure with gradient colors.
    
    Args:
        pt_paths: Path to JSON file containing pt_paths and labels
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        start_color: Starting color for gradient (hex format)
        end_color: Ending color for gradient (hex format)
        linewidth: Width of the plot lines
        title: Plot title
    """
    # Load configuration from JSON file
    with open(pt_paths, "r") as f:
        config = json.load(f)
    
    labels = config["labels"]
    pt_file_paths = config["pt_paths"]
    
    if not pt_file_paths:
        raise ValueError("At least one .pt file path must be provided")
    
    # Create figure with programmer styling
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')  # White background for figure
    ax.set_facecolor('white')  # White background for plot area
    
    # Collect all plot data first to determine total number of lines
    plot_data = []
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
                    plot_data.append({
                        'data': result_tensor,
                        'label': f"{label}",
                        'file_idx': i,
                        'sub_idx': j
                    })
                           
            elif isinstance(results, torch.Tensor):
                # Single tensor result
                if results.dim() == 1 and len(results) == 100:
                    # 1D tensor with 100 points
                    plot_data.append({
                        'data': results,
                        'label': label,
                        'file_idx': i,
                        'sub_idx': 0
                    })
                           
                elif results.dim() == 2:
                    # 2D tensor (multiple sequences or layer results)
                    if results.shape[1] == 100:
                        # Each row is a sequence
                        for j, seq in enumerate(results):
                            plot_data.append({
                                'data': seq,
                                'label': f"{label}",
                                'file_idx': i,
                                'sub_idx': j
                            })
                    else:
                        print(f"Warning: Unexpected tensor shape {results.shape} in {pt_path}")
                else:
                    print(f"Warning: Unexpected tensor format in {pt_path}")
            else:
                print(f"Warning: Unexpected data type {type(results)} in {pt_path}")
                
        except Exception as e:
            print(f"Error loading {pt_path}: {e}")
            continue
    
    # Generate gradient colors
    total_lines = len(plot_data)
    if total_lines == 0:
        print("No valid data to plot")
        return
    
    # Create gradient colors from start_color to end_color
    def hex_to_rgb(hex_color):
        """Convert hex color to RGB tuple (0-1 range)"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    
    start_rgb = hex_to_rgb(start_color)
    end_rgb = hex_to_rgb(end_color)
    
    # Generate colors for each line
    gradient_colors = []
    for i in range(total_lines):
        if total_lines == 1:
            # If only one line, use start color
            ratio = 0
        else:
            ratio = i / (total_lines - 1)  # 0 to 1
        
        # Interpolate between start and end colors
        r = start_rgb[0] + ratio * (end_rgb[0] - start_rgb[0])
        g = start_rgb[1] + ratio * (end_rgb[1] - start_rgb[1])
        b = start_rgb[2] + ratio * (end_rgb[2] - start_rgb[2])
        
        gradient_colors.append((r, g, b))
    
    # Plot all data with gradient colors
    for i, data_item in enumerate(plot_data):
        result_tensor = data_item['data']
        label = data_item['label']
        color = gradient_colors[i]
        
        # Sample points every 5 positions, plus position 99
        sample_positions = np.concatenate([np.arange(0, 100, 5), [99]])  # 0, 5, 10, 15, ..., 95, 99
        sample_values = result_tensor[sample_positions].numpy()
        
        # Plot only the sampled points connected directly
        ax.plot(sample_positions, sample_values, 
               label=label, 
               color=color, 
               linewidth=linewidth,
               alpha=0.8)  # Consistent alpha for entire line
        
        # Add circular markers at sampled positions
        ax.scatter(sample_positions, sample_values,
                 facecolor=color, edgecolor='none', s=15, 
                 zorder=6, alpha=0.9)
    
    # Configure plot with programmer styling
    ax.set_title(title, fontsize=10, fontweight='normal', color='#333333', pad=10)
    ax.set_xlabel('Normalized Position', fontsize=10, fontweight='normal', color='#333333')
    ax.set_ylabel('Refusal Score', fontsize=10, fontweight='normal', color='#333333')
    
    # Add grayer background for the last 10% of x-axis (right 10%)
    ax.axvspan(90, 105, alpha=0.25, color='#d0d0d0', zorder=0)  # Darker gray background
    
    # Set axis limits and ticks
    ax.set_xlim(-5, 105)  # Start from -5 for better spacing
    ax.set_ylim(-0.1, 0.85)
    
    # Customize ticks with light colors
    ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, 
                   color='#cccccc', labelcolor='#666666')
    ax.set_xticks(np.arange(0, 105, 20))  # This will show 0, 20, 40, 60, 80, 100
    ax.set_yticks(np.arange(0, 1.0, 0.2))  # This will show 0.0, 0.2, 0.4, 0.6, 0.8
    
    # Add grid with subtle styling (this will include vertical line at 100)
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='#dddddd')
    
    # Configure legend with white background, positioned in top-left corner
    legend = ax.legend(
        frameon=True, 
        fancybox=True,  # Enable rounded corners
        edgecolor='none',  # No border
        facecolor='white',  # White background
        framealpha=0.95,
        fontsize=8,
        loc='upper left',  # Position in top-left corner
        borderpad=1.0,  # Increase padding between text and legend border
        handletextpad=0.5,  # Space between legend markers and text
        handlelength=1.0,  # Length of legend lines (shorter)
        columnspacing=1.0  # Space between columns if multiple
    )
    legend.get_frame().set_linewidth(0)
    
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
        print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    import fire
    fire.Fire(plot_multiple_prober_results)