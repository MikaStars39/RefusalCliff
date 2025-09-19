import torch
import matplotlib.pyplot as plt
import os
from typing import Optional
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
    'axes.spines.top': True,  # Show top spine
    'axes.spines.right': True,  # Show right spine
    'xtick.direction': 'out',  # Ticks point outward
    'ytick.direction': 'out',  # Ticks point outward
    'xtick.minor.visible': False,  # Hide minor x ticks
    'ytick.minor.visible': False,  # Hide minor y ticks
})

def plot_single_curve_with_references(
    pt_path: str,
    save_path: Optional[str] = None,
    figsize: tuple = (6, 4),
    linewidth: float = 1.5,
    title: str = "Refusal Score Analysis",
    normal_refusal_score: float = 0.3,
    safe_model_plateau: float = 0.7,
    curve_label: str = "Model Response",
) -> None:
    """
    Plot a single curve with two horizontal reference lines.
    
    Args:
        pt_path: Path to .pt file containing the data
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        linewidth: Width of the plot line
        title: Plot title
        normal_refusal_score: Y-value for Normal Refusal Score horizontal line
        safe_model_plateau: Y-value for Safe Model Plateau horizontal line
        curve_label: Label for the main curve
    """
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"File not found: {pt_path}")
    
    # Load the .pt file
    results = torch.load(pt_path, map_location='cpu')
    
    # Get retro colors from matplotlib's current color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Create figure with styling
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')  # White background for figure
    ax.set_facecolor('#f8f8f8')  # Light gray background for plot area
    
    # Handle different result formats
    if isinstance(results, dict):
        # If multiple item types in one file, plot the first one
        result_tensor = list(results.values())[0]
    elif isinstance(results, torch.Tensor):
        if results.dim() == 1 and len(results) == 100:
            result_tensor = results
        elif results.dim() == 2:
            # Take the first sequence if 2D
            result_tensor = results[0]
        else:
            raise ValueError(f"Unexpected tensor format: {results.shape}")
    else:
        raise ValueError(f"Unexpected data type: {type(results)}")
    
    # Sample points every 5 positions, plus position 99
    sample_positions = np.concatenate([np.arange(0, 100, 5), [99]])  # 0, 5, 10, 15, ..., 95, 99
    sample_values = result_tensor[sample_positions].numpy()
    
    # Plot only the sampled points connected directly
    main_color = colors[0]
    ax.plot(sample_positions, sample_values, 
           label=curve_label, 
           color=main_color, 
           linewidth=linewidth,
           alpha=0.8)
    
    # Add circular markers at sampled positions
    ax.scatter(sample_positions, sample_values,
             facecolor=main_color, edgecolor='black', s=25, 
             linewidth=0.8, zorder=6, alpha=0.9)
    
    # Add horizontal reference lines
    ax.axhline(y=normal_refusal_score, color='#d62728', linestyle='--', 
               linewidth=1.5, alpha=0.7, label='Normal Refusal Score')
    ax.axhline(y=safe_model_plateau, color='#2ca02c', linestyle='--', 
               linewidth=1.5, alpha=0.7, label='Safe Model Plateau')
    
    # Configure plot styling
    ax.set_title(title, fontsize=12, fontweight='normal', color='#333333', pad=10)
    ax.set_xlabel('Normalized Position', fontsize=10, fontweight='normal', color='#333333')
    ax.set_ylabel('Refusal Score', fontsize=10, fontweight='normal', color='#333333')
    
    # Add grayer background for the last 5% of x-axis (right 5%)
    ax.axvspan(95, 105, alpha=0.2, color='#c8c8c8', zorder=0)  # More subtle background
    
    # Set axis limits and ticks
    ax.set_xlim(-5, 105)  # Start from -5 for better spacing
    ax.set_ylim(-0.1, 0.75)
    
    # Customize ticks with light colors
    ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, 
                   color='#cccccc', labelcolor='#666666')
    ax.set_xticks(np.arange(0, 105, 20))  # This will show 0, 20, 40, 60, 80, 100
    ax.set_yticks(np.arange(0, 1.0, 0.2))  # This will show 0.0, 0.2, 0.4, 0.6, 0.8
    
    # Add grid with subtle styling
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='#dddddd')
    
    # Configure legend with darker background, no border, and rounded corners
    legend = ax.legend(
        frameon=True, 
        fancybox=True,  # Enable rounded corners
        edgecolor='none',  # No border
        facecolor='#e8e8e8',  # Darker gray background
        framealpha=0.95,
        fontsize=8,
        loc=(0.05, 0.60),  # Position away from top-left corner
        borderpad=1.0,  # Increase padding between text and legend border
        handletextpad=0.8,  # Space between legend markers and text
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
    fire.Fire(plot_single_curve_with_references) 