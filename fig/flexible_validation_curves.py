import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict
import fire

# Set matplotlib style to use scienceplots retro
import scienceplots
plt.style.use(['science', 'no-latex', 'retro'])

# Override specific settings to use Georgia font
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

def plot_flexible_curves(
    curves_data: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None,
    figsize: tuple = (5.5, 3),
    linewidth: float = 2.0,
    title: str = "Training Metrics Over Epochs",
    xlabel: str = "Training Data",
    ylabel: str = "Attack Successful Rate"
) -> None:
    """
    Plot multiple curves with flexible naming.
    
    Args:
        curves_data: Dictionary mapping curve names to dict with 'x' and 'y' values
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        linewidth: Width of the plot lines
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    if not curves_data:
        raise ValueError("curves_data cannot be empty")
    
    # Validate that all curves have x and y data with same length
    for name, data in curves_data.items():
        if 'x' not in data or 'y' not in data:
            raise ValueError(f"Curve '{name}' must have both 'x' and 'y' data")
        if len(data['x']) != len(data['y']):
            raise ValueError(f"Curve '{name}': x and y data must have the same length")
    
    # No need for epoch range since we use custom x values
    
    # Get retro colors from matplotlib's current color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')  # White background for figure
    ax.set_facecolor('#f8f8f8')  # Light gray background for plot area
    
    # Plot each curve
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    linestyles = ['-', '--', '-.', ':']
    
    for i, (name, data) in enumerate(curves_data.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        
        ax.plot(data['x'], data['y'],
                label=name,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.8,
                marker=marker,
                markersize=6,
                markerfacecolor=color,
                markeredgecolor='black',
                markeredgewidth=0.8)
    
    # Configure axes
    ax.set_xlabel(xlabel, fontsize=12, fontweight='normal', color='#333333')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='normal', color='#333333')
    
    # Set axis limits and ticks
    ax.set_xlim(-5, 105)  # X-axis from 0% to 100%
    ax.set_xticks(np.arange(0, 101, 20))  # Ticks every 20%
    
    # Set y-axis limits inverted (from 0.4 to -0.1, top to bottom)
    ax.set_ylim(0.35, -0.05)
    
    # Add grid
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='#dddddd')
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, 
                   color='#cccccc', labelcolor='#666666')
    
    # Set light colored spines
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['top'].set_color('#cccccc')
    ax.spines['right'].set_color('#cccccc')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['top'].set_linewidth(0.8)
    ax.spines['right'].set_linewidth(0.8)
    
    # Create legend in bottom right with distance from axes
    legend = ax.legend(frameon=True, 
                       fancybox=True,  # Enable rounded corners
                       edgecolor='none',  # No border
                       facecolor='#e8e8e8',  # Darker gray background
                       framealpha=0.95,
                       fontsize=9,
                       loc='lower right',
                       bbox_to_anchor=(0.98, 0.02),  # Position with distance from axes
                       borderpad=1.0,  # Increase padding between text and legend border
                       handletextpad=0.8,  # Space between legend markers and text
                       columnspacing=1.0)  # Space between columns if multiple
    legend.get_frame().set_linewidth(0)
    

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

def main(
    save_path: Optional[str] = None,
    title: str = "Training Metrics Over Epochs",
    xlabel: str = "Training Data",
    ylabel: str = "Attack Successful Rate",
    **kwargs
) -> None:
    """
    Main function to create flexible validation curves plot.
    
    Args:
        save_path: Optional path to save the plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        **kwargs: Curve names as keys and comma-separated values as strings
    
    Example:
        python flexible_validation_curves.py main \
            --validation_accuracy_x="0,25,50,75,100" \
            --validation_accuracy_y="0.15,0.12,0.09,0.06,0.03" \
            --validation_loss_x="0,25,50,75,100" \
            --validation_loss_y="0.35,0.32,0.28,0.25,0.23" \
            --save_path="my_curves.pdf" \
            --title="My Training Curves"
    """
    # Helper function to parse input data
    def parse_input(data):
        if isinstance(data, str):
            return [float(x.strip()) for x in data.split(',')]
        elif isinstance(data, (list, tuple)):
            return [float(x) for x in data]
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    # Parse curve data from kwargs
    # Expected format: curve_name_x and curve_name_y pairs
    curves_data = {}
    
    # Group x and y data by curve name
    curve_groups = {}
    for param_name, param_values in kwargs.items():
        if param_name.endswith('_x'):
            curve_name = param_name[:-2]  # Remove '_x' suffix
            if curve_name not in curve_groups:
                curve_groups[curve_name] = {}
            curve_groups[curve_name]['x'] = parse_input(param_values)
        elif param_name.endswith('_y'):
            curve_name = param_name[:-2]  # Remove '_y' suffix
            if curve_name not in curve_groups:
                curve_groups[curve_name] = {}
            curve_groups[curve_name]['y'] = parse_input(param_values)
    
    # Convert to final format with proper names
    for curve_name, data in curve_groups.items():
        if 'x' in data and 'y' in data:
            display_name = curve_name.replace('_', ' ').title()
            curves_data[display_name] = data
    
    if not curves_data:
        # Provide example data if no curves specified
        print("No curve data provided. Using example data...")
        # Create example data with x values as percentages (0% to 100%)
        x_values = [0, 5, 12.5, 25, 50, 100]  # Percentages
        curves_data = {
            "Cliff-as-a-Judge": {
                'x': x_values,
                'y': [0.235, 0.055, 0.01, 0.02, 0.02, 0.03]
            },
            "Baseline": {
                'x': x_values,
                'y': [0.235, 0.06, 0.04, 0.08, 0.15, 0.03]
            },
            "Rule-Based": {
                'x': x_values,
                'y': [0.235, 0.07, 0.09, 0.08, 0.07, 0.03]
            },
            "LLM-as-a-Judge": {
                'x': x_values,
                'y': [0.235, 0.06, 0.014, 0.06, 0.05, 0.03]
            },
        }
    
    # Create the plot
    plot_flexible_curves(
        curves_data=curves_data,
        save_path=save_path,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel
    )
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    fire.Fire(main) 