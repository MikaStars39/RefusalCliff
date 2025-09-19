import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import fire

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

def plot_attack_metrics(
    attack_success_rate: List[float],
    refusal_score: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (4, 4),
    linewidth: float = 2.0
) -> None:
    """
    Plot attack successful rate and refusal score curves.
    
    Args:
        attack_success_rate: List of attack successful rate values for each percentage
        refusal_score: List of refusal score values for each percentage
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        linewidth: Width of the plot lines
    """
    # Validate input lengths
    data_points = len(attack_success_rate)
    if len(refusal_score) != data_points:
        raise ValueError("Attack success rate and refusal score lists must have the same length")
    
    # Create percentage range from 100% to 0%
    percentages = np.linspace(100, 0, data_points)
    
    # Get retro colors from matplotlib's current color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Create plot with dual y-axis
    fig, ax1 = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')  # White background for figure
    ax1.set_facecolor('#f8f8f8')  # Light gray background for plot area
    
    # Create second y-axis
    ax2 = ax1.twinx()
    
    # Plot attack success rate on left y-axis (solid line)
    line1 = ax1.plot(percentages, attack_success_rate, 
                     label='Attack Successful Rate (ASR)', 
                     color=colors[0], 
                     linewidth=linewidth,
                     linestyle='-',  # solid line
                     alpha=0.8,
                     marker='o',
                     markersize=6,
                     markerfacecolor=colors[0],
                     markeredgecolor='black',
                     markeredgewidth=0.8)
    
    # Plot refusal score on right y-axis (dashed line)
    line2 = ax2.plot(percentages, refusal_score, 
                     label='Refusal Score', 
                     color=colors[1], 
                     linewidth=linewidth,
                     linestyle='--',  # dashed line
                     alpha=0.8,
                     marker='s',
                     markersize=6,
                     markerfacecolor=colors[1],
                     markeredgecolor='black',
                     markeredgewidth=0.8)
    
    # Configure axes
    ax1.set_xlabel('Thinking Pruning', fontsize=10, fontweight='normal', color='#333333')
    ax1.set_ylabel('Attack Successful Rate (ASR)', fontsize=10, fontweight='normal', color='#333333', labelpad=2)
    ax2.set_ylabel('Refusal Score', fontsize=10, fontweight='normal', color='#333333', labelpad=2)
    
    # Set x-axis limits and ticks (110% to -10% for padding)
    ax1.set_xlim(110, -10)  # Extended range to prevent edge clipping
    x_ticks = np.arange(100, -1, -20)  # 100%, 80%, 60%, 40%, 20%, 0%
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f'{int(x)}%' for x in x_ticks])
    
    # Set y-axis limits (-0.1 to 0.8) for both axes
    ax1.set_ylim(-0.1, 0.8)
    ax2.set_ylim(-0.1, 0.8)
    
    # Add grid
    ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='#dddddd')
    
    # Customize ticks
    ax1.tick_params(axis='both', which='major', labelsize=10, width=0.8, 
                    color='#cccccc', labelcolor='#666666')
    ax2.tick_params(axis='y', which='major', labelsize=10, width=0.8, 
                    color='#cccccc', labelcolor='#666666')
    
    # Set light colored spines
    for spine_ax in [ax1, ax2]:
        spine_ax.spines['left'].set_color('#cccccc')
        spine_ax.spines['bottom'].set_color('#cccccc')
        spine_ax.spines['top'].set_color('#cccccc')
        spine_ax.spines['right'].set_color('#cccccc')
        spine_ax.spines['left'].set_linewidth(0.8)
        spine_ax.spines['bottom'].set_linewidth(0.8)
        spine_ax.spines['top'].set_linewidth(0.8)
        spine_ax.spines['right'].set_linewidth(0.8)
    
    # Create combined legend in upper left
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    legend = ax1.legend(lines, labels,
                        frameon=True, 
                        fancybox=True,  # Enable rounded corners
                        edgecolor='none',  # No border
                        facecolor='#e8e8e8',  # Darker gray background
                        framealpha=0.95,
                        fontsize=8,
                        loc='upper left',
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
    attack_success_rate,
    refusal_score,
    save_path: Optional[str] = None
) -> None:
    """
    Main function to parse input and create attack metrics plot.
    
    Args:
        attack_success_rate: Comma-separated string or list of attack successful rate values
        refusal_score: Comma-separated string or list of refusal score values
        save_path: Optional path to save the plot
    
    Example:
        python draw_attack_metrics.py main --attack_success_rate="0.15,0.13,0.11,0.10,0.09" \
                                           --refusal_score="0.75,0.67,0.59,0.50,0.41" \
                                           --save_path="attack_metrics.pdf"
    """
    # Helper function to parse input data
    def parse_input(data):
        if isinstance(data, str):
            return [float(x.strip()) for x in data.split(',')]
        elif isinstance(data, (list, tuple)):
            return [float(x) for x in data]
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    # Parse input data to float lists
    try:
        attack_rate_list = parse_input(attack_success_rate)
        refusal_score_list = parse_input(refusal_score)
    except ValueError as e:
        raise ValueError(f"Error parsing input values: {e}")
    
    # Create the plot
    plot_attack_metrics(
        attack_success_rate=attack_rate_list,
        refusal_score=refusal_score_list,
        save_path=save_path
    )
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    fire.Fire(main) 