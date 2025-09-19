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

def plot_validation_curves(
    val_accuracy: List[float],
    val_loss: List[float],
    ood_val_accuracy: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (5, 4),
    linewidth: float = 2.0,
    title: str = "Validation Metrics Over Epochs"
) -> None:
    """
    Plot validation accuracy and loss curves.
    
    Args:
        val_accuracy: List of validation accuracy values for each epoch
        val_loss: List of validation loss values for each epoch
        ood_val_accuracy: List of OOD validation accuracy values for each epoch
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        linewidth: Width of the plot lines
        title: Plot title
    """
    # Validate input lengths
    epochs = len(val_accuracy)
    if not all(len(lst) == epochs for lst in [val_loss, ood_val_accuracy]):
        raise ValueError("All metric lists must have the same length")
    
    # Create epoch range (0-indexed)
    epoch_range = np.arange(0, epochs)
    
    # Get retro colors from matplotlib's current color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Create single plot
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')  # White background for figure
    ax.set_facecolor('#f8f8f8')  # Light gray background for plot area
    
    # Create dual y-axis
    ax2 = ax.twinx()
    
    # Plot accuracy curves on left y-axis (solid lines)
    line1 = ax.plot(epoch_range, val_accuracy, 
                    label='Validation Accuracy', 
                    color=colors[0], 
                    linewidth=linewidth,
                    linestyle='-',  # solid line
                    alpha=0.8,
                    marker='o',
                    markersize=6,
                    markerfacecolor=colors[0],
                    markeredgecolor='black',
                    markeredgewidth=0.8)
    
    line2 = ax.plot(epoch_range, ood_val_accuracy, 
                    label='OOD Validation Accuracy', 
                    color=colors[1], 
                    linewidth=linewidth,
                    linestyle='-',  # solid line
                    alpha=0.8,
                    marker='s',
                    markersize=6,
                    markerfacecolor=colors[1],
                    markeredgecolor='black',
                    markeredgewidth=0.8)
    
    # Plot loss curve on right y-axis (dashed line)
    line3 = ax2.plot(epoch_range, val_loss, 
                     label='Validation Loss', 
                     color=colors[2], 
                     linewidth=linewidth,
                     linestyle='--',  # dashed line
                     alpha=0.8,
                     marker='^',
                     markersize=6,
                     markerfacecolor=colors[2],
                     markeredgecolor='black',
                     markeredgewidth=0.8)
    
    # Configure axes
    ax.set_xlabel('Epoch', fontsize=10, fontweight='normal', color='#333333')
    ax.set_ylabel('Accuracy', fontsize=10, fontweight='normal', color='#333333')
    ax2.set_ylabel('Loss', fontsize=10, fontweight='normal', color='#333333')
    
    # Set axis limits and ticks
    ax.set_xlim(-0.5, epochs - 0.5)
    ax.set_xticks(epoch_range)
    
    # Set accuracy y-axis limits (-0.1 to 1.1)
    ax.set_ylim(-0.1, 1.1)
    
    # Set loss y-axis limits (-0.1 to 1.1)
    ax2.set_ylim(-0.1, 1.1)
    
    # Add grid
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='#dddddd')
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, 
                   color='#cccccc', labelcolor='#666666')
    ax2.tick_params(axis='y', which='major', labelsize=10, width=0.8, 
                    color='#cccccc', labelcolor='#666666')
    
    # Set light colored spines
    for spine_ax in [ax, ax2]:
        spine_ax.spines['left'].set_color('#cccccc')
        spine_ax.spines['bottom'].set_color('#cccccc')
        spine_ax.spines['top'].set_color('#cccccc')
        spine_ax.spines['right'].set_color('#cccccc')
        spine_ax.spines['left'].set_linewidth(0.8)
        spine_ax.spines['bottom'].set_linewidth(0.8)
        spine_ax.spines['top'].set_linewidth(0.8)
        spine_ax.spines['right'].set_linewidth(0.8)
    
    # Create combined legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    legend = ax.legend(lines, labels,
                       frameon=True, 
                       fancybox=True,  # Enable rounded corners
                       edgecolor='none',  # No border
                       facecolor='#e8e8e8',  # Darker gray background
                       framealpha=0.95,
                       fontsize=8,
                       loc='center right',
                       borderpad=1.0,  # Increase padding between text and legend border
                       handletextpad=0.8,  # Space between legend markers and text
                       columnspacing=1.0)  # Space between columns if multiple
    legend.get_frame().set_linewidth(0)
    
    # Add title with smaller, non-bold font
    ax.set_title(title, fontsize=10, fontweight='normal', color='#333333', pad=15)
    
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
    val_accuracy,
    val_loss,
    ood_val_accuracy,
    save_path: Optional[str] = None,
    title: str = "Validation Metrics Over Epochs"
) -> None:
    """
    Main function to parse input and create validation curves plot.
    
    Args:
        val_accuracy: Comma-separated string or list of validation accuracy values
        val_loss: Comma-separated string or list of validation loss values
        ood_val_accuracy: Comma-separated string or list of OOD validation accuracy values
        save_path: Optional path to save the plot
        title: Plot title
    
    Example:
        python draw_validation_curves.py main --val_accuracy="0.85,0.87,0.89,0.90,0.91" \
                                              --val_loss="0.45,0.42,0.38,0.35,0.33" \
                                              --ood_val_accuracy="0.82,0.84,0.85,0.86,0.87" \
                                              --save_path="validation_curves.pdf"
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
        val_acc_list = parse_input(val_accuracy)
        val_loss_list = parse_input(val_loss)
        ood_val_acc_list = parse_input(ood_val_accuracy)
    except ValueError as e:
        raise ValueError(f"Error parsing input values: {e}")
    
    # Create the plot
    plot_validation_curves(
        val_accuracy=val_acc_list,
        val_loss=val_loss_list,
        ood_val_accuracy=ood_val_acc_list,
        save_path=save_path,
        title=title
    )
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    fire.Fire(main) 