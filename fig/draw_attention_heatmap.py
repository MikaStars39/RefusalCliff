import torch
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Dict, Any
import json
import numpy as np
import fire
from matplotlib.colors import TwoSlopeNorm

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

def plot_attention_heatmap(
    data_path: str,  # Path to JSON file containing attention head data
    value_key: str = "cosine_similarity",  # Key name for the value to plot
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8),
    colormap: str = 'RdBu_r',
    show_values: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center_zero: bool = True,  # Whether to center colormap on zero
) -> None:
    """
    Plot attention head heatmap with academic style matching the programmer aesthetic.
    
    Args:
        data_path: Path to JSON file containing attention head data
        value_key: Key name for the value to plot (e.g., 'cosine_similarity', 'attention_score')
        save_path: Optional path to save the plot
        figsize: Figure size tuple (width, height)
        colormap: Colormap for the heatmap
        show_values: Whether to show values in each cell
        vmin: Minimum value for colormap normalization
        vmax: Maximum value for colormap normalization
        center_zero: Whether to center the colormap on zero (makes 0 white)
    """
    # Load data from JSON file
    with open(data_path, "r") as f:
        data = json.load(f)
    
    if not data:
        raise ValueError("No data found in the JSON file")
    
    # Extract layer_idx, head_idx, and values
    layer_indices = []
    head_indices = []
    values = []
    
    for item in data:
        if "layer_idx" not in item or "head_idx" not in item or value_key not in item:
            print(f"Warning: Missing required keys in data item: {item}")
            continue
        
        layer_indices.append(item["layer_idx"])
        head_indices.append(item["head_idx"])
        values.append(item[value_key])
    
    if not values:
        raise ValueError(f"No valid data found with key '{value_key}'")
    
    # Determine the dimensions of the heatmap
    max_layer = max(layer_indices)
    max_head = max(head_indices)
    min_layer = min(layer_indices)
    min_head = min(head_indices)
    
    # Create heatmap matrix
    n_layers = max_layer - min_layer + 1
    n_heads = max_head - min_head + 1
    heatmap_matrix = np.full((n_layers, n_heads), np.nan)
    
    # Fill the matrix with values
    for layer_idx, head_idx, value in zip(layer_indices, head_indices, values):
        row = layer_idx - min_layer
        col = head_idx - min_head
        heatmap_matrix[row, col] = value
    
    # Create figure with programmer styling
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')  # White background for figure
    
    # Set value limits for colormap if not provided
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)
    
    # Create normalization that centers on zero (makes 0 white)
    if center_zero:
        # Use TwoSlopeNorm to center the colormap on zero
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = None
    
    # Create the heatmap
    im = ax.imshow(heatmap_matrix, cmap=colormap, aspect='auto', 
                   origin='lower', norm=norm,
                   interpolation='nearest')
    
    # Configure axes
    ax.set_xlabel('Head Index', fontsize=16, fontweight='normal', color='#333333')
    ax.set_ylabel('Layer Index', fontsize=16, fontweight='normal', color='#333333')
    
    # Set ticks
    x_ticks = np.arange(0, n_heads, max(1, n_heads // 10))  # Show at most 10 x-ticks
    y_ticks = np.arange(0, n_layers, max(1, n_layers // 10))  # Show at most 10 y-ticks
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(min_head + tick) for tick in x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(min_layer + tick) for tick in y_ticks])
    
    # Customize ticks with light colors
    ax.tick_params(axis='both', which='major', labelsize=12, width=0.8, 
                   color='#cccccc', labelcolor='#666666')
    
    # Add values to cells if requested
    if show_values:
        for i in range(n_layers):
            for j in range(n_heads):
                if not np.isnan(heatmap_matrix[i, j]):
                    # Determine text color based on value relative to center (0)
                    if center_zero:
                        text_color = 'white' if abs(heatmap_matrix[i, j]) > max(abs(vmin), abs(vmax)) / 3 else 'black'
                    else:
                        text_color = 'white' if abs(heatmap_matrix[i, j] - (vmin + vmax) / 2) > (vmax - vmin) / 3 else 'black'
                    ax.text(j, i, f'{heatmap_matrix[i, j]:.3f}', 
                           ha='center', va='center', color=text_color, 
                           fontsize=8, fontweight='normal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.ax.tick_params(labelsize=12, width=0.8, color='#cccccc', labelcolor='#666666')
    
    # Set light colored spines for colorbar
    cbar.outline.set_color('#cccccc')
    cbar.outline.set_linewidth(0.8)
    
    # Set light colored spines for main plot
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        # Ensure PDF extension
        if save_path and not save_path.lower().endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', format='pdf')
        print(f"Attention heatmap saved to: {save_path}")
    
    plt.show()

def plot_attention_heatmap_filtered(
    data_path: str,  # Path to JSON file containing attention head data
    value_key: str = "cosine_similarity",  # Key name for the value to plot
    threshold: Optional[float] = None,  # Only show values above/below this threshold
    threshold_mode: str = "above",  # "above" or "below"
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8),
    colormap: str = 'RdBu_r',
    show_values: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center_zero: bool = True,  # Whether to center colormap on zero
) -> None:
    """
    Plot filtered attention head heatmap showing only values meeting threshold criteria.
    
    Args:
        data_path: Path to JSON file containing attention head data
        value_key: Key name for the value to plot
        threshold: Threshold value for filtering
        threshold_mode: "above" to show values >= threshold, "below" to show values <= threshold
        save_path: Optional path to save the plot
        figsize: Figure size tuple (width, height)
        colormap: Colormap for the heatmap
        show_values: Whether to show values in each cell
        vmin: Minimum value for colormap normalization
        vmax: Maximum value for colormap normalization
        center_zero: Whether to center the colormap on zero (makes 0 white)
    """
    # Load data from JSON file
    with open(data_path, "r") as f:
        data = json.load(f)
    
    if not data:
        raise ValueError("No data found in the JSON file")
    
    # Filter data based on threshold if provided
    if threshold is not None:
        if threshold_mode == "above":
            data = [item for item in data if item.get(value_key, float('-inf')) >= threshold]
        elif threshold_mode == "below":
            data = [item for item in data if item.get(value_key, float('inf')) <= threshold]
        else:
            raise ValueError("threshold_mode must be 'above' or 'below'")
    
    if not data:
        print(f"Warning: No data points meet the threshold criteria ({threshold_mode} {threshold})")
        return
    
    # Extract layer_idx, head_idx, and values
    layer_indices = []
    head_indices = []
    values = []
    
    for item in data:
        if "layer_idx" not in item or "head_idx" not in item or value_key not in item:
            continue
        
        layer_indices.append(item["layer_idx"])
        head_indices.append(item["head_idx"])
        values.append(item[value_key])
    
    # Determine the dimensions of the heatmap
    max_layer = max(layer_indices)
    max_head = max(head_indices)
    min_layer = min(layer_indices)
    min_head = min(head_indices)
    
    # Create heatmap matrix
    n_layers = max_layer - min_layer + 1
    n_heads = max_head - min_head + 1
    heatmap_matrix = np.full((n_layers, n_heads), np.nan)
    
    # Fill the matrix with values
    for layer_idx, head_idx, value in zip(layer_indices, head_indices, values):
        row = layer_idx - min_layer
        col = head_idx - min_head
        heatmap_matrix[row, col] = value
    
    # Create figure with programmer styling
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    
    # Set value limits for colormap if not provided
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)
    
    # Create normalization that centers on zero (makes 0 white)
    if center_zero:
        # Use TwoSlopeNorm to center the colormap on zero
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = None
    
    # Create the heatmap
    im = ax.imshow(heatmap_matrix, cmap=colormap, aspect='auto', 
                   origin='lower', norm=norm,
                   interpolation='nearest')
    
    # Configure axes
    ax.set_xlabel('Head Index', fontsize=12, fontweight='normal', color='#333333')
    ax.set_ylabel('Layer Index', fontsize=12, fontweight='normal', color='#333333')
    
    threshold_text = f" ({threshold_mode} {threshold})" if threshold is not None else ""
    ax.set_title(f'Filtered Attention Head {value_key.replace("_", " ").title()} Heatmap{threshold_text}', 
                fontsize=18, fontweight='normal', color='#333333', pad=20)
    
    # Set ticks to show full range even if filtered
    x_ticks = np.arange(0, n_heads, max(1, n_heads // 10))
    y_ticks = np.arange(0, n_layers, max(1, n_layers // 10))
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(min_head + tick) for tick in x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(min_layer + tick) for tick in y_ticks])
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=12, width=0.8, 
                   color='#cccccc', labelcolor='#666666')
    
    # Add values to cells if requested
    if show_values:
        for i in range(n_layers):
            for j in range(n_heads):
                if not np.isnan(heatmap_matrix[i, j]):
                    # Determine text color based on value relative to center (0)
                    if center_zero:
                        text_color = 'white' if abs(heatmap_matrix[i, j]) > max(abs(vmin), abs(vmax)) / 3 else 'black'
                    else:
                        text_color = 'white' if abs(heatmap_matrix[i, j] - (vmin + vmax) / 2) > (vmax - vmin) / 3 else 'black'
                    ax.text(j, i, f'{heatmap_matrix[i, j]:.3f}', 
                           ha='center', va='center', color=text_color, 
                           fontsize=8, fontweight='normal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.ax.tick_params(labelsize=12, width=0.8, color='#cccccc', labelcolor='#666666')
    cbar.outline.set_color('#cccccc')
    cbar.outline.set_linewidth(0.8)
    
    # Set light colored spines
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    
    if save_path:
        # Ensure PDF extension
        if save_path and not save_path.lower().endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', format='pdf')
        print(f"Filtered attention heatmap saved to: {save_path}")
    
    plt.show()

class AttentionHeatmapCLI:
    """Command line interface for attention heatmap plotting using Fire."""
    
    def plot(
        self,
        data_path: str,
        value_key: str = "cosine_similarity",
        save_path: str = None,
        width: int = 12,
        height: int = 8,
        colormap: str = 'RdBu_r',
        show_values: bool = False,
        vmin: float = None,
        vmax: float = None,
        center_zero: bool = True,
    ):
        """
        Plot attention head heatmap.
        
        Args:
            data_path: Path to JSON file containing attention head data
            value_key: Key name for the value to plot (default: cosine_similarity)
            save_path: Path to save the plot (optional)
            width: Figure width (default: 12)
            height: Figure height (default: 8)
            colormap: Colormap name (default: RdBu_r)
            show_values: Show values in cells (default: False)
            vmin: Minimum value for colormap (optional)
            vmax: Maximum value for colormap (optional)
            center_zero: Center colormap on zero to make 0 white (default: True)
        """
        plot_attention_heatmap(
            data_path=data_path,
            value_key=value_key,
            save_path=save_path,
            figsize=(width, height),
            colormap=colormap,
            show_values=show_values,
            vmin=vmin,
            vmax=vmax,
            center_zero=center_zero,
        )
    
    def plot_filtered(
        self,
        data_path: str,
        threshold: float,
        threshold_mode: str = "above",
        value_key: str = "cosine_similarity",
        save_path: str = None,
        width: int = 12,
        height: int = 8,
        colormap: str = 'RdBu_r',
        show_values: bool = False,
        vmin: float = None,
        vmax: float = None,
        center_zero: bool = True,
    ):
        """
        Plot filtered attention head heatmap.
        
        Args:
            data_path: Path to JSON file containing attention head data
            threshold: Threshold value for filtering
            threshold_mode: Filter mode - 'above' or 'below' (default: above)
            value_key: Key name for the value to plot (default: cosine_similarity)
            save_path: Path to save the plot (optional)
            width: Figure width (default: 12)
            height: Figure height (default: 8)
            colormap: Colormap name (default: RdBu_r)
            show_values: Show values in cells (default: False)
            vmin: Minimum value for colormap (optional)
            vmax: Maximum value for colormap (optional)
            center_zero: Center colormap on zero to make 0 white (default: True)
        """
        plot_attention_heatmap_filtered(
            data_path=data_path,
            value_key=value_key,
            threshold=threshold,
            threshold_mode=threshold_mode,
            save_path=save_path,
            figsize=(width, height),
            colormap=colormap,
            show_values=show_values,
            vmin=vmin,
            vmax=vmax,
            center_zero=center_zero,
        )
    
    def batch_plot(
        self,
        data_path: str,
        output_dir: str = "outputs/fig",
        value_key: str = "cosine_similarity",
        width: int = 14,
        height: int = 10,
        negative_threshold: float = -1.0,
        positive_threshold: float = 1.0,
        center_zero: bool = True,
    ):
        """
        Generate a batch of plots: full heatmap, negative filtered, positive filtered.
        
        Args:
            data_path: Path to JSON file containing attention head data
            output_dir: Directory to save plots (default: outputs/fig)
            value_key: Key name for the value to plot (default: cosine_similarity)
            width: Figure width (default: 14)
            height: Figure height (default: 10)
            negative_threshold: Threshold for negative filtering (default: -1.0)
            positive_threshold: Threshold for positive filtering (default: 1.0)
            center_zero: Center colormap on zero to make 0 white (default: True)
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(data_path))[0]
        
        print("Generating full attention heatmap...")
        plot_attention_heatmap(
            data_path=data_path,
            value_key=value_key,
            save_path=os.path.join(output_dir, f"{base_name}_heatmap_full.pdf"),
            figsize=(width, height),
            colormap='RdBu_r',
            show_values=False,
            center_zero=center_zero
        )
        
        print("Generating negative filtered heatmap...")
        plot_attention_heatmap_filtered(
            data_path=data_path,
            value_key=value_key,
            threshold=negative_threshold,
            threshold_mode="below",
            save_path=os.path.join(output_dir, f"{base_name}_heatmap_negative.pdf"),
            figsize=(width, height),
            colormap='Blues_r',
            show_values=True,
            center_zero=False  # For filtered plots, don't center on zero
        )
        
        print("Generating positive filtered heatmap...")
        plot_attention_heatmap_filtered(
            data_path=data_path,
            value_key=value_key,
            threshold=positive_threshold,
            threshold_mode="above",
            save_path=os.path.join(output_dir, f"{base_name}_heatmap_positive.pdf"),
            figsize=(width, height),
            colormap='Reds',
            show_values=True,
            center_zero=False  # For filtered plots, don't center on zero
        )
        
        print(f"All plots saved to: {output_dir}")

if __name__ == "__main__":
    fire.Fire(AttentionHeatmapCLI) 