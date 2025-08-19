import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from fire import Fire

def visualize_dual_conceptors_2d_3d(C_neg: torch.Tensor, C_pos: torch.Tensor, output_path: str = "outputs/dual_conceptors_2d_3d.png"):
    """
    Visualize two conceptor matrices in both 2D and 3D views side by side.
    
    Args:
        C_neg (torch.Tensor): Negative conceptor matrix, shape (hidden_dim, hidden_dim)
        C_pos (torch.Tensor): Positive conceptor matrix, shape (hidden_dim, hidden_dim)
        output_path (str): Path to save the visualization
    """
    print("\nPerforming PCA on both conceptor matrices for 2D and 3D visualization...")

    # Convert tensors to NumPy arrays for sklearn PCA
    C_neg_numpy = C_neg.cpu().numpy()
    C_pos_numpy = C_pos.cpu().numpy()

    # Fit PCA on the concatenated data to ensure same transformation space
    combined_data = np.vstack([C_neg_numpy, C_pos_numpy])
    
    # PCA for 2D visualization
    pca_2d = PCA(n_components=2)
    pca_2d.fit(combined_data)
    conceptor_neg_pca_2d = pca_2d.transform(C_neg_numpy)
    conceptor_pos_pca_2d = pca_2d.transform(C_pos_numpy)
    
    # PCA for 3D visualization
    pca_3d = PCA(n_components=3)
    pca_3d.fit(combined_data)
    conceptor_neg_pca_3d = pca_3d.transform(C_neg_numpy)
    conceptor_pos_pca_3d = pca_3d.transform(C_pos_numpy)

    print(f"2D PCA complete. Negative: {conceptor_neg_pca_2d.shape}, Positive: {conceptor_pos_pca_2d.shape}")
    print(f"3D PCA complete. Negative: {conceptor_neg_pca_3d.shape}, Positive: {conceptor_pos_pca_3d.shape}")
    print("Generating combined 2D and 3D plot...")

    # Create figure with side-by-side subplots
    fig = plt.figure(figsize=(20, 9))
    
    # ============================================================================
    # 2D Plot (Left subplot)
    # ============================================================================
    ax1 = fig.add_subplot(121)
    
    # Plot 2D negative conceptor
    scatter_neg_2d = ax1.scatter(
        conceptor_neg_pca_2d[:, 0], 
        conceptor_neg_pca_2d[:, 1], 
        alpha=0.8, 
        c='red', 
        s=80,
        label='Negative Conceptor',
        marker='o',
        edgecolors='darkred',
        linewidth=0.8
    )
    
    # Plot 2D positive conceptor
    scatter_pos_2d = ax1.scatter(
        conceptor_pos_pca_2d[:, 0], 
        conceptor_pos_pca_2d[:, 1], 
        alpha=0.8, 
        c='blue', 
        s=80,
        label='Positive Conceptor',
        marker='^',
        edgecolors='darkblue',
        linewidth=0.8
    )
    
    ax1.set_title('2D PCA Visualization\nNegative vs Positive Conceptors', fontsize=16, pad=20)
    ax1.set_xlabel(f'Principal Component 1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax1.set_ylabel(f'Principal Component 2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add total variance explained
    total_variance_2d = pca_2d.explained_variance_ratio_[:2].sum()
    ax1.text(0.02, 0.98, f'Total Variance Explained: {total_variance_2d:.1%}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ============================================================================
    # 3D Plot (Right subplot)
    # ============================================================================
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot 3D negative conceptor
    scatter_neg_3d = ax2.scatter(
        conceptor_neg_pca_3d[:, 0], 
        conceptor_neg_pca_3d[:, 1], 
        conceptor_neg_pca_3d[:, 2], 
        alpha=0.8, 
        c='red', 
        s=60,
        label='Negative Conceptor',
        marker='o',
        edgecolors='darkred',
        linewidth=0.5
    )
    
    # Plot 3D positive conceptor
    scatter_pos_3d = ax2.scatter(
        conceptor_pos_pca_3d[:, 0], 
        conceptor_pos_pca_3d[:, 1], 
        conceptor_pos_pca_3d[:, 2], 
        alpha=0.8, 
        c='blue', 
        s=60,
        label='Positive Conceptor',
        marker='^',
        edgecolors='darkblue',
        linewidth=0.5
    )
    
    ax2.set_title('3D PCA Visualization\nNegative vs Positive Conceptors', fontsize=16, pad=20)
    ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})', fontsize=11)
    ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    # Enhance 3D appearance
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.xaxis.pane.set_alpha(0.1)
    ax2.yaxis.pane.set_alpha(0.1)
    ax2.zaxis.pane.set_alpha(0.1)
    
    # Add total variance explained for 3D
    total_variance_3d = pca_3d.explained_variance_ratio_[:3].sum()
    ax2.text2D(0.02, 0.02, f'Total Variance: {total_variance_3d:.1%}', 
               transform=ax2.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ============================================================================
    # Final adjustments
    # ============================================================================
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined 2D and 3D visualization saved to: {output_path}")
    
    return fig, (ax1, ax2)

def visualize_conceptor_2d_3d(C: torch.Tensor, output_path: str = "outputs/conceptor_2d_3d.png"):
    """
    Visualize a single conceptor matrix in both 2D and 3D views side by side.
    
    Args:
        C (torch.Tensor): Conceptor matrix, shape (hidden_dim, hidden_dim)
        output_path (str): Path to save the visualization
    """
    print("\nPerforming PCA on conceptor matrix for 2D and 3D visualization...")

    C_numpy = C.cpu().numpy()
    
    # PCA for 2D
    pca_2d = PCA(n_components=2)
    conceptor_pca_2d = pca_2d.fit_transform(C_numpy)
    
    # PCA for 3D
    pca_3d = PCA(n_components=3)
    conceptor_pca_3d = pca_3d.fit_transform(C_numpy)

    print(f"2D PCA complete. Shape: {conceptor_pca_2d.shape}")
    print(f"3D PCA complete. Shape: {conceptor_pca_3d.shape}")
    print("Generating combined 2D and 3D plot...")

    # Create figure with side-by-side subplots
    fig = plt.figure(figsize=(20, 9))
    
    # ============================================================================
    # 2D Plot (Left subplot)
    # ============================================================================
    ax1 = fig.add_subplot(121)
    
    scatter_2d = ax1.scatter(
        conceptor_pca_2d[:, 0], 
        conceptor_pca_2d[:, 1], 
        alpha=0.7, 
        c=conceptor_pca_2d[:, 0], 
        cmap='viridis', 
        s=80,
        edgecolors='black',
        linewidth=0.5
    )
    
    ax1.set_title('2D PCA Visualization\nConceptor Matrix', fontsize=16, pad=20)
    ax1.set_xlabel(f'Principal Component 1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax1.set_ylabel(f'Principal Component 2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for 2D plot
    cbar_2d = plt.colorbar(scatter_2d, ax=ax1, shrink=0.8)
    cbar_2d.set_label('PC1 Value', fontsize=10)
    
    # Add variance info
    total_variance_2d = pca_2d.explained_variance_ratio_[:2].sum()
    ax1.text(0.02, 0.98, f'Total Variance Explained: {total_variance_2d:.1%}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ============================================================================
    # 3D Plot (Right subplot)
    # ============================================================================
    ax2 = fig.add_subplot(122, projection='3d')
    
    scatter_3d = ax2.scatter(
        conceptor_pca_3d[:, 0], 
        conceptor_pca_3d[:, 1], 
        conceptor_pca_3d[:, 2], 
        alpha=0.7, 
        c=conceptor_pca_3d[:, 0], 
        cmap='viridis', 
        s=50
    )
    
    ax2.set_title('3D PCA Visualization\nConceptor Matrix', fontsize=16, pad=20)
    ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})', fontsize=11)
    
    # Enhance 3D appearance
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.xaxis.pane.set_alpha(0.1)
    ax2.yaxis.pane.set_alpha(0.1)
    ax2.zaxis.pane.set_alpha(0.1)
    
    # Add colorbar for 3D plot
    cbar_3d = plt.colorbar(scatter_3d, ax=ax2, shrink=0.5, aspect=10)
    cbar_3d.set_label('PC1 Value', fontsize=10)
    
    # Add variance info
    total_variance_3d = pca_3d.explained_variance_ratio_[:3].sum()
    ax2.text2D(0.02, 0.02, f'Total Variance: {total_variance_3d:.1%}', 
               transform=ax2.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined 2D and 3D visualization saved to: {output_path}")
    
    return fig, (ax1, ax2)

def visualize_dual_activations_2d_3d(activations_neg: torch.Tensor, activations_pos: torch.Tensor, output_path: str = "outputs/dual_activations_2d_3d.png"):
    """
    Visualize raw activation vectors (without conceptor computation) in both 2D and 3D views side by side.
    
    Args:
        activations_neg (torch.Tensor): Negative activation vectors, shape (n_samples, hidden_dim)
        activations_pos (torch.Tensor): Positive activation vectors, shape (n_samples, hidden_dim)
        output_path (str): Path to save the visualization
    """
    print("\nPerforming PCA on raw activation vectors for 2D and 3D visualization...")

    # Convert tensors to NumPy arrays for sklearn PCA
    activations_neg_numpy = activations_neg.cpu().numpy()
    activations_pos_numpy = activations_pos.cpu().numpy()

    # Fit PCA on the concatenated data to ensure same transformation space
    combined_data = np.vstack([activations_neg_numpy, activations_pos_numpy])
    
    # PCA for 2D visualization
    pca_2d = PCA(n_components=2)
    pca_2d.fit(combined_data)
    activations_neg_pca_2d = pca_2d.transform(activations_neg_numpy)
    activations_pos_pca_2d = pca_2d.transform(activations_pos_numpy)
    
    # PCA for 3D visualization
    pca_3d = PCA(n_components=3)
    pca_3d.fit(combined_data)
    activations_neg_pca_3d = pca_3d.transform(activations_neg_numpy)
    activations_pos_pca_3d = pca_3d.transform(activations_pos_numpy)

    print(f"2D PCA complete. Negative: {activations_neg_pca_2d.shape}, Positive: {activations_pos_pca_2d.shape}")
    print(f"3D PCA complete. Negative: {activations_neg_pca_3d.shape}, Positive: {activations_pos_pca_3d.shape}")
    print("Generating combined 2D and 3D plot for raw activations...")

    # Create figure with side-by-side subplots
    fig = plt.figure(figsize=(20, 9))
    
    # ============================================================================
    # 2D Plot (Left subplot)
    # ============================================================================
    ax1 = fig.add_subplot(121)
    
    # Plot 2D negative activations
    scatter_neg_2d = ax1.scatter(
        activations_neg_pca_2d[:, 0], 
        activations_neg_pca_2d[:, 1], 
        alpha=0.7, 
        c='red', 
        s=60,
        label='Negative Activations',
        marker='o',
        edgecolors='darkred',
        linewidth=0.5
    )
    
    # Plot 2D positive activations
    scatter_pos_2d = ax1.scatter(
        activations_pos_pca_2d[:, 0], 
        activations_pos_pca_2d[:, 1], 
        alpha=0.7, 
        c='blue', 
        s=60,
        label='Positive Activations',
        marker='^',
        edgecolors='darkblue',
        linewidth=0.5
    )
    
    ax1.set_title('2D PCA Visualization\nRaw Activation Vectors (Negative vs Positive)', fontsize=16, pad=20)
    ax1.set_xlabel(f'Principal Component 1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax1.set_ylabel(f'Principal Component 2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add total variance explained and sample counts
    total_variance_2d = pca_2d.explained_variance_ratio_[:2].sum()
    ax1.text(0.02, 0.98, f'Total Variance Explained: {total_variance_2d:.1%}\nNeg samples: {len(activations_neg_numpy)}, Pos samples: {len(activations_pos_numpy)}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ============================================================================
    # 3D Plot (Right subplot)
    # ============================================================================
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot 3D negative activations
    scatter_neg_3d = ax2.scatter(
        activations_neg_pca_3d[:, 0], 
        activations_neg_pca_3d[:, 1], 
        activations_neg_pca_3d[:, 2], 
        alpha=0.7, 
        c='red', 
        s=40,
        label='Negative Activations',
        marker='o',
        edgecolors='darkred',
        linewidth=0.3
    )
    
    # Plot 3D positive activations
    scatter_pos_3d = ax2.scatter(
        activations_pos_pca_3d[:, 0], 
        activations_pos_pca_3d[:, 1], 
        activations_pos_pca_3d[:, 2], 
        alpha=0.7, 
        c='blue', 
        s=40,
        label='Positive Activations',
        marker='^',
        edgecolors='darkblue',
        linewidth=0.3
    )
    
    ax2.set_title('3D PCA Visualization\nRaw Activation Vectors (Negative vs Positive)', fontsize=16, pad=20)
    ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})', fontsize=11)
    ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    # Enhance 3D appearance
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.xaxis.pane.set_alpha(0.1)
    ax2.yaxis.pane.set_alpha(0.1)
    ax2.zaxis.pane.set_alpha(0.1)
    
    # Add total variance explained for 3D
    total_variance_3d = pca_3d.explained_variance_ratio_[:3].sum()
    ax2.text2D(0.02, 0.02, f'Total Variance: {total_variance_3d:.1%}', 
               transform=ax2.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ============================================================================
    # Final adjustments
    # ============================================================================
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined 2D and 3D raw activation visualization saved to: {output_path}")
    
    return fig, (ax1, ax2)

def visualize_single_activations_2d_3d(activations: torch.Tensor, output_path: str = "outputs/single_activations_2d_3d.png"):
    """
    Visualize single set of raw activation vectors in both 2D and 3D views side by side.
    
    Args:
        activations (torch.Tensor): Activation vectors, shape (n_samples, hidden_dim)
        output_path (str): Path to save the visualization
    """
    print("\nPerforming PCA on raw activation vectors for 2D and 3D visualization...")

    activations_numpy = activations.cpu().numpy()
    
    # PCA for 2D
    pca_2d = PCA(n_components=2)
    activations_pca_2d = pca_2d.fit_transform(activations_numpy)
    
    # PCA for 3D
    pca_3d = PCA(n_components=3)
    activations_pca_3d = pca_3d.fit_transform(activations_numpy)

    print(f"2D PCA complete. Shape: {activations_pca_2d.shape}")
    print(f"3D PCA complete. Shape: {activations_pca_3d.shape}")
    print("Generating combined 2D and 3D plot for raw activations...")

    # Create figure with side-by-side subplots
    fig = plt.figure(figsize=(20, 9))
    
    # ============================================================================
    # 2D Plot (Left subplot)
    # ============================================================================
    ax1 = fig.add_subplot(121)
    
    scatter_2d = ax1.scatter(
        activations_pca_2d[:, 0], 
        activations_pca_2d[:, 1], 
        alpha=0.7, 
        c=activations_pca_2d[:, 0], 
        cmap='viridis', 
        s=60,
        edgecolors='black',
        linewidth=0.3
    )
    
    ax1.set_title('2D PCA Visualization\nRaw Activation Vectors', fontsize=16, pad=20)
    ax1.set_xlabel(f'Principal Component 1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax1.set_ylabel(f'Principal Component 2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for 2D plot
    cbar_2d = plt.colorbar(scatter_2d, ax=ax1, shrink=0.8)
    cbar_2d.set_label('PC1 Value', fontsize=10)
    
    # Add variance info and sample count
    total_variance_2d = pca_2d.explained_variance_ratio_[:2].sum()
    ax1.text(0.02, 0.98, f'Total Variance Explained: {total_variance_2d:.1%}\nSamples: {len(activations_numpy)}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ============================================================================
    # 3D Plot (Right subplot)
    # ============================================================================
    ax2 = fig.add_subplot(122, projection='3d')
    
    scatter_3d = ax2.scatter(
        activations_pca_3d[:, 0], 
        activations_pca_3d[:, 1], 
        activations_pca_3d[:, 2], 
        alpha=0.7, 
        c=activations_pca_3d[:, 0], 
        cmap='viridis', 
        s=40
    )
    
    ax2.set_title('3D PCA Visualization\nRaw Activation Vectors', fontsize=16, pad=20)
    ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})', fontsize=11)
    
    # Enhance 3D appearance
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.xaxis.pane.set_alpha(0.1)
    ax2.yaxis.pane.set_alpha(0.1)
    ax2.zaxis.pane.set_alpha(0.1)
    
    # Add colorbar for 3D plot
    cbar_3d = plt.colorbar(scatter_3d, ax=ax2, shrink=0.5, aspect=10)
    cbar_3d.set_label('PC1 Value', fontsize=10)
    
    # Add variance info
    total_variance_3d = pca_3d.explained_variance_ratio_[:3].sum()
    ax2.text2D(0.02, 0.02, f'Total Variance: {total_variance_3d:.1%}', 
               transform=ax2.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined 2D and 3D raw activation visualization saved to: {output_path}")
    
    return fig, (ax1, ax2)

def compute_conceptor(X: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Computes a Conceptor matrix from a set of activation vectors.
    Source: Postmus & Abreu, 2025 [cite: 73, 74]

    Args:
        X (torch.Tensor): The activation matrix, shape (n_samples, hidden_dim).
        alpha (float): The aperture hyperparameter.

    Returns:
        torch.Tensor: The computed Conceptor matrix, shape (hidden_dim, hidden_dim).
    """
    if not X.is_floating_point():
        X = X.to(torch.float32)
    n_samples, hidden_dim = X.shape
    device = X.device
    R = (X.T @ X) / n_samples
    identity_matrix = torch.eye(hidden_dim, device=device, dtype=R.dtype)
    term_to_invert = R + (alpha**-2) * identity_matrix
    inverted_term = torch.inverse(term_to_invert)
    C = R @ inverted_term
    return C

def main(
    test_path: str = "outputs/evil_vectors.pkl",
    output_path: str = "outputs/dual_conceptors_2d_3d.png",
    alpha: float = 0.1,
    layer: int = 15,
    skip_conceptor: bool = False,
):
    """
    Main function to compute and visualize conceptors or raw activations in both 2D and 3D.
    
    Args:
        test_path (str): Path to the activation data pickle file
        output_path (str): Path to save the visualization
        alpha (float): Aperture hyperparameter for conceptor computation (ignored if skip_conceptor=True)
        layer (int): Which layer to use for visualization
        skip_conceptor (bool): If True, skip conceptor computation and directly visualize raw activations
    """
    
    # Load activation data - now expecting a list with [negative, positive] elements
    activation_data = pickle.load(open(test_path, "rb"))
    
    # Check if the data structure is a list containing negative and positive activations
    if isinstance(activation_data, list) and len(activation_data) == 2:
        print("Detected dual activation structure (negative and positive)")
        activation_neg, activation_pos = activation_data
        
        # Process negative activations for the specified layer
        activation_matrix_neg = activation_neg[layer].reshape(-1, activation_neg[layer].shape[-1]).float()
        print(f"Negative activation vectors shape: {activation_matrix_neg.shape}")
        
        # Process positive activations for the specified layer  
        activation_matrix_pos = activation_pos[layer].reshape(-1, activation_pos[layer].shape[-1]).float()
        print(f"Positive activation vectors shape: {activation_matrix_pos.shape}")
        
        # Move data to GPU if available
        if torch.cuda.is_available():
            activation_matrix_neg = activation_matrix_neg.to('cuda')
            activation_matrix_pos = activation_matrix_pos.to('cuda')
            print("Data moved to CUDA device.")

        if skip_conceptor:
            print("\n" + "="*70)
            print("SKIP CONCEPTOR MODE: Visualizing raw activation vectors directly")
            print("="*70)
            
            # Adjust output path for raw activations
            raw_output_path = output_path.replace("conceptors", "raw_activations")
            visualize_dual_activations_2d_3d(activation_matrix_neg, activation_matrix_pos, raw_output_path)
            
        else:
            # =========================================================================
            # Compute Both Conceptors
            # =========================================================================

            aperture_alpha = alpha
            print(f"\nComputing both Conceptors with alpha = {aperture_alpha}...")

            try:
                # Compute negative conceptor
                conceptor_matrix_neg = compute_conceptor(activation_matrix_neg, aperture_alpha)
                print(f"Negative conceptor matrix shape: {conceptor_matrix_neg.shape}")
                
                # Compute positive conceptor
                conceptor_matrix_pos = compute_conceptor(activation_matrix_pos, aperture_alpha)
                print(f"Positive conceptor matrix shape: {conceptor_matrix_pos.shape}")

                print("\nBoth computations successful!")

                # Visualize both conceptors in 2D and 3D side by side
                visualize_dual_conceptors_2d_3d(conceptor_matrix_neg, conceptor_matrix_pos, output_path)

            except torch.linalg.LinAlgError as e:
                print(f"\nComputation failed: Linear algebra error occurred.")
                print("This usually means the correlation matrix R is singular or ill-conditioned. Try increasing the sample size or adjusting alpha.")
                print(f"Error message: {e}")
    
    else:
        # Fallback to original single activation processing
        print("Using original single activation structure")
        activation_matrix_X = activation_data[layer].reshape(-1, activation_data[layer].shape[-1]).float()
        print(f"Activation vectors shape: {activation_matrix_X.shape}")

        # Move simulated data to GPU if available
        if torch.cuda.is_available():
            activation_matrix_X = activation_matrix_X.to('cuda')
            print("Data moved to CUDA device.")

        if skip_conceptor:
            print("\n" + "="*70)
            print("SKIP CONCEPTOR MODE: Visualizing raw activation vectors directly")
            print("="*70)
            
            # Adjust output path for raw activations
            raw_output_path = output_path.replace("conceptors", "raw_activations")
            visualize_single_activations_2d_3d(activation_matrix_X, raw_output_path)
            
        else:
            # Compute single conceptor
            aperture_alpha = alpha
            print(f"\nComputing Conceptor with alpha = {aperture_alpha}...")

            try:
                conceptor_matrix_C = compute_conceptor(activation_matrix_X, aperture_alpha)
                print("\nComputation successful!")
                print(f"Conceptor matrix shape: {conceptor_matrix_C.shape}")

                # Use 2D and 3D combined visualization
                visualize_conceptor_2d_3d(conceptor_matrix_C, output_path.replace("dual_conceptors", "conceptor"))

            except torch.linalg.LinAlgError as e:
                print(f"\nComputation failed: Linear algebra error occurred.")
                print("This usually means the correlation matrix R is singular or ill-conditioned. Try increasing the sample size or adjusting alpha.")
                print(f"Error message: {e}")

if __name__ == '__main__':
    Fire(main)