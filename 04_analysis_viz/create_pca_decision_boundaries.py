#!/usr/bin/env python3
"""
Create PCA visualizations showing decision boundaries for different ML models.
Similar to the scikit-learn classifier comparison visualization.
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from pathlib import Path

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

def create_pca_decision_boundary_plots(data_dir, output_dir):
    """Create PCA plots showing decision boundaries for different models."""
    
    # Load PCA data
    with open(f'{data_dir}/pca_visualization_data.pkl', 'rb') as f:
        pca_data = pickle.load(f)
    
    # Load original data for labels
    with open(f'{data_dir}/train_samples_fixed.csv', 'r') as f:
        train_df = pd.read_csv(f)
    
    # Define colors
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#d62728', '#2ca02c'])  # Red for non-responder, Green for responder
    
    # Models to visualize (matching the example)
    models = {
        'Input Data': None,
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss'),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
        'RBF SVM': SVC(kernel='rbf', gamma='scale', probability=True, random_state=42)
    }
    
    # Process each modality
    for modality in ['protein', 'methylation']:
        if modality not in pca_data:
            print(f"No PCA data for {modality}")
            continue
            
        print(f"\nProcessing {modality}...")
        
        # Get data
        mod_data = pca_data[modality]
        X_subset = mod_data['X_raw']
        y_subset = mod_data['y_labels']
        n_features = mod_data['n_features']
        
        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(StandardScaler().fit_transform(X_subset))
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{modality.capitalize()} Expression - PCA Decision Boundaries\n'
                    f'({n_features} features, {len(y_subset)} samples)', fontsize=16)
        
        # Create mesh for decision boundaries
        h = 0.1  # step size (increased from 0.02 to speed up)
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Plot each model
        for idx, (name, model) in enumerate(models.items()):
            ax = axes[idx // 3, idx % 3]
            
            if model is None:
                # Just plot the data
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_subset, 
                                   cmap=cm_bright, edgecolor='k', s=20, alpha=0.8)
            else:
                # Train model on PCA data
                model.fit(X_pca, y_subset)
                
                # Get predictions on mesh
                Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
                Z = Z.reshape(xx.shape)
                
                # Plot decision boundary
                contour = ax.contourf(xx, yy, Z, levels=10, cmap=cm, alpha=0.8)
                ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
                
                # Plot data points
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_subset, 
                                   cmap=cm_bright, edgecolor='k', s=20)
            
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            
            # Add accuracy if model exists
            if model is not None:
                accuracy = model.score(X_pca, y_subset)
                ax.text(0.02, 0.98, f'Acc: {accuracy:.2f}', 
                       transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'{output_dir}/pca_decision_boundaries_{modality}.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/pca_decision_boundaries_{modality}.pdf', 
                   bbox_inches='tight')
        plt.close()
        
        print(f"  Created PCA visualization for {modality}")
        print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, "
              f"PC2={pca.explained_variance_ratio_[1]:.1%}")
        print(f"  Total: {pca.explained_variance_ratio_.sum():.1%}")


def create_combined_pca_figure(data_dir, output_dir):
    """Create a combined figure with all modalities if data exists."""
    
    # Load PCA data
    with open(f'{data_dir}/pca_visualization_data.pkl', 'rb') as f:
        pca_data = pickle.load(f)
    
    # Check available modalities
    available_modalities = [m for m in ['expression', 'protein', 'methylation', 'mutation'] 
                           if m in pca_data]
    
    if len(available_modalities) < 2:
        print("Not enough modalities for combined figure")
        return
    
    # Create figure with subplots for each modality
    n_modalities = len(available_modalities)
    fig, axes = plt.subplots(1, n_modalities, figsize=(5*n_modalities, 5))
    if n_modalities == 1:
        axes = [axes]
    
    fig.suptitle('PCA Visualization Across Data Modalities', fontsize=16, y=1.05)
    
    # Colors
    colors = ['#d62728', '#2ca02c']  # Red for non-responder, Green for responder
    
    for idx, modality in enumerate(available_modalities):
        ax = axes[idx]
        
        # Get data
        mod_data = pca_data[modality]
        X_subset = mod_data['X_raw']
        y_subset = mod_data['y_labels']
        
        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(StandardScaler().fit_transform(X_subset))
        
        # Plot
        for class_val in [0, 1]:
            mask = y_subset == class_val
            label = 'Responder' if class_val == 1 else 'Non-responder'
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      c=colors[class_val], label=label,
                      alpha=0.6, edgecolor='k', linewidth=0.5, s=30)
        
        ax.set_title(f'{modality.capitalize()}\n({mod_data["n_features"]} features)',
                    fontsize=12)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        
        if idx == 0:
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(f'{output_dir}/pca_all_modalities.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/pca_all_modalities.pdf', bbox_inches='tight')
    plt.close()
    
    print("\nCreated combined PCA figure for all modalities")


def main():
    """Generate PCA decision boundary visualizations."""
    # Setup paths
    data_dir = '/Users/tobyliu/bladder'
    output_dir = Path(data_dir) / '04_analysis_viz' / 'pca_visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating PCA decision boundary visualizations...")
    print("=" * 60)
    
    # Create decision boundary plots
    create_pca_decision_boundary_plots(data_dir, output_dir)
    
    # Create combined simple PCA
    print("\nCreating combined PCA visualization...")
    create_combined_pca_figure(data_dir, output_dir)
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()