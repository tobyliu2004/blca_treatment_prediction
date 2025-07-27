#!/usr/bin/env python3
"""
Create model performance heatmaps for bladder cancer prediction analysis.
Generates two heatmaps:
1. Individual model performance across modalities
2. Fusion method performance across strategies
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11

def create_model_performance_heatmap(data_dir, output_dir):
    """Create heatmap showing individual model performance across modalities."""
    # Load data
    with open(f'{data_dir}/individual_model_results.json', 'r') as f:
        results = json.load(f)
    
    model_matrix = results['model_performance_matrix']
    
    # Convert to DataFrame
    df = pd.DataFrame(model_matrix).T
    
    # Reorder columns for better presentation
    column_order = ['Expression', 'Methylation', 'Mutation', 'Protein']
    df.columns = ['Expression', 'Protein', 'Methylation', 'Mutation']
    df = df[column_order]
    
    # Reorder rows
    row_labels = {
        'logistic': 'Logistic Regression',
        'rf': 'Random Forest',
        'xgboost': 'XGBoost',
        'elasticnet': 'ElasticNet',
        'mlp': 'Neural Network (MLP)'
    }
    df.index = [row_labels[idx] for idx in df.index]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                center=0.60, vmin=0.45, vmax=0.70,
                cbar_kws={'label': 'AUC Score'},
                linewidths=0.5, linecolor='gray',
                ax=ax)
    
    # Highlight best model for each modality
    for col_idx, col in enumerate(df.columns):
        max_idx = df[col].idxmax()
        if pd.notna(df.loc[max_idx, col]):
            row_idx = df.index.get_loc(max_idx)
            # Add border around best cell
            rect = plt.Rectangle((col_idx, row_idx), 1, 1, 
                               fill=False, edgecolor='green', 
                               linewidth=3)
            ax.add_patch(rect)
    
    # Add title and labels
    plt.title('Model Performance Across Different Data Modalities', pad=20, fontsize=16)
    plt.xlabel('Data Modality', fontsize=14)
    plt.ylabel('Machine Learning Model', fontsize=14)
    
    # Add note about null values
    plt.figtext(0.5, 0.02, 
                'Note: ElasticNet is only applied to high-dimensional modalities (Expression & Methylation)',
                ha='center', fontsize=10, style='italic')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save figure
    plt.savefig(f'{output_dir}/model_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/model_performance_heatmap.pdf', bbox_inches='tight')
    plt.close()
    
    print("Model performance heatmap created successfully!")
    
    # Print summary
    print("\nBest models by modality:")
    for col in df.columns:
        best_model = df[col].idxmax()
        best_score = df[col].max()
        print(f"  {col}: {best_model} (AUC: {best_score:.3f})")


def create_fusion_methods_heatmap(data_dir, output_dir):
    """Create heatmap showing fusion method performance across strategies."""
    # Load data
    with open(f'{data_dir}/advanced_fusion_results/advanced_fusion_results.json', 'r') as f:
        results = json.load(f)
    
    cv_summary = results['cv_performance_summary']
    
    # Extract data for heatmap
    strategies = ['minimal', 'diverse', 'mixed_1', 'mixed_2']
    methods = ['ensemble', 'rank', 'weighted', 'geometric', 'stacking']
    
    # Create matrix
    matrix = []
    for strategy in strategies:
        row = []
        for method in methods:
            if strategy in cv_summary and 'all_methods' in cv_summary[strategy]:
                if method in cv_summary[strategy]['all_methods']:
                    auc = cv_summary[strategy]['all_methods'][method]['mean_auc']
                    row.append(auc)
                else:
                    row.append(np.nan)
            else:
                row.append(np.nan)
        matrix.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(matrix, 
                     index=['Minimal', 'Diverse', 'Mixed 1', 'Mixed 2'],
                     columns=['Ensemble', 'Rank', 'Weighted', 'Geometric', 'Stacking'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    sns.heatmap(df, annot=True, fmt='.4f', cmap='RdYlBu_r',
                center=0.70, vmin=0.50, vmax=0.78,
                cbar_kws={'label': 'Mean AUC Score'},
                linewidths=0.5, linecolor='gray',
                ax=ax)
    
    # Highlight best method for each strategy
    for row_idx, strategy in enumerate(df.index):
        max_col = df.loc[strategy].idxmax()
        if pd.notna(df.loc[strategy, max_col]):
            col_idx = df.columns.get_loc(max_col)
            # Add border around best cell
            rect = plt.Rectangle((col_idx, row_idx), 1, 1,
                               fill=False, edgecolor='green',
                               linewidth=3)
            ax.add_patch(rect)
    
    # Highlight overall best
    best_val = df.max().max()
    best_positions = np.where(df.values == best_val)
    if len(best_positions[0]) > 0:
        for i in range(len(best_positions[0])):
            rect = plt.Rectangle((best_positions[1][i], best_positions[0][i]), 1, 1,
                               fill=False, edgecolor='red',
                               linewidth=4)
            ax.add_patch(rect)
    
    # Add title and labels
    plt.title('Fusion Method Performance Across Different Strategies', pad=20, fontsize=16)
    plt.xlabel('Fusion Method', fontsize=14)
    plt.ylabel('Feature Selection Strategy', fontsize=14)
    
    # Add legend for borders
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='green', linewidth=3, label='Best method per strategy'),
        Patch(facecolor='none', edgecolor='red', linewidth=4, label='Overall best (0.7712)')
    ]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, -0.15))
    
    # Add strategy descriptions
    strategy_desc = {
        'Minimal': 'Expression(300), Methylation(400), Protein(110), Mutation(400)',
        'Diverse': 'Expression(6000), Methylation(1000), Protein(110), Mutation(1000)',
        'Mixed 1': 'Expression(300), Methylation(400), Protein(150), Mutation(1000)',
        'Mixed 2': 'Expression(300), Methylation(1500), Protein(110), Mutation(1000)'
    }
    
    # Add as text below figure
    desc_text = '\n'.join([f"{k}: {v}" for k, v in strategy_desc.items()])
    plt.figtext(0.5, -0.05, desc_text, ha='center', fontsize=9, style='italic')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    # Save figure
    plt.savefig(f'{output_dir}/fusion_methods_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fusion_methods_heatmap.pdf', bbox_inches='tight')
    plt.close()
    
    print("\nFusion methods heatmap created successfully!")
    
    # Print summary
    print("\nBest fusion methods by strategy:")
    for strategy in df.index:
        best_method = df.loc[strategy].idxmax()
        best_score = df.loc[strategy].max()
        print(f"  {strategy}: {best_method} (AUC: {best_score:.4f})")
    
    print(f"\nOverall best: {df.max().max():.4f}")


def create_cv_performance_heatmap(data_dir, output_dir):
    """Create heatmap showing cross-validation performance across folds (fusion only for now)."""
    # Load fusion data
    with open(f'{data_dir}/advanced_fusion_results/advanced_fusion_results.json', 'r') as f:
        results = json.load(f)
    
    cv_summary = results['cv_performance_summary']
    
    # Extract fold AUCs for minimal and diverse strategies
    fold_data = {}
    
    # Get minimal fusion fold AUCs (using best method)
    if 'minimal' in cv_summary and 'fold_aucs' in cv_summary['minimal']:
        fold_data['Minimal Fusion'] = cv_summary['minimal']['fold_aucs']
    
    # Get diverse fusion fold AUCs (using best method)
    if 'diverse' in cv_summary and 'fold_aucs' in cv_summary['diverse']:
        fold_data['Diverse Fusion'] = cv_summary['diverse']['fold_aucs']
    
    # Create DataFrame
    df = pd.DataFrame(fold_data, index=[f'Fold {i+1}' for i in range(5)])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(df, annot=True, fmt='.4f', cmap='RdYlBu_r',
                center=0.75, vmin=0.65, vmax=0.88,
                cbar_kws={'label': 'AUC Score'},
                linewidths=0.5, linecolor='gray',
                ax=ax)
    
    # Add mean AUC row
    mean_row = df.mean()
    std_row = df.std()
    
    # Add text below heatmap with means
    text_y = -0.15
    for i, col in enumerate(df.columns):
        ax.text(i + 0.5, 5.7, f'Mean: {mean_row[col]:.4f}', 
                ha='center', va='top', fontsize=10, weight='bold')
        ax.text(i + 0.5, 6.1, f'(±{std_row[col]:.4f})', 
                ha='center', va='top', fontsize=9, style='italic')
    
    # Highlight best and worst folds
    for col_idx, col in enumerate(df.columns):
        best_fold_idx = df[col].idxmax()
        worst_fold_idx = df[col].idxmin()
        
        # Best fold - green border
        best_row = df.index.get_loc(best_fold_idx)
        rect = plt.Rectangle((col_idx, best_row), 1, 1,
                           fill=False, edgecolor='green',
                           linewidth=2)
        ax.add_patch(rect)
        
        # Worst fold - red border
        worst_row = df.index.get_loc(worst_fold_idx)
        rect = plt.Rectangle((col_idx, worst_row), 1, 1,
                           fill=False, edgecolor='red',
                           linewidth=2, linestyle='--')
        ax.add_patch(rect)
    
    # Add title and labels
    plt.title('Cross-Validation Performance: Fusion Strategies', pad=20, fontsize=16)
    plt.xlabel('Fusion Strategy', fontsize=14)
    plt.ylabel('Cross-Validation Fold', fontsize=14)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='green', linewidth=2, label='Best fold'),
        Patch(facecolor='none', edgecolor='red', linewidth=2, linestyle='--', label='Worst fold')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add note
    plt.figtext(0.5, 0.02,
                'Note: Shows AUC performance for each CV fold. Complete heatmap will include individual modalities after re-running scripts.',
                ha='center', fontsize=10, style='italic')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save figure
    plt.savefig(f'{output_dir}/cv_performance_heatmap_sample.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/cv_performance_heatmap_sample.pdf', bbox_inches='tight')
    plt.close()
    
    print("CV performance heatmap (sample) created successfully!")
    
    # Print summary
    print("\nCross-validation performance summary:")
    for strategy in df.columns:
        print(f"\n{strategy}:")
        print(f"  Mean AUC: {mean_row[strategy]:.4f} ± {std_row[strategy]:.4f}")
        print(f"  Best fold: {df[strategy].idxmax()} ({df[strategy].max():.4f})")
        print(f"  Worst fold: {df[strategy].idxmin()} ({df[strategy].min():.4f})")
        print(f"  Range: {df[strategy].max() - df[strategy].min():.4f}")


def main():
    """Generate both performance heatmaps."""
    # Setup paths
    data_dir = '/Users/tobyliu/bladder'
    output_dir = Path(data_dir) / '04_analysis_viz' / 'performance_heatmaps'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating performance heatmaps...")
    print("=" * 60)
    
    # Create model performance heatmap
    print("\n1. Creating individual model performance heatmap...")
    create_model_performance_heatmap(data_dir, output_dir)
    
    # Create fusion methods heatmap
    print("\n2. Creating fusion methods performance heatmap...")
    create_fusion_methods_heatmap(data_dir, output_dir)
    
    # Create CV performance heatmap (sample with fusion only)
    print("\n3. Creating cross-validation performance heatmap (sample)...")
    create_cv_performance_heatmap(data_dir, output_dir)
    
    print("\n" + "=" * 60)
    print(f"All heatmaps saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()