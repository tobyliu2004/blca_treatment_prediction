#!/usr/bin/env python3
"""
Create feature importance figure for bladder cancer prediction analysis.
Shows top features for each modality (Expression, Protein, Mutation).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def create_feature_importance_figure(data_dir, output_dir, top_n=15):
    """Create multi-panel feature importance figure."""
    # Load data
    with open(f'{data_dir}/individual_model_results.json', 'r') as f:
        results = json.load(f)
    
    feature_data = results['feature_importance_data']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('Top Predictive Features by Data Modality', fontsize=16, y=1.02)
    
    # Define colors for each modality
    colors = {
        'expression': '#FF6B6B',  # Red
        'protein': '#4ECDC4',      # Teal
        'mutation': '#45B7D1'      # Blue
    }
    
    # Order of modalities
    modalities = ['expression', 'protein', 'mutation']
    titles = {
        'expression': 'Gene Expression',
        'protein': 'Protein Expression',
        'mutation': 'Mutation Status'
    }
    
    for idx, (ax, modality) in enumerate(zip(axes, modalities)):
        if modality not in feature_data:
            ax.text(0.5, 0.5, f'No data for {modality}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(titles[modality])
            continue
        
        # Get data
        mod_data = feature_data[modality]
        features = mod_data['top_features'][:top_n]
        
        # Extract feature names and importances
        feature_names = [f[0] for f in features]
        importances = [f[1] for f in features]
        
        # Create dataframe for plotting
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        df = df.sort_values('Importance', ascending=True)
        
        # Create horizontal bar plot
        bars = ax.barh(df['Feature'], df['Importance'], 
                       color=colors[modality], alpha=0.8)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, df['Importance'])):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', ha='left', va='center', fontsize=8)
        
        # Customize subplot
        ax.set_title(f'{titles[modality]}\n(Best Model: {mod_data["best_model"].upper()})', 
                    fontsize=12, pad=10)
        ax.set_xlabel('Feature Importance Score', fontsize=10)
        ax.set_ylabel('')
        
        # Add grid for readability
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Adjust x-axis limits to make room for labels
        ax.set_xlim(0, max(importances) * 1.15)
        
        # Highlight known cancer genes for mutation data
        if modality == 'mutation':
            cancer_genes = ['TP53', 'RB1', 'FGFR3', 'PIK3CA', 'ERBB2', 'CDKN2A']
            for tick in ax.get_yticklabels():
                gene = tick.get_text().split('_')[0]  # Handle pathway names
                if any(cg in gene for cg in cancer_genes):
                    tick.set_weight('bold')
                    tick.set_color('darkred')
    
    # Adjust layout
    plt.tight_layout()
    
    # Add note about model performance
    auc_scores = {
        'expression': 0.676,
        'protein': 0.701,
        'mutation': 0.620
    }
    
    note_text = "Feature importance derived from best-performing model for each modality. "
    note_text += f"Mean AUC: Expression={auc_scores['expression']:.3f}, "
    note_text += f"Protein={auc_scores['protein']:.3f}, "
    note_text += f"Mutation={auc_scores['mutation']:.3f}"
    
    plt.figtext(0.5, -0.02, note_text, ha='center', fontsize=9, style='italic')
    
    # Save figure
    plt.savefig(f'{output_dir}/feature_importance_by_modality.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/feature_importance_by_modality.pdf', 
                bbox_inches='tight')
    plt.close()
    
    print("Feature importance figure created successfully!")
    
    # Print summary
    print("\nTop 5 features by modality:")
    for modality in modalities:
        if modality in feature_data:
            print(f"\n{titles[modality]}:")
            for i, (feat, imp) in enumerate(feature_data[modality]['top_features'][:5]):
                print(f"  {i+1}. {feat} ({imp:.4f})")


def create_simplified_feature_importance(data_dir, output_dir, top_n=10):
    """Create a simplified version with fewer features for poster clarity."""
    # Load data
    with open(f'{data_dir}/individual_model_results.json', 'r') as f:
        results = json.load(f)
    
    feature_data = results['feature_importance_data']
    
    # Create single figure combining all modalities
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Collect all features
    all_features = []
    
    for modality in ['expression', 'protein', 'mutation']:
        if modality not in feature_data:
            continue
            
        features = feature_data[modality]['top_features'][:top_n]
        for feat, imp in features:
            # Normalize importance scores across modalities
            if modality == 'mutation':
                # Convert count to proportion
                norm_imp = imp / 5.0
            else:
                # Keep as is (already normalized 0-1)
                norm_imp = imp
            
            all_features.append({
                'Feature': feat,
                'Importance': norm_imp,
                'Modality': modality.capitalize()
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Create grouped bar plot
    df_pivot = df.pivot(index='Feature', columns='Modality', values='Importance')
    
    # Plot
    df_pivot.plot(kind='barh', ax=ax, width=0.8)
    
    # Customize
    ax.set_xlabel('Normalized Feature Importance', fontsize=12)
    ax.set_ylabel('')
    ax.set_title('Top Predictive Features Across Data Modalities', fontsize=14, pad=20)
    ax.legend(title='Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(f'{output_dir}/feature_importance_combined.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/feature_importance_combined.pdf', 
                bbox_inches='tight')
    plt.close()
    
    print("\nSimplified feature importance figure created successfully!")


def main():
    """Generate feature importance figures."""
    # Setup paths
    data_dir = '/Users/tobyliu/bladder'
    output_dir = Path(data_dir) / '04_analysis_viz' / 'feature_importance_figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating feature importance figures...")
    print("=" * 60)
    
    # Create detailed figure
    print("\n1. Creating detailed feature importance figure...")
    create_feature_importance_figure(data_dir, output_dir)
    
    # Create simplified figure
    print("\n2. Creating simplified combined figure...")
    create_simplified_feature_importance(data_dir, output_dir)
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()