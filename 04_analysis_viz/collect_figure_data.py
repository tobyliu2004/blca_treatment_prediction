#!/usr/bin/env python3
"""
Collect Figure Data: Extract real predictions and features for additional poster figures
Author: ML Engineers Team
Date: 2025-01-21

This script collects actual model predictions, probabilities, and feature importances
from the best configuration to create confusion matrix, calibration plot, feature
importance, and decision curve analysis figures.
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import os
import sys
sys.path.append('/Users/tobyliu/bladder')
from collections import defaultdict
from scipy.stats import fisher_exact

# Import functions from step7_fusion_optimized
sys.path.append('/Users/tobyliu/bladder/03_fusion_approaches')
from step7_fusion_optimized import (
    load_data_and_labels, 
    load_modality_data,
    select_features_optimized,
    train_optimized_models
)


def collect_detailed_predictions(modalities_data, y_train, sample_ids, class_weights, 
                               n_expr=3000, n_meth=3000, n_prot=185, n_mut=300, n_folds=5):
    """
    Collect detailed predictions and feature information from the best configuration.
    """
    print("\n" + "="*70)
    print("COLLECTING DETAILED PREDICTIONS FOR FIGURES")
    print("="*70)
    print(f"\nConfiguration: expr={n_expr}, meth={n_meth}, prot={n_prot}, mut={n_mut}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Storage for all results
    all_predictions = []
    all_probabilities = []
    all_true_labels = []
    feature_counts = defaultdict(lambda: defaultdict(int))
    modality_predictions_all = defaultdict(list)
    modality_probabilities_all = defaultdict(list)
    fold_predictions = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(modalities_data['expression'], y_train)):
        print(f"\nProcessing fold {fold_idx + 1}/{n_folds}...")
        
        y_train_fold = y_train[train_idx]
        y_val_fold = y_train[val_idx]
        
        if len(np.unique(y_train_fold)) < 2:
            print(f"  Skipping fold {fold_idx+1} - only one class present")
            continue
        
        modality_predictions = {}
        modality_probabilities = {}
        
        # Process each modality
        for modality, n_features in [('expression', n_expr), ('methylation', n_meth), 
                                     ('protein', n_prot), ('mutation', n_mut)]:
            
            print(f"  Processing {modality}...")
            
            X_train_fold = modalities_data[modality].iloc[train_idx]
            X_val_fold = modalities_data[modality].iloc[val_idx]
            
            # Select features
            selected_features = select_features_optimized(
                X_train_fold, y_train_fold, modality, n_features
            )
            
            if len(selected_features) < 5:
                print(f"    Skipping {modality} - too few features")
                continue
            
            # Track feature selection frequency
            for feature in selected_features[:20]:  # Top 20 features
                feature_counts[modality][feature] += 1
            
            X_train_selected = X_train_fold[selected_features]
            X_val_selected = X_val_fold[selected_features]
            
            # Train models
            predictions = train_optimized_models(
                X_train_selected, y_train_fold,
                X_val_selected, y_val_fold,
                class_weights, modality
            )
            
            # Use best model for this modality
            best_model = max(predictions.items(), 
                           key=lambda x: roc_auc_score(y_val_fold, x[1]))
            
            modality_probabilities[modality] = best_model[1]
            modality_predictions[modality] = (best_model[1] >= 0.5).astype(int)
            
            # Store for analysis
            modality_probabilities_all[modality].extend(best_model[1])
            modality_predictions_all[modality].extend((best_model[1] >= 0.5).astype(int))
            
            auc = roc_auc_score(y_val_fold, best_model[1])
            print(f"    {modality}: {len(selected_features)} features, best={best_model[0]} (AUC={auc:.3f})")
        
        # Fusion - weighted by performance
        if len(modality_probabilities) >= 2:
            weights = {}
            for modality, prob in modality_probabilities.items():
                weights[modality] = roc_auc_score(y_val_fold, prob)
            
            total_weight = sum(weights.values())
            fusion_prob = np.zeros_like(next(iter(modality_probabilities.values())))
            
            for modality, prob in modality_probabilities.items():
                fusion_prob += prob * (weights[modality] / total_weight)
            
            fusion_pred = (fusion_prob >= 0.5).astype(int)
            
            # Store results
            all_probabilities.extend(fusion_prob)
            all_predictions.extend(fusion_pred)
            all_true_labels.extend(y_val_fold)
            
            # Store fold-specific results
            fold_predictions.append({
                'fold': fold_idx,
                'val_indices': val_idx,
                'probabilities': fusion_prob,
                'predictions': fusion_pred,
                'true_labels': y_val_fold,
                'modality_weights': weights
            })
            
            fusion_auc = roc_auc_score(y_val_fold, fusion_prob)
            print(f"  Fusion AUC: {fusion_auc:.3f}")
    
    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_true_labels = np.array(all_true_labels)
    
    # Get top features for each modality
    top_features = {}
    for modality, features in feature_counts.items():
        # Sort by frequency across folds
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        top_features[modality] = [feat for feat, count in sorted_features[:10]]
    
    # Calculate overall metrics
    overall_auc = roc_auc_score(all_true_labels, all_probabilities)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total predictions collected: {len(all_predictions)}")
    print(f"Overall AUC: {overall_auc:.3f}")
    print(f"Class distribution: {np.sum(all_true_labels==1)} positive, {np.sum(all_true_labels==0)} negative")
    
    # Compile all results
    detailed_results = {
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'true_labels': all_true_labels,
        'feature_importance': top_features,
        'modality_predictions': {
            modality: np.array(preds) 
            for modality, preds in modality_predictions_all.items()
        },
        'modality_probabilities': {
            modality: np.array(probs) 
            for modality, probs in modality_probabilities_all.items()
        },
        'fold_results': fold_predictions,
        'overall_auc': overall_auc,
        'config': {
            'n_expr': n_expr,
            'n_meth': n_meth,
            'n_prot': n_prot,
            'n_mut': n_mut
        }
    }
    
    return detailed_results


def main():
    """Main function to collect figure data."""
    data_dir = '/Users/tobyliu/bladder'
    output_dir = f'{data_dir}/fusion_optimized_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load existing results to get best configuration
    with open(f'{output_dir}/optimized_results.json', 'r') as f:
        results = json.load(f)
    
    print(f"Best configuration from previous run: {results['best_config']}")
    print(f"Best AUC: {results['best_auc']:.3f}")
    
    # Load data
    y_train, sample_ids, class_weights = load_data_and_labels(data_dir)
    
    # Load all modality data
    print("\nLoading modality data...")
    modalities_data = {}
    for modality in ['expression', 'protein', 'methylation', 'mutation']:
        X_data = load_modality_data(modality, data_dir)
        modalities_data[modality] = X_data.loc[sample_ids]
        print(f"  {modality}: {X_data.shape}")
    
    # Collect detailed predictions using best configuration (high_features)
    detailed_results = collect_detailed_predictions(
        modalities_data, y_train, sample_ids, class_weights,
        n_expr=3000, n_meth=3000, n_prot=185, n_mut=300
    )
    
    # Save results
    output_file = f'{output_dir}/figure_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(detailed_results, f)
    
    print(f"\nFigure data saved to: {output_file}")
    
    # Print top features for verification
    print("\nTop features by modality:")
    for modality, features in detailed_results['feature_importance'].items():
        print(f"\n{modality.upper()}:")
        for i, feature in enumerate(features[:5], 1):
            print(f"  {i}. {feature}")


if __name__ == "__main__":
    main()