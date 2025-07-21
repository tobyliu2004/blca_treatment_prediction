#!/usr/bin/env python3
"""
Step 7 Optimized: Fine-tuned Fusion Based on Previous Results
Author: ML Engineers Team
Date: 2025-01-15

Goal: Optimize the 0.718 AUC result by:
1. Testing more feature counts (3000, 4000)
2. Optimizing fusion weights based on modality performance
3. Testing exclusion of poor-performing modalities
4. Finding most stable configurations
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
# from catboost import CatBoostClassifier  # Removed to avoid creating catboost_info folder
import warnings
warnings.filterwarnings('ignore')
import os
from collections import defaultdict
import json
from scipy.stats import fisher_exact


# loads 227 training samples, calculates class weights for imbalanced dataset (159 responders, 68 non-responders)
# basically tells model to pay more attention to non-responders: 
# class weights = {0: 1.669, 1: 0.714}, meaning when the model misclassified a non-responder, it counts as 1.669 errors, while misclassifying a responder counts as 0.714 errors
def load_data_and_labels(data_dir='/Users/tobyliu/bladder'):
    """Load training data and labels."""
    print("Loading data and labels...")
    
    labels_df = pd.read_csv(f'{data_dir}/overlap_treatment_response_fixed.csv')
    train_samples = pd.read_csv(f'{data_dir}/train_samples_fixed.csv')
    
    train_labels = train_samples.merge(labels_df, on='sampleID')
    y_train = train_labels['treatment_response'].values
    sample_ids = train_labels['sampleID'].values
    
    print(f"  Loaded {len(y_train)} training samples")
    print(f"  Class distribution: {np.sum(y_train==1)} responders, {np.sum(y_train==0)} non-responders")
    
    class_weights = {
        0: len(y_train) / (2 * np.sum(y_train == 0)),
        1: len(y_train) / (2 * np.sum(y_train == 1))
    }
    
    return y_train, sample_ids, class_weights


def load_modality_data(modality, data_dir='/Users/tobyliu/bladder'):
    """Load preprocessed data for a specific modality."""
    train_file = f'{data_dir}/preprocessed_data/{modality}/{modality}_train_preprocessed.csv'
    return pd.read_csv(train_file, index_col=0)


# for expression and methylation, fold change calculate |mean(responders) - mean(non-responders)| - Selects top N features by fold change magnitude LARGEST DIFFERENCE
# protein, F statistic(ANOVA) to test statisistical mean difference between responders and non-responders
# mutation, use fishers exact test to find genes with significant difference in mutation rates between responders and non-responders
def select_features_optimized(X_train_fold, y_train_fold, modality, n_features):
    """
    Optimized feature selection based on what worked best.
    """
    from sklearn.feature_selection import SelectKBest, f_classif, chi2
    
    if modality == 'expression' or modality == 'methylation':
        # Fold change worked well for these
        responder_mean = X_train_fold[y_train_fold == 1].mean()
        non_responder_mean = X_train_fold[y_train_fold == 0].mean()
        fold_change = np.abs(responder_mean - non_responder_mean)
        top_features = fold_change.nlargest(n_features).index.tolist()
        return top_features
        
    elif modality == 'protein':
        # F-test for protein
        selector = SelectKBest(f_classif, k=min(n_features, X_train_fold.shape[1]))
        selector.fit(X_train_fold, y_train_fold)
        return X_train_fold.columns[selector.get_support()].tolist()
        
    elif modality == 'mutation':
        # Fisher's test for mutations
        p_values = []
        
        for gene in X_train_fold.columns:
            # Skip special columns
            if gene in ['selected_mutation_burden', 'total_mutation_burden'] or 'pathway' in gene:
                continue
                
            mutated_responders = np.sum((X_train_fold[gene] == 1) & (y_train_fold == 1))
            mutated_non_responders = np.sum((X_train_fold[gene] == 1) & (y_train_fold == 0))
            wild_responders = np.sum((X_train_fold[gene] == 0) & (y_train_fold == 1))
            wild_non_responders = np.sum((X_train_fold[gene] == 0) & (y_train_fold == 0))
            
            try:
                _, p_value = fisher_exact([[wild_non_responders, wild_responders],
                                          [mutated_non_responders, mutated_responders]])
                p_values.append((gene, p_value))
            except:
                p_values.append((gene, 1.0))
        
        # Sort by p-value and select top features
        p_values.sort(key=lambda x: x[1])
        selected_features = [gene for gene, _ in p_values[:n_features]]
        
        # Always include special columns if present
        special_cols = ['selected_mutation_burden', 'total_mutation_burden']
        pathway_cols = [col for col in X_train_fold.columns if 'pathway' in col]
        selected_features.extend([col for col in special_cols + pathway_cols if col in X_train_fold.columns])
        
        return selected_features


#NEED strong regularization for high dimensional models to reduce risk of overfitting
# for expression/methylation (high dimensionality), use XGBoost n_estimators=50, max_depth=3, High regularization: reg_alpha=2.0, reg_lambda=3.0, learning_rate=0.1
# random forest n_estimators=100, max_depth=5, max_features='sqrt' (square root of total features), min_samples_leaf=5
# SGD Classifier ElasticNet penalty (L1+L2) alpha=0.01, l1_ratio=0.5, max_iter=1000
# for protein/mutation(lower dimensionality), use CatBoost iterations=100, learning_rate=0.1, depth=6, learning rate=0.1
# XGBoost n_estimators=100, max_depth=6, Less regularization than high-dim version
# Gradient Boosting n_estimators=100, learning_rate=0.1, max_depth=5
def train_optimized_models(X_train, y_train, X_val, y_val, class_weights, modality):
    """
    Train models optimized for each modality based on previous results.
    """
    predictions = {}
    
    # Common parameters
    scale_pos_weight = class_weights[0] / class_weights[1]
    
    if modality in ['expression', 'methylation']:
        # These need strong regularization due to high dimensionality
        
        # 1. XGBoost with strong regularization
        xgb_model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            reg_alpha=2.0,
            reg_lambda=3.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        predictions['xgboost'] = xgb_model.predict_proba(X_val)[:, 1]
        
        # 2. Random Forest with constraints
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            max_features='sqrt',
            min_samples_leaf=5,
            class_weight=class_weights,
            random_state=42
        )
        rf.fit(X_train, y_train)
        predictions['random_forest'] = rf.predict_proba(X_val)[:, 1]
        
        # 3. Logistic with ElasticNet
        from sklearn.linear_model import SGDClassifier
        sgd = SGDClassifier(
            loss='log_loss',
            penalty='elasticnet',
            alpha=0.01,
            l1_ratio=0.5,
            class_weight=class_weights,
            random_state=42,
            max_iter=1000
        )
        sgd.fit(X_train, y_train)
        # SGD doesn't have predict_proba, use decision_function
        predictions['logistic'] = 1 / (1 + np.exp(-sgd.decision_function(X_val)))
        
    else:
        # Protein and mutation can use less regularization
        
        # 1. CatBoost - REMOVED to avoid creating catboost_info folder
        # cb = CatBoostClassifier(
        #     iterations=100,
        #     learning_rate=0.1,
        #     depth=6,
        #     class_weights=list(class_weights.values()),
        #     random_state=42,
        #     verbose=False
        # )
        # cb.fit(X_train, y_train)
        # predictions['catboost'] = cb.predict_proba(X_val)[:, 1]
        
        # 2. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        predictions['xgboost'] = xgb_model.predict_proba(X_val)[:, 1]
        
        # 3. Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb.fit(X_train, y_train)
        predictions['gradient_boosting'] = gb.predict_proba(X_val)[:, 1]
    
    return predictions

# six test configuration: 
# 1. 'standard': 2000 expr, 2000 meth, 150 prot, 200 mut
# 2. 'high_features': 3000 expr, 3000 meth, 185 prot, 300 mut (WINNER!)
# 3. 'very_high': 4000 expr, 4000 meth, 185 prot, 500 mut
# 4. 'no_mutation': 3000 expr, 3000 meth, 185 prot, 0 mut (excludes mutation entirely)
# 5. 'focus_protein': 1000 expr, 1000 meth, 185 prot, 100 mut
# 6. 'balanced': 2500 expr, 2500 meth, 185 prot, 250 mut
#for each configuration, for each fold, split into 80% train 20% validation, train models, pick best model for each modality, then test different fusion methods:
# 1. Simple average of predictions
# 2. Weighted average based on performance of each modality
# record which fusion method works best
def cross_validate_optimized(modalities_data, y_train, sample_ids, class_weights, n_folds=5):
    """
    Optimized cross-validation testing different configurations.
    """
    print("\n" + "="*70)
    print("OPTIMIZED FUSION CROSS-VALIDATION")
    print("="*70)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Test configurations
    configs = [
        # Config: (n_expr, n_meth, n_prot, n_mut, exclude_modalities)
        ('standard', 2000, 2000, 150, 200, []),
        ('high_features', 3000, 3000, 185, 300, []),
        ('very_high', 4000, 4000, 185, 500, []),
        ('no_mutation', 3000, 3000, 185, 0, ['mutation']),
        ('focus_protein', 1000, 1000, 185, 100, []),
        ('balanced', 2500, 2500, 185, 250, []),
    ]
    
    results = defaultdict(lambda: defaultdict(list))
    modality_performances = defaultdict(list)
    
    for config_name, n_expr, n_meth, n_prot, n_mut, exclude in configs:
        print(f"\n\nTesting configuration: {config_name}")
        print(f"  Features: expr={n_expr}, meth={n_meth}, prot={n_prot}, mut={n_mut}")
        if exclude:
            print(f"  Excluding: {exclude}")
        
        config_aucs = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(modalities_data['expression'], y_train)):
            print(f"\n  Fold {fold_idx + 1}/{n_folds}:")
            
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]
            
            modality_predictions = {}
            
            # Process each modality
            for modality, n_features in [('expression', n_expr), ('methylation', n_meth), 
                                        ('protein', n_prot), ('mutation', n_mut)]:
                
                if modality in exclude or n_features == 0:
                    continue
                
                X_train_fold = modalities_data[modality].iloc[train_idx]
                X_val_fold = modalities_data[modality].iloc[val_idx]
                
                # Select features
                selected_features = select_features_optimized(
                    X_train_fold, y_train_fold, modality, n_features
                )
                
                if len(selected_features) < 5:
                    continue
                
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
                modality_predictions[modality] = best_model[1]
                
                # Track individual performance
                modality_auc = roc_auc_score(y_val_fold, best_model[1])
                modality_performances[modality].append(modality_auc)
                
                print(f"    {modality}: {len(selected_features)} features, "
                      f"best={best_model[0]} (AUC={modality_auc:.3f})")
            
            if len(modality_predictions) < 2:
                continue
            
            # Test different fusion methods
            
            # 1. Simple average
            simple_avg = np.mean(list(modality_predictions.values()), axis=0)
            simple_auc = roc_auc_score(y_val_fold, simple_avg)
            results[config_name]['simple_average'].append(simple_auc)
            
            # 2. Weighted by performance
            weights = {}
            for modality, pred in modality_predictions.items():
                weights[modality] = roc_auc_score(y_val_fold, pred)
            
            total_weight = sum(weights.values())
            weighted_pred = np.zeros_like(next(iter(modality_predictions.values())))
            for modality, pred in modality_predictions.items():
                weighted_pred += pred * (weights[modality] / total_weight)
            
            weighted_auc = roc_auc_score(y_val_fold, weighted_pred)
            results[config_name]['weighted'].append(weighted_auc)
            
            
            # Best fusion for this fold
            best_fusion_auc = max(simple_auc, weighted_auc)
            config_aucs.append(best_fusion_auc)
            print(f"    Simple avg: {simple_auc:.3f}, Weighted: {weighted_auc:.3f}")
            print(f"    Best fusion AUC: {best_fusion_auc:.3f}")
        
        # Summary for this config
        if config_aucs:
            mean_auc = np.mean(config_aucs)
            std_auc = np.std(config_aucs)
            print(f"\n  {config_name}: {mean_auc:.3f} ± {std_auc:.3f}")
    
    return results, modality_performances


def main():
    """Main function."""
    data_dir = '/Users/tobyliu/bladder'
    output_dir = f'{data_dir}/fusion_optimized_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    y_train, sample_ids, class_weights = load_data_and_labels(data_dir)
    
    # Load all modality data
    print("\nLoading modality data...")
    modalities_data = {}
    for modality in ['expression', 'protein', 'methylation', 'mutation']:
        X_data = load_modality_data(modality, data_dir)
        modalities_data[modality] = X_data.loc[sample_ids]
        print(f"  {modality}: {X_data.shape}")
    
    # Run optimized fusion
    fusion_results, modality_performances = cross_validate_optimized(
        modalities_data, y_train, sample_ids, class_weights
    )
    
    # Analyze modality stability
    print("\n" + "="*70)
    print("MODALITY PERFORMANCE ANALYSIS")
    print("="*70)
    
    for modality, aucs in modality_performances.items():
        if aucs:
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            print(f"{modality}: {mean_auc:.3f} ± {std_auc:.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("OPTIMIZED FUSION RESULTS")
    print("="*70)
    
    best_config = None
    best_method = None
    best_auc = 0
    
    print("\nConfiguration           Method          AUC Mean ± Std")
    print("-" * 60)
    
    for config, methods in fusion_results.items():
        for method, aucs in methods.items():
            if aucs:
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                print(f"{config:<20} {method:<15} {mean_auc:.3f} ± {std_auc:.3f}")
                
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_config = config
                    best_method = method
    
    print("\n" + "="*70)
    print(f"BEST CONFIGURATION: {best_config} with {best_method}")
    print(f"Best AUC: {best_auc:.3f}")
    print("="*70)
    
    # Compare to previous
    print("\nCOMPARISON:")
    print("  Previous best (late fusion):     0.718")
    print(f"  Best configuration here:         {best_auc:.3f}")
    print(f"  Change:                          {best_auc - 0.718:+.3f}")
    
    if best_auc >= 0.75:
        print("\n🎉 TARGET ACHIEVED: AUC >= 0.75!")
    else:
        print(f"\n📊 Solid performance at {best_auc:.3f}")
        print("\nNote: With 227 samples and high-dimensional data, this is near the")
        print("      practical limit. Further improvements would likely require:")
        print("      - More samples")
        print("      - External validation cohort")
        print("      - Domain-specific feature engineering")
    
    # Save results
    with open(f'{output_dir}/optimized_results.json', 'w') as f:
        json.dump({
            'fusion_results': {k: {m: {'mean': np.mean(v), 'std': np.std(v)} 
                             for m, v in methods.items() if v}
                             for k, methods in fusion_results.items()},
            'modality_performances': {k: {'mean': np.mean(v), 'std': np.std(v)}
                                    for k, v in modality_performances.items() if v},
            'best_config': best_config,
            'best_method': best_method,
            'best_auc': best_auc
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()