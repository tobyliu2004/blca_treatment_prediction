#!/usr/bin/env python3
"""
Compare Expression vs Embedded Expression Performance
Author: ML Engineers Team
Date: 2025-01-16

Goal: Compare performance of:
1. Standard preprocessed expression data (from step4)
2. scFoundation embedded expression data

Both will use the same models and cross-validation strategy from step6
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
# from catboost import CatBoostClassifier  # Removed to avoid creating catboost_info folder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns
# import time  # Removed - no longer tracking timing
# import sys  # Removed - not used


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
    
    # Calculate class weights
    class_weights = {
        0: len(y_train) / (2 * np.sum(y_train == 0)),
        1: len(y_train) / (2 * np.sum(y_train == 1))
    }
    print(f"  Class weights: {class_weights}")
    
    return y_train, sample_ids, class_weights


def load_preprocessed_expression(data_dir='/Users/tobyliu/bladder'):
    """Load preprocessed expression data from step4."""
    print("\nLoading preprocessed expression data...")
    train_file = f'{data_dir}/preprocessed_data/expression/expression_train_preprocessed.csv'
    train_data = pd.read_csv(train_file, index_col=0)
    print(f"  Preprocessed expression shape: {train_data.shape}")
    return train_data


def load_embedded_expression(data_dir='/Users/tobyliu/bladder'):
    """Load scFoundation embedded expression data."""
    print("\nLoading embedded expression data...")
    
    # Load embedded data
    embedded_data = np.load(f'{data_dir}/BLCA_expression_embed.npy', allow_pickle=True)
    
    # Load original expression to get sample order
    original_expr = pd.read_csv(f'{data_dir}/BLCA_expression', sep='\t', index_col=0)
    all_sample_ids = list(original_expr.columns)
    
    # Load train samples
    train_samples = pd.read_csv(f'{data_dir}/train_samples_fixed.csv')['sampleID'].tolist()
    
    # Get indices of training samples
    train_indices = [i for i, sid in enumerate(all_sample_ids) if sid in train_samples]
    
    # Extract training data
    train_embedded = embedded_data[train_indices]
    
    # Create DataFrame with proper sample IDs
    train_df = pd.DataFrame(
        train_embedded,
        index=[all_sample_ids[i] for i in train_indices],
        columns=[f'embed_{i}' for i in range(embedded_data.shape[1])]
    )
    
    print(f"  Embedded expression shape: {train_df.shape}")
    print(f"  Features: {train_df.shape[1]} embeddings (vs {embedded_data.shape[1]} total)")
    
    return train_df


def select_features_fold_change(X_train_fold, y_train_fold, n_features):
    """Select features using fold change (mean difference)."""
    # Calculate fold change
    responder_mean = X_train_fold[y_train_fold == 1].mean()
    non_responder_mean = X_train_fold[y_train_fold == 0].mean()
    fold_change = np.abs(responder_mean - non_responder_mean)
    
    # Select top features
    top_features = fold_change.nlargest(n_features).index.tolist()
    return top_features


def select_features_f_test(X_train_fold, y_train_fold, n_features):
    """Select features using F-test (ANOVA)."""
    selector = SelectKBest(f_classif, k=min(n_features, X_train_fold.shape[1]))
    selector.fit(X_train_fold, y_train_fold)
    return X_train_fold.columns[selector.get_support()].tolist()


def train_models(X_train, y_train, X_val, y_val, class_weights):
    """Train multiple models and return predictions."""
    predictions = {}
    scale_pos_weight = class_weights[0] / class_weights[1]
    
    # 1. Logistic Regression
    lr = LogisticRegression(class_weight=class_weights, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    predictions['logistic'] = lr.predict_proba(X_val)[:, 1]
    
    # 2. Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight=class_weights,
        random_state=42
    )
    rf.fit(X_train, y_train)
    predictions['random_forest'] = rf.predict_proba(X_val)[:, 1]
    
    # 3. XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    predictions['xgboost'] = xgb_model.predict_proba(X_val)[:, 1]
    
    # 4. CatBoost - REMOVED to avoid creating catboost_info folder
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
    
    # 5. LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    predictions['lightgbm'] = lgb_model.predict_proba(X_val)[:, 1]
    
    # 6. Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    gb.fit(X_train, y_train)
    predictions['gradient_boosting'] = gb.predict_proba(X_val)[:, 1]
    
    # 7. SVM
    svm = SVC(
        kernel='rbf',
        probability=True,
        class_weight=class_weights,
        random_state=42
    )
    # Scale features for SVM
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    svm.fit(X_train_scaled, y_train)
    predictions['svm'] = svm.predict_proba(X_val_scaled)[:, 1]
    
    return predictions


def cross_validate_expression(X_data, y_train, sample_ids, class_weights, 
                            data_type='preprocessed', n_features_list=[100, 500, 1000, 2000], 
                            n_folds=5, output_dir=None):
    """Cross-validate expression data with different feature counts."""
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATING {data_type.upper()} EXPRESSION DATA")
    print(f"{'='*70}")
    
    # Ensure data is properly aligned
    X_data = X_data.loc[sample_ids]
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    results = defaultdict(lambda: defaultdict(list))
    
    total_configs = len(n_features_list)
    config_count = 0
    
    for n_features in n_features_list:
        config_count += 1
        print(f"\n\nTesting with {n_features} features ({config_count}/{total_configs}):")
        
        fold_results = defaultdict(list)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_data, y_train)):
            print(f"\n  Fold {fold_idx + 1}/{n_folds}:")
            
            # Split data
            X_train_fold = X_data.iloc[train_idx]
            X_val_fold = X_data.iloc[val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]
            
            # Feature selection (use F-test as requested)
            if n_features < X_train_fold.shape[1]:
                selected_features = select_features_f_test(X_train_fold, y_train_fold, n_features)
                X_train_selected = X_train_fold[selected_features]
                X_val_selected = X_val_fold[selected_features]
            else:
                X_train_selected = X_train_fold
                X_val_selected = X_val_fold
                selected_features = X_train_fold.columns.tolist()
            
            print(f"    Selected {len(selected_features)} features")
            
            # Train models
            predictions = train_models(X_train_selected, y_train_fold, 
                                     X_val_selected, y_val_fold, class_weights)
            
            # Evaluate each model
            for model_name, pred in predictions.items():
                auc = roc_auc_score(y_val_fold, pred)
                fold_results[model_name].append(auc)
                print(f"    {model_name}: AUC = {auc:.3f}")
        
        # Store results for this feature count
        for model_name, aucs in fold_results.items():
            results[n_features][model_name] = {
                'mean': np.mean(aucs),
                'std': np.std(aucs),
                'all_folds': aucs
            }
        
        # Print summary for this feature count
        print(f"\n  Summary for {n_features} features:")
        for model_name, metrics in results[n_features].items():
            print(f"    {model_name}: {metrics['mean']:.3f} ± {metrics['std']:.3f}")
        
        # Removed timing and intermediate file saving
    
    return results


def plot_comparison(preprocessed_results, embedded_results, output_dir):
    """Create comparison plots."""
    plt.figure(figsize=(15, 10))
    
    # Extract data for plotting
    feature_counts = sorted(preprocessed_results.keys())
    models = list(next(iter(preprocessed_results.values())).keys())
    
    # Create subplots for each model
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    for idx, model in enumerate(models):
        plt.subplot(n_rows, n_cols, idx + 1)
        
        # Get means and stds for each feature count
        prep_means = [preprocessed_results[fc][model]['mean'] for fc in feature_counts]
        prep_stds = [preprocessed_results[fc][model]['std'] for fc in feature_counts]
        
        # For embedded, we might only have certain feature counts
        embed_feature_counts = [fc for fc in feature_counts if fc in embedded_results]
        embed_means = [embedded_results[fc][model]['mean'] for fc in embed_feature_counts]
        embed_stds = [embedded_results[fc][model]['std'] for fc in embed_feature_counts]
        
        # Plot
        plt.errorbar(feature_counts, prep_means, yerr=prep_stds, 
                    label='Preprocessed', marker='o', linewidth=2)
        plt.errorbar(embed_feature_counts, embed_means, yerr=embed_stds, 
                    label='Embedded', marker='s', linewidth=2)
        
        plt.xlabel('Number of Features')
        plt.ylabel('AUC')
        plt.title(f'{model}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0.5, 0.8)
    
    plt.suptitle('Expression Data: Preprocessed vs Embedded Performance', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/expression_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary bar plot
    plt.figure(figsize=(12, 8))
    
    # Find best performance for each approach
    best_prep = 0
    best_prep_config = None
    best_embed = 0
    best_embed_config = None
    
    for fc in feature_counts:
        for model in models:
            if fc in preprocessed_results and model in preprocessed_results[fc]:
                auc = preprocessed_results[fc][model]['mean']
                if auc > best_prep:
                    best_prep = auc
                    best_prep_config = (fc, model)
            
            if fc in embedded_results and model in embedded_results[fc]:
                auc = embedded_results[fc][model]['mean']
                if auc > best_embed:
                    best_embed = auc
                    best_embed_config = (fc, model)
    
    # Bar plot
    methods = ['Preprocessed\n(Our Pipeline)', 'Embedded\n(scFoundation)']
    aucs = [best_prep, best_embed]
    colors = ['#3498db', '#e74c3c']
    
    bars = plt.bar(methods, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, auc, config in zip(bars, aucs, [best_prep_config, best_embed_config]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{auc:.3f}\n({config[0]} features, {config[1]})',
                ha='center', va='bottom', fontsize=10)
    
    plt.ylabel('Best AUC', fontsize=14)
    plt.title('Expression Data: Best Performance Comparison', fontsize=16)
    plt.ylim(0, 0.8)
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/expression_best_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function."""
    data_dir = '/Users/tobyliu/bladder'
    output_dir = f'{data_dir}/expression_comparison_results'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data and labels
    y_train, sample_ids, class_weights = load_data_and_labels(data_dir)
    
    # Load both expression datasets
    preprocessed_expr = load_preprocessed_expression(data_dir)
    embedded_expr = load_embedded_expression(data_dir)
    
    # Test different feature counts
    # For preprocessed: test up to available features (~17,689)
    preprocessed_features = [100, 500, 1000, 2000, 3000, 5000]
    # For embedded: limited to 3,072 features
    embedded_features = [100, 500, 1000, 2000, 3000]
    
    print(f"\nTotal experiments to run:")
    print(f"  Preprocessed: {len(preprocessed_features)} feature counts × 5 folds × 7 models = {len(preprocessed_features)*5*7} model fits")
    print(f"  Embedded: {len(embedded_features)} feature counts × 5 folds × 7 models = {len(embedded_features)*5*7} model fits")
    print(f"  Total: {(len(preprocessed_features) + len(embedded_features))*5*7} model fits")
    print(f"\nThis will take several minutes to complete...")
    
    # Removed time tracking
    
    # Cross-validate preprocessed expression
    print("\n" + "="*80)
    print("PART 1: PREPROCESSED EXPRESSION")
    print("="*80)
    preprocessed_results = cross_validate_expression(
        preprocessed_expr, y_train, sample_ids, class_weights,
        data_type='preprocessed', n_features_list=preprocessed_features,
        output_dir=output_dir
    )
    
    # Cross-validate embedded expression
    print("\n" + "="*80)
    print("PART 2: EMBEDDED EXPRESSION")
    print("="*80)
    embedded_results = cross_validate_expression(
        embedded_expr, y_train, sample_ids, class_weights,
        data_type='embedded', n_features_list=embedded_features,
        output_dir=output_dir
    )
    
    # Find best configurations
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    # Best preprocessed
    best_prep_auc = 0
    best_prep_config = None
    for fc, models in preprocessed_results.items():
        for model, metrics in models.items():
            if metrics['mean'] > best_prep_auc:
                best_prep_auc = metrics['mean']
                best_prep_config = (fc, model, metrics['std'])
    
    # Best embedded
    best_embed_auc = 0
    best_embed_config = None
    for fc, models in embedded_results.items():
        for model, metrics in models.items():
            if metrics['mean'] > best_embed_auc:
                best_embed_auc = metrics['mean']
                best_embed_config = (fc, model, metrics['std'])
    
    print(f"\nBest Preprocessed Configuration:")
    print(f"  Features: {best_prep_config[0]}")
    print(f"  Model: {best_prep_config[1]}")
    print(f"  AUC: {best_prep_auc:.3f} ± {best_prep_config[2]:.3f}")
    
    print(f"\nBest Embedded Configuration:")
    print(f"  Features: {best_embed_config[0]}")
    print(f"  Model: {best_embed_config[1]}")
    print(f"  AUC: {best_embed_auc:.3f} ± {best_embed_config[2]:.3f}")
    
    print(f"\nDifference: {best_embed_auc - best_prep_auc:+.3f}")
    
    if best_embed_auc > best_prep_auc:
        print("✓ scFoundation embedding IMPROVED performance")
    else:
        print("✗ Our preprocessing performed BETTER than scFoundation")
    
    # Create plots
    plot_comparison(preprocessed_results, embedded_results, output_dir)
    
    # Save detailed results
    all_results = {
        'preprocessed': {
            fc: {model: {k: v for k, v in metrics.items() if k != 'all_folds'} 
                 for model, metrics in models.items()}
            for fc, models in preprocessed_results.items()
        },
        'embedded': {
            fc: {model: {k: v for k, v in metrics.items() if k != 'all_folds'} 
                 for model, metrics in models.items()}
            for fc, models in embedded_results.items()
        },
        'summary': {
            'best_preprocessed': {
                'features': best_prep_config[0],
                'model': best_prep_config[1],
                'auc_mean': best_prep_auc,
                'auc_std': best_prep_config[2]
            },
            'best_embedded': {
                'features': best_embed_config[0],
                'model': best_embed_config[1],
                'auc_mean': best_embed_auc,
                'auc_std': best_embed_config[2]
            },
            'improvement': best_embed_auc - best_prep_auc
        }
    }
    
    with open(f'{output_dir}/comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Removed runtime tracking
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - comparison_results.json")
    print(f"  - expression_comparison.png")
    print(f"  - expression_best_comparison.png")


if __name__ == "__main__":
    main()


# ok, so I just dont really understand which models were actually used for each modality for us to achieve the .68 AUC score
# like for expression did we use XGboost, random forest or logositc, same for methylation
# than for protein and mutation did we use xgboost, random forest, or gradient, etc
# than expand on the feature selection methods we used, like I dont really understand what fold change, f statistic and fishers exact test are/do
# than also explain to me the fusion methods, Im just confused on how we if we basically just weight each modality a certain weight, how the final AUC is greater than all individual that doesnt make sense to me
# thats it, please answer my questions above thoroughly so I understand the code and the results and what we did, thank you