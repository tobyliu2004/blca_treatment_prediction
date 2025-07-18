#!/usr/bin/env python3
"""
Step 6: Model Training with Cross-Validation (CORRECTED VERSION)
Author: ML Engineers Team
Date: 2025-01-13

CRITICAL FIX: Feature selection is now done INSIDE each CV fold to prevent data leakage

Goal: Train multiple models with proper cross-validation
- Feature selection happens inside each fold using only training data
- No information from validation set is used for any decisions
- This will give more realistic (lower) performance estimates

Models:
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost  
4. SVM
5. Multi-Layer Perceptron (MLP)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
from scipy.stats import fisher_exact
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')
import os
from collections import defaultdict
import json


class MLP(nn.Module):
    """Multi-Layer Perceptron for binary classification."""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


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
    print(f"  Class weights: {class_weights}")
    
    return y_train, sample_ids, class_weights


def load_modality_data(modality, data_dir='/Users/tobyliu/bladder/preprocessed_data'):
    """Load preprocessed data for a specific modality."""
    train_file = f'{data_dir}/{modality}/{modality}_train_preprocessed.csv'
    X_train = pd.read_csv(train_file, index_col=0)
    return X_train


def select_features_inside_fold(X_train_fold, y_train_fold, modality, method='f_test', k=100):
    """
    Select features using only the training fold data.
    This prevents data leakage by ensuring validation data is never used for selection.
    """
    
    # Handle 'none' method - use all features
    if method == 'none' or k == -1:
        return X_train_fold.columns.tolist()
    
    if modality == 'mutation':
        # For mutation data, handle special columns
        special_cols = ['selected_mutation_burden', 'total_mutation_burden', 
                       'RTK_RAS_pathway', 'PI3K_AKT_pathway', 'Cell_Cycle_TP53_pathway',
                       'Chromatin_Remodeling_pathway', 'DNA_Repair_pathway']
        
        # Separate special columns from gene columns
        gene_cols = [col for col in X_train_fold.columns if col not in special_cols]
        X_genes = X_train_fold[gene_cols]
        
        if len(gene_cols) > 0 and k < len(X_train_fold.columns):
            if method == 'fisher':
                # Fisher's exact test for mutations
                selected_features = fisher_test_selection(X_genes, y_train_fold, k=min(k, len(gene_cols)))
            else:
                # Default to chi2 for mutations
                selector = SelectKBest(chi2, k=min(k, len(gene_cols)))
                selector.fit(X_genes, y_train_fold)
                selected_indices = selector.get_support(indices=True)
                selected_features = [gene_cols[i] for i in selected_indices]
            
            # Always include special columns
            selected_features.extend([col for col in special_cols if col in X_train_fold.columns])
            return selected_features
        else:
            return X_train_fold.columns.tolist()
    
    elif modality == 'methylation' and method == 'multi_stage':
        # Multi-stage filtering for methylation (all inside this fold!)
        return multi_stage_methylation_selection(X_train_fold, y_train_fold, final_k=k)
    
    else:
        # Standard feature selection for other cases
        if method == 'f_test':
            selector = SelectKBest(f_classif, k=min(k, X_train_fold.shape[1]))
        elif method == 'fold_change':
            return fold_change_selection(X_train_fold, y_train_fold, target_k=k)
        elif method == 'lasso':
            return lasso_feature_selection(X_train_fold, y_train_fold, target_k=k)
        else:
            selector = SelectKBest(f_classif, k=min(k, X_train_fold.shape[1]))
        
        if method not in ['fold_change', 'lasso']:
            selector.fit(X_train_fold, y_train_fold)
            selected_indices = selector.get_support(indices=True)
            return X_train_fold.columns[selected_indices].tolist()
        else:
            return X_train_fold.columns.tolist()  # Already handled above


def multi_stage_methylation_selection(X_train_fold, y_train_fold, 
                                    p_threshold=0.001, corr_threshold=0.95, final_k=100):
    """
    Multi-stage feature selection for methylation data.
    ALL STEPS use only the current training fold to prevent leakage.
    """
    
    # Stage 1: Aggressive p-value filtering
    p_values = []
    for cpg in X_train_fold.columns:
        values = X_train_fold[cpg].values
        values_pos = values[y_train_fold == 1]
        values_neg = values[y_train_fold == 0]
        
        try:
            _, p_val = stats.ttest_ind(values_pos, values_neg)
            p_values.append(p_val)
        except:
            p_values.append(1.0)
    
    p_values = np.array(p_values)
    stage1_mask = p_values < p_threshold
    stage1_features = X_train_fold.columns[stage1_mask].tolist()
    
    if len(stage1_features) == 0:
        # If too aggressive, relax threshold
        stage1_mask = p_values < 0.01
        stage1_features = X_train_fold.columns[stage1_mask].tolist()
    
    # Stage 2: Remove highly correlated features
    X_stage1 = X_train_fold[stage1_features]
    if len(stage1_features) > 1:
        corr_matrix = X_stage1.corr().abs()
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        high_corr_pairs = np.where((corr_matrix.values > corr_threshold) & upper_triangle)
        
        features_to_remove = set()
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            # Remove the one with higher p-value (less significant)
            if p_values[stage1_mask][i] > p_values[stage1_mask][j]:
                features_to_remove.add(stage1_features[i])
            else:
                features_to_remove.add(stage1_features[j])
        
        stage2_features = [f for f in stage1_features if f not in features_to_remove]
    else:
        stage2_features = stage1_features
    
    # Stage 3: Final selection using F-test
    if len(stage2_features) > final_k:
        X_stage2 = X_train_fold[stage2_features]
        selector = SelectKBest(f_classif, k=final_k)
        selector.fit(X_stage2, y_train_fold)
        selected_indices = selector.get_support(indices=True)
        final_features = [stage2_features[i] for i in selected_indices]
    else:
        final_features = stage2_features
    
    return final_features


def lasso_feature_selection(X_train_fold, y_train_fold, target_k=100):
    """LASSO feature selection using only training fold data."""
    # Standardize for LASSO
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_fold)
    
    # Use LassoCV to find optimal alpha
    lasso = LassoCV(cv=3, random_state=42, max_iter=10000)
    lasso.fit(X_scaled, y_train_fold)
    
    # Get non-zero coefficients
    feature_importance = np.abs(lasso.coef_)
    
    # Select top k features
    if np.sum(feature_importance > 0) < target_k:
        # If LASSO selected fewer than target, take all non-zero
        selected_indices = np.where(feature_importance > 0)[0]
    else:
        # Otherwise, take top k by importance
        selected_indices = np.argsort(feature_importance)[-target_k:][::-1]
    
    return X_train_fold.columns[selected_indices].tolist()


def fold_change_selection(X_train_fold, y_train_fold, target_k=100, threshold=0.5):
    """Select features based on fold change between responders and non-responders."""
    # Calculate mean expression for each group
    responder_mean = X_train_fold[y_train_fold == 1].mean()
    non_responder_mean = X_train_fold[y_train_fold == 0].mean()
    
    # Calculate absolute fold change
    fold_change = np.abs(responder_mean - non_responder_mean)
    
    # Sort by fold change and select top k
    top_features = fold_change.nlargest(target_k).index.tolist()
    
    return top_features


def fisher_test_selection(X_train_fold, y_train_fold, k=50):
    """Fisher's exact test for binary features (mutations)."""
    p_values = []
    
    for gene in X_train_fold.columns:
        # Create 2x2 contingency table
        mutated_responders = np.sum((X_train_fold[gene] == 1) & (y_train_fold == 1))
        mutated_non_responders = np.sum((X_train_fold[gene] == 1) & (y_train_fold == 0))
        wild_responders = np.sum((X_train_fold[gene] == 0) & (y_train_fold == 1))
        wild_non_responders = np.sum((X_train_fold[gene] == 0) & (y_train_fold == 0))
        
        # Run Fisher's exact test
        try:
            _, p_value = fisher_exact([[wild_non_responders, wild_responders],
                                      [mutated_non_responders, mutated_responders]])
            p_values.append((gene, p_value))
        except:
            p_values.append((gene, 1.0))
    
    # Sort by p-value and select top k
    p_values.sort(key=lambda x: x[1])
    selected_features = [gene for gene, _ in p_values[:k]]
    
    return selected_features


def train_traditional_models(X_train, y_train, X_val, y_val, class_weights):
    """Train traditional ML models."""
    results = {}
    models = {}
    
    # Logistic Regression
    lr = LogisticRegression(class_weight=class_weights, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    results['logistic_regression'] = lr.predict_proba(X_val)[:, 1]
    models['logistic_regression'] = lr
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight=class_weights, 
                               random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results['random_forest'] = rf.predict_proba(X_val)[:, 1]
    models['random_forest'] = rf
    
    # XGBoost
    scale_pos_weight = class_weights[0] / class_weights[1]
    xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, 
                                  random_state=42, n_jobs=-1, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    results['xgboost'] = xgb_model.predict_proba(X_val)[:, 1]
    models['xgboost'] = xgb_model
    
    # SVM
    svm = SVC(kernel='rbf', class_weight=class_weights, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    results['svm'] = svm.predict_proba(X_val)[:, 1]
    models['svm'] = svm
    
    return results, models


def train_mlp(X_train, y_train, X_val, y_val, class_weights, 
              input_dim, epochs=100, batch_size=32, learning_rate=0.001):
    """Train MLP model."""
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val.values if hasattr(X_val, 'values') else X_val)
    
    # Create data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = MLP(input_dim)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val_tensor).squeeze().numpy()
    
    return y_pred, model


def evaluate_predictions(y_true, y_pred, threshold=0.5):
    """Evaluate model predictions."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    metrics = {
        'auc': roc_auc_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0)
    }
    
    return metrics


def cross_validate_modality(X_data, y_train, modality, sample_ids, 
                          class_weights, n_folds=5):
    """
    Perform PROPER cross-validation with feature selection inside each fold.
    This is the CORRECTED version that prevents data leakage.
    """
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATING {modality.upper()} (with proper feature selection)")
    print(f"{'='*70}")
    
    # Initialize results storage
    cv_results = defaultdict(lambda: defaultdict(list))
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Define feature selection configurations based on modality
    # Now includes supervised methods that were previously in preprocessing
    if modality == 'expression':
        # Expression now has more features after removing fold change filter
        configs = [
            ('f_test_100', 'f_test', 100),
            ('f_test_300', 'f_test', 300),
            ('f_test_500', 'f_test', 500),
            ('fold_change_100', 'fold_change', 100),
            ('fold_change_300', 'fold_change', 300),
            ('lasso_100', 'lasso', 100),
            ('all_features', 'none', -1)
        ]
    elif modality == 'mutation':
        # Mutation now has more features without Fisher's test
        configs = [
            ('fisher_20', 'fisher', 20),
            ('fisher_50', 'fisher', 50),
            ('fisher_100', 'fisher', 100),
            ('chi2_50', 'chi2', 50),
            ('all_features', 'none', -1)
        ]
    elif modality == 'protein':
        # Protein has 185 features
        configs = [
            ('f_test_50', 'f_test', 50),
            ('f_test_100', 'f_test', 100),
            ('lasso_50', 'lasso', 50),
            ('all_features', 'none', -1)  # Use all 185 features
        ]
    elif modality == 'methylation':
        # Methylation now has more features without fold change filter
        configs = [
            ('f_test_100', 'f_test', 100),
            ('f_test_300', 'f_test', 300),
            ('f_test_500', 'f_test', 500),
            ('fold_change_100', 'fold_change', 100),
            ('fold_change_300', 'fold_change', 300),
            ('lasso_100', 'lasso', 100),
            ('multi_stage_100', 'multi_stage', 100),
            ('all_features', 'none', -1)
        ]
    
    # Test each configuration
    for config_name, method, k in configs:
        print(f"\n{config_name}:")
        
        # Cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_data, y_train)):
            print(f"  Fold {fold_idx + 1}/{n_folds}")
            
            # Split data
            X_train_fold = X_data.iloc[train_idx]
            X_val_fold = X_data.iloc[val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]
            
            # CRITICAL: Feature selection using ONLY training fold
            selected_features = select_features_inside_fold(
                X_train_fold, y_train_fold, modality, method, k
            )
            
            print(f"    Selected {len(selected_features)} features")
            
            if len(selected_features) < 2:
                print("    Too few features - skipping")
                continue
            
            # Apply feature selection to both folds
            X_train_selected = X_train_fold[selected_features]
            X_val_selected = X_val_fold[selected_features]
            
            # Train models
            print("    Training models...")
            trad_predictions, _ = train_traditional_models(
                X_train_selected, y_train_fold, 
                X_val_selected, y_val_fold, 
                class_weights
            )
            
            mlp_pred, _ = train_mlp(
                X_train_selected, y_train_fold,
                X_val_selected, y_val_fold,
                class_weights,
                input_dim=X_train_selected.shape[1]
            )
            
            # Evaluate
            all_predictions = {**trad_predictions, 'mlp': mlp_pred}
            
            for model_name, predictions in all_predictions.items():
                metrics = evaluate_predictions(y_val_fold, predictions)
                
                for metric_name, metric_value in metrics.items():
                    cv_results[f"{config_name}_{model_name}"][metric_name].append(metric_value)
    
    # Calculate summary statistics
    summary_results = {}
    for config_name, metrics in cv_results.items():
        summary_results[config_name] = {}
        for metric_name, values in metrics.items():
            summary_results[config_name][f"{metric_name}_mean"] = np.mean(values)
            summary_results[config_name][f"{metric_name}_std"] = np.std(values)
    
    return summary_results


def main():
    """Main function for corrected model training."""
    data_dir = '/Users/tobyliu/bladder'
    
    print("="*80)
    print("CORRECTED MODEL TRAINING - Feature Selection Inside CV Folds")
    print("="*80)
    print("\nThis version properly prevents data leakage by:")
    print("- Selecting features using ONLY training fold data")
    print("- Never using validation data for any selection decisions")
    print("- Expecting more realistic (lower) performance metrics")
    
    # Load data
    y_train, sample_ids, class_weights = load_data_and_labels(data_dir)
    
    # Store results
    all_results = {}
    
    # Process each modality
    modalities = ['expression', 'mutation', 'protein', 'methylation']
    
    for modality in modalities:
        print(f"\n{'='*80}")
        print(f"PROCESSING {modality.upper()}")
        print(f"{'='*80}")
        
        # Load data
        X_data = load_modality_data(modality)
        X_data = X_data.loc[sample_ids]
        
        # Cross-validate with proper feature selection
        modality_results = cross_validate_modality(
            X_data, y_train, modality, sample_ids, class_weights
        )
        
        # Save results
        all_results[modality] = modality_results

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF BEST PERFORMING MODELS (CORRECTED)")
    print("="*80)
    
    for modality, results in all_results.items():
        print(f"\n{modality.upper()}:")
        
        # Find best by AUC
        best_config = None
        best_auc = 0
        
        for config_name, metrics in results.items():
            if 'auc_mean' in metrics and metrics['auc_mean'] > best_auc:
                best_auc = metrics['auc_mean']
                best_config = config_name
        
        if best_config:
            print(f"  Best: {best_config}")
            print(f"  AUC: {results[best_config]['auc_mean']:.3f} ± {results[best_config]['auc_std']:.3f}")
            print(f"  Accuracy: {results[best_config]['accuracy_mean']:.3f} ± {results[best_config]['accuracy_std']:.3f}")
            print(f"  F1: {results[best_config]['f1_mean']:.3f} ± {results[best_config]['f1_std']:.3f}")
    
    print("\nCorrected model training completed!")

if __name__ == "__main__":
    main()