#!/usr/bin/env python3
"""
Step 6: COMPREHENSIVE Model Training with Extensive Cross-Validation
Author: ML Engineers Team
Date: 2025-01-17

COMPREHENSIVE VERSION: Tests many more feature configurations and models to find optimal performance

Goal: Extensively test feature selection methods and model combinations
- Feature selection happens inside each fold using only training data
- Tests multiple feature counts per modality based on actual preprocessed data sizes
- Includes t-test and mutual information feature selection
- Tests more models including LightGBM and GradientBoosting

Models:
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost  
4. SVM
5. Multi-Layer Perceptron (MLP) - 3 architectures
6. LightGBM
7. Gradient Boosting
8. ElasticNet
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
from scipy.stats import fisher_exact
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')
import os
from collections import defaultdict
import json
import time
from datetime import datetime


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
        # Remove Sigmoid for BCEWithLogitsLoss
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# Define different MLP architectures
def get_mlp_architecture(arch_type='standard', input_dim=100):
    """Get MLP with specified architecture."""
    if arch_type == 'shallow':
        return MLP(input_dim, hidden_dims=[128, 64], dropout_rate=0.2)
    elif arch_type == 'standard':
        return MLP(input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3)
    elif arch_type == 'deep':
        return MLP(input_dim, hidden_dims=[512, 256, 128, 64], dropout_rate=0.4)
    else:
        return MLP(input_dim)


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
        elif method == 't_test':
            return t_test_selection(X_train_fold, y_train_fold, k=k)
        elif method == 'mutual_info':
            return mutual_info_selection(X_train_fold, y_train_fold, k=k)
        else:
            selector = SelectKBest(f_classif, k=min(k, X_train_fold.shape[1]))
        
        if method not in ['fold_change', 'lasso', 't_test', 'mutual_info']:
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
        except (ValueError, ZeroDivisionError) as e:
            print(f"      Warning: t-test failed for {cpg}: {str(e)}")
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
    from sklearn.linear_model import LogisticRegression
    
    # Standardize for LASSO
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_fold)
    
    # Use LogisticRegression with L1 penalty for classification
    # Start with C=1.0 and adjust if needed
    lasso = LogisticRegression(penalty='l1', solver='liblinear', 
                              C=1.0, random_state=42, max_iter=10000)
    try:
        lasso.fit(X_scaled, y_train_fold)
        feature_importance = np.abs(lasso.coef_[0])
    except:
        # If convergence fails, try with higher C (less regularization)
        lasso = LogisticRegression(penalty='l1', solver='liblinear', 
                                  C=10.0, random_state=42, max_iter=10000)
        lasso.fit(X_scaled, y_train_fold)
        feature_importance = np.abs(lasso.coef_[0])
    
    # Select top k features
    if np.sum(feature_importance > 0) < target_k:
        # If LASSO selected fewer than target, take all non-zero
        selected_indices = np.where(feature_importance > 0)[0]
        # If LASSO selected NO features (all coefficients = 0), fall back to top k by magnitude
        if len(selected_indices) == 0:
            print(f"      WARNING: LASSO selected 0 features, using top {min(target_k, 50)} by magnitude")
            selected_indices = np.argsort(feature_importance)[-min(target_k, 50):][::-1]
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
        except (ValueError, ZeroDivisionError) as e:
            print(f"      Warning: Fisher test failed for {gene}: {str(e)}")
            p_values.append((gene, 1.0))
    
    # Sort by p-value and select top k
    p_values.sort(key=lambda x: x[1])
    selected_features = [gene for gene, _ in p_values[:k]]
    
    return selected_features


def t_test_selection(X_train_fold, y_train_fold, k=100):
    """T-test feature selection for continuous features."""
    t_scores = []
    
    for feature in X_train_fold.columns:
        values = X_train_fold[feature].values
        values_pos = values[y_train_fold == 1]
        values_neg = values[y_train_fold == 0]
        
        try:
            # Calculate t-statistic
            t_stat, p_val = stats.ttest_ind(values_pos, values_neg)
            t_scores.append((feature, abs(t_stat)))
        except (ValueError, ZeroDivisionError) as e:
            print(f"      Warning: t-test failed for {feature}: {str(e)}")
            t_scores.append((feature, 0))
    
    # Sort by absolute t-statistic and select top k
    t_scores.sort(key=lambda x: x[1], reverse=True)
    selected_features = [feature for feature, _ in t_scores[:k]]
    
    return selected_features


def mutual_info_selection(X_train_fold, y_train_fold, k=100):
    """Mutual information feature selection."""
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X_train_fold, y_train_fold, random_state=42)
    
    # Get indices of top k features
    top_indices = np.argsort(mi_scores)[-k:][::-1]
    
    # Return selected feature names
    return X_train_fold.columns[top_indices].tolist()


def train_all_models(X_train, y_train, X_val, y_val, class_weights):
    """Train all models including new additions."""
    results = {}
    models = {}
    
    # Scale data for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 1. Logistic Regression
    lr = LogisticRegression(class_weight=class_weights, max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    results['logistic_regression'] = lr.predict_proba(X_val_scaled)[:, 1]
    models['logistic_regression'] = lr
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight=class_weights, 
                               random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results['random_forest'] = rf.predict_proba(X_val)[:, 1]
    models['random_forest'] = rf
    
    # 3. XGBoost
    scale_pos_weight = class_weights[0] / class_weights[1]
    xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, 
                                  random_state=42, n_jobs=-1, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    results['xgboost'] = xgb_model.predict_proba(X_val)[:, 1]
    models['xgboost'] = xgb_model
    
    # 4. LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=100, num_leaves=31,
                                   scale_pos_weight=scale_pos_weight,
                                   random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)
    results['lightgbm'] = lgb_model.predict_proba(X_val)[:, 1]
    models['lightgbm'] = lgb_model
    
    # 6. Gradient Boosting (Note: doesn't support class_weight directly)
    # We'll use sample_weight instead to handle class imbalance
    # IMPORTANT: Sample weights should be normalized to sum to n_samples
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train == 0] = class_weights[0]
    sample_weights[y_train == 1] = class_weights[1]
    # Normalize to sum to n_samples (standard practice)
    sample_weights = sample_weights * len(y_train) / sample_weights.sum()
    
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                   max_depth=4, random_state=42)
    gb.fit(X_train, y_train, sample_weight=sample_weights)
    results['gradient_boosting'] = gb.predict_proba(X_val)[:, 1]
    models['gradient_boosting'] = gb
    
    # 7. SVM
    svm = SVC(kernel='rbf', class_weight=class_weights, probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    results['svm'] = svm.predict_proba(X_val_scaled)[:, 1]
    models['svm'] = svm
    
    # 8. Logistic Regression with ElasticNet penalty (proper for classification)
    from sklearn.linear_model import LogisticRegressionCV
    try:
        elastic = LogisticRegressionCV(penalty='elasticnet', solver='saga', l1_ratios=[0.5],
                                      cv=3, random_state=42, max_iter=10000, n_jobs=-1,
                                      tol=0.001)  # Increased tolerance for convergence
        elastic.fit(X_train_scaled, y_train)
        results['elasticnet'] = elastic.predict_proba(X_val_scaled)[:, 1]
        models['elasticnet'] = elastic
    except (ValueError, RuntimeError) as e:
        print(f"    ElasticNet failed to converge: {str(e)}")
        print("    Falling back to standard LogisticRegression")
        # Fallback to standard logistic regression
        elastic = LogisticRegression(penalty='l2', random_state=42, max_iter=10000)
        elastic.fit(X_train_scaled, y_train)
        results['elasticnet'] = elastic.predict_proba(X_val_scaled)[:, 1]
        models['elasticnet'] = elastic
    
    return results, models


def train_mlp(X_train, y_train, X_val, y_val, class_weights, 
              input_dim, arch_type='standard', epochs=100, batch_size=32, learning_rate=0.001):
    """Train MLP model with specified architecture."""
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val.values if hasattr(X_val, 'values') else X_val)
    
    # Create data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model with specified architecture
    model = get_mlp_architecture(arch_type, input_dim)
    
    # Loss with class weights
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
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
        logits = model(X_val_tensor).squeeze()
        y_pred = torch.sigmoid(logits).numpy()  # Apply sigmoid for probabilities
    
    return y_pred, model


def evaluate_predictions(y_true, y_pred, threshold=0.5):
    """Evaluate model predictions."""
    # Safety check for valid predictions
    if len(y_pred) == 0 or len(y_true) == 0:
        print("    WARNING: Empty predictions or labels")
        return {'auc': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    # Check for NaN or infinite values
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        print("    WARNING: NaN or Inf in predictions - replacing with 0.5")
        y_pred = np.nan_to_num(y_pred, nan=0.5, posinf=1.0, neginf=0.0)
    
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    try:
        metrics = {
            'auc': roc_auc_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary),
            'f1': f1_score(y_true, y_pred_binary, zero_division=0)
        }
    except ValueError as e:
        print(f"    WARNING: Evaluation failed: {str(e)}")
        metrics = {'auc': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
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
    
    # Define COMPREHENSIVE feature selection configurations
    # Expression: 17,689 features available
    if modality == 'expression':
        configs = [
            # F-test with various feature counts
            ('f_test_100', 'f_test', 100),
            ('f_test_300', 'f_test', 300),
            ('f_test_500', 'f_test', 500),
            ('f_test_750', 'f_test', 750),
            ('f_test_1000', 'f_test', 1000),
            ('f_test_1500', 'f_test', 1500),
            ('f_test_2000', 'f_test', 2000),  # Previous best
            ('f_test_2500', 'f_test', 2500),
            ('f_test_3000', 'f_test', 3000),
            ('f_test_5000', 'f_test', 5000),
            ('f_test_7500', 'f_test', 7500),
            ('f_test_10000', 'f_test', 10000),
            # T-test (new)
            ('t_test_1000', 't_test', 1000),
            ('t_test_2000', 't_test', 2000),
            ('t_test_3000', 't_test', 3000),
            ('t_test_5000', 't_test', 5000),
            # Other methods
            ('fold_change_1000', 'fold_change', 1000),
            ('fold_change_2000', 'fold_change', 2000),
            ('lasso_1000', 'lasso', 1000),
            ('lasso_2000', 'lasso', 2000),
            ('mutual_info_2000', 'mutual_info', 2000),
            ('all_features', 'none', -1)
        ]
    # Mutation: 1,725 features available (CORRECTED)
    elif modality == 'mutation':
        configs = [
            # Fisher's test with more feature counts
            ('fisher_50', 'fisher', 50),
            ('fisher_100', 'fisher', 100),
            ('fisher_200', 'fisher', 200),
            ('fisher_300', 'fisher', 300),  # Previous best
            ('fisher_400', 'fisher', 400),
            ('fisher_500', 'fisher', 500),
            ('fisher_750', 'fisher', 750),
            ('fisher_1000', 'fisher', 1000),
            # Chi-squared test
            ('chi2_100', 'chi2', 100),
            ('chi2_300', 'chi2', 300),
            ('chi2_500', 'chi2', 500),
            # T-test (new)
            ('t_test_300', 't_test', 300),
            ('t_test_500', 't_test', 500),
            # Mutual info
            ('mutual_info_300', 'mutual_info', 300),
            ('all_features', 'none', -1)
        ]
    # Protein: 185 features available
    elif modality == 'protein':
        configs = [
            # F-test
            ('f_test_50', 'f_test', 50),
            ('f_test_75', 'f_test', 75),
            ('f_test_100', 'f_test', 100),  # Previous best
            ('f_test_125', 'f_test', 125),
            ('f_test_150', 'f_test', 150),
            # T-test (new)
            ('t_test_75', 't_test', 75),
            ('t_test_100', 't_test', 100),
            ('t_test_125', 't_test', 125),
            # Other methods
            ('lasso_100', 'lasso', 100),
            ('mutual_info_100', 'mutual_info', 100),
            ('all_features', 'none', -1)
        ]
    # Methylation: 39,575 features available
    elif modality == 'methylation':
        configs = [
            # F-test with extensive feature counts
            ('f_test_500', 'f_test', 500),
            ('f_test_1000', 'f_test', 1000),
            ('f_test_2000', 'f_test', 2000),
            ('f_test_3000', 'f_test', 3000),
            ('f_test_5000', 'f_test', 5000),  # Previous best
            ('f_test_7500', 'f_test', 7500),
            ('f_test_10000', 'f_test', 10000),
            ('f_test_15000', 'f_test', 15000),
            ('f_test_20000', 'f_test', 20000),
            # T-test (new)
            ('t_test_3000', 't_test', 3000),
            ('t_test_5000', 't_test', 5000),
            ('t_test_7500', 't_test', 7500),
            ('t_test_10000', 't_test', 10000),
            # Other methods
            ('fold_change_3000', 'fold_change', 3000),
            ('fold_change_5000', 'fold_change', 5000),
            ('lasso_3000', 'lasso', 3000),
            ('lasso_5000', 'lasso', 5000),
            ('multi_stage_5000', 'multi_stage', 5000),
            ('mutual_info_5000', 'mutual_info', 5000),
            # REMOVED all_features for methylation to prevent memory crashes
            # 39,575 features would likely cause OOM errors
            # Maximum tested: 20,000 features
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
            
            # CRITICAL: Check for single-class folds
            if len(np.unique(y_train_fold)) < 2 or len(np.unique(y_val_fold)) < 2:
                print(f"    WARNING: Single class in fold - skipping to avoid errors")
                print(f"    Train classes: {np.unique(y_train_fold)}, Val classes: {np.unique(y_val_fold)}")
                continue
            
            # CRITICAL: Feature selection using ONLY training fold
            selected_features = select_features_inside_fold(
                X_train_fold, y_train_fold, modality, method, k
            )
            
            print(f"    Selected {len(selected_features)} features")
            
            if len(selected_features) < 2:
                print("    Too few features - skipping")
                continue
            
            # Additional safety check
            if len(selected_features) > 20000:
                print(f"    WARNING: {len(selected_features)} features may cause memory issues!")
                print("    Consider reducing feature count or monitoring memory usage")
            
            # Apply feature selection to both folds
            X_train_selected = X_train_fold[selected_features]
            X_val_selected = X_val_fold[selected_features]
            
            # Train all models
            print("    Training models...")
            all_predictions, _ = train_all_models(
                X_train_selected, y_train_fold, 
                X_val_selected, y_val_fold, 
                class_weights
            )
            
            # Train MLPs with different architectures
            for arch_type in ['shallow', 'standard', 'deep']:
                mlp_pred, _ = train_mlp(
                    X_train_selected, y_train_fold,
                    X_val_selected, y_val_fold,
                    class_weights,
                    input_dim=X_train_selected.shape[1],
                    arch_type=arch_type
                )
                all_predictions[f'mlp_{arch_type}'] = mlp_pred
            
            # Evaluate all models
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
    """Main function for COMPREHENSIVE model training."""
    data_dir = '/Users/tobyliu/bladder'
    
    print("="*80)
    print("COMPREHENSIVE MODEL TRAINING - Extensive Feature & Model Testing")
    print("="*80)
    print("\nThis version tests:")
    print("- Multiple feature selection methods including t-test")
    print("- Many more feature count configurations")
    print("- 8 different models including LightGBM, XGBoost, etc.")
    print("- 3 different MLP architectures")
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    y_train, sample_ids, class_weights = load_data_and_labels(data_dir)
    
    # Store results
    all_results = {}
    
    # Create checkpoint directory
    checkpoint_dir = f'{data_dir}/comprehensive_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Process each modality
    modalities = ['expression', 'mutation', 'protein', 'methylation']
    
    start_time = time.time()
    
    # Add garbage collection for memory management
    import gc
    
    for modality_idx, modality in enumerate(modalities):
        modality_start = time.time()
        print(f"\n{'='*80}")
        print(f"PROCESSING {modality.upper()} ({modality_idx+1}/{len(modalities)})")
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
        
        # Save checkpoint after each modality
        checkpoint_file = f'{checkpoint_dir}/results_{modality}.pkl'
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(modality_results, f)
        print(f"\nCheckpoint saved: {checkpoint_file}")
        
        # Time tracking
        modality_time = time.time() - modality_start
        print(f"{modality.upper()} completed in {modality_time/60:.1f} minutes")
        
        # Estimate remaining time
        if modality_idx < len(modalities) - 1:
            avg_time_per_modality = (time.time() - start_time) / (modality_idx + 1)
            remaining_modalities = len(modalities) - modality_idx - 1
            estimated_remaining = avg_time_per_modality * remaining_modalities
            print(f"Estimated time remaining: {estimated_remaining/60:.1f} minutes")
        
        # Force garbage collection after each modality to free memory
        gc.collect()

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
    
    # Save final comprehensive results
    final_results_file = f'{data_dir}/comprehensive_results.json'
    
    # Convert results to JSON-serializable format
    json_results = {}
    for modality, results in all_results.items():
        json_results[modality] = {}
        for config, metrics in results.items():
            json_results[modality][config] = metrics
    
    with open(final_results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time/60:.1f} minutes")
    print(f"Results saved to: {final_results_file}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nComprehensive model training completed!")

if __name__ == "__main__":
    main()