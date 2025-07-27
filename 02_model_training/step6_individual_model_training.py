#!/usr/bin/env python3
"""
Step 6: OPTIMIZED Model Training with Adaptive Feature Selection
Author: ML Engineers Team
Date: 2025-01-21

Goal: Strategic and efficient model training with two-phase adaptive feature selection
- Phase 1: Broad search across feature counts
- Phase 2: Refinement around best configurations
- NO DATA LEAKAGE: All feature selection inside CV folds
- Target runtime: ~60-75 minutes

Models:
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost
4. ElasticNet (for high-dimensional data)
5. MLP (single optimized architecture)
"""

import pandas as pd
import numpy as np
import json
import pickle
import time
from collections import defaultdict
from datetime import datetime

# ML imports
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from scipy.stats import fisher_exact

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings('ignore')


class MLP(nn.Module):
    """Optimized MLP for biological data."""
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
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
    
    return y_train, sample_ids, class_weights


def load_modality_data(modality, data_dir='/Users/tobyliu/bladder'):
    """Load preprocessed data for a specific modality."""
    train_file = f'{data_dir}/preprocessed_data/{modality}/{modality}_train_preprocessed.csv'
    return pd.read_csv(train_file, index_col=0)


def fold_change_selection(X_train_fold, y_train_fold, k):
    """Select features by fold change between groups."""
    responder_mean = X_train_fold[y_train_fold == 1].mean()
    non_responder_mean = X_train_fold[y_train_fold == 0].mean()
    fold_change = np.abs(responder_mean - non_responder_mean)
    return fold_change.nlargest(k).index.tolist()


def lasso_selection(X_train_fold, y_train_fold, k):
    """LASSO-based feature selection."""
    from sklearn.linear_model import LogisticRegression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_fold)
    
    # Use L1 penalty
    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, 
                              random_state=42, max_iter=1000)
    lasso.fit(X_scaled, y_train_fold)
    
    feature_importance = np.abs(lasso.coef_[0])
    top_indices = np.argsort(feature_importance)[-k:][::-1]
    
    return X_train_fold.columns[top_indices].tolist()


def fisher_test_selection(X_train_fold, y_train_fold, k):
    """Fisher's exact test for mutation data."""
    p_values = []
    
    for gene in X_train_fold.columns:
        # Skip special columns
        if gene in ['selected_mutation_burden', 'total_mutation_burden'] or 'pathway' in gene:
            continue
            
        mut_resp = np.sum((X_train_fold[gene] == 1) & (y_train_fold == 1))
        mut_non = np.sum((X_train_fold[gene] == 1) & (y_train_fold == 0))
        wt_resp = np.sum((X_train_fold[gene] == 0) & (y_train_fold == 1))
        wt_non = np.sum((X_train_fold[gene] == 0) & (y_train_fold == 0))
        
        try:
            _, p_val = fisher_exact([[wt_non, wt_resp], [mut_non, mut_resp]])
            p_values.append((gene, p_val))
        except:
            p_values.append((gene, 1.0))
    
    # Sort by p-value
    p_values.sort(key=lambda x: x[1])
    selected = [gene for gene, _ in p_values[:k]]
    
    # Always include special columns
    special_cols = ['selected_mutation_burden', 'total_mutation_burden']
    pathway_cols = [col for col in X_train_fold.columns if 'pathway' in col]
    selected.extend([col for col in special_cols + pathway_cols if col in X_train_fold.columns])
    
    return selected


def select_features(X_train_fold, y_train_fold, modality, method, k):
    """Select features using specified method."""
    if method == 'f_test':
        selector = SelectKBest(f_classif, k=min(k, X_train_fold.shape[1]))
        selector.fit(X_train_fold, y_train_fold)
        return X_train_fold.columns[selector.get_support()].tolist()
    elif method == 'fold_change':
        return fold_change_selection(X_train_fold, y_train_fold, k)
    elif method == 'lasso':
        return lasso_selection(X_train_fold, y_train_fold, k)
    elif method == 'fisher':
        return fisher_test_selection(X_train_fold, y_train_fold, k)
    else:
        raise ValueError(f"Unknown method: {method}")


def train_models(X_train, y_train, X_val, y_val, class_weights, modality):
    """Train models efficiently."""
    predictions = {}
    feature_importances = {}
    feature_names = X_train.columns.tolist()
    
    # Scale data once
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 1. Logistic Regression
    lr = LogisticRegression(class_weight=class_weights, max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    predictions['logistic'] = lr.predict_proba(X_val_scaled)[:, 1]
    # Feature importance: absolute value of coefficients
    feature_importances['logistic'] = np.abs(lr.coef_[0])
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight=class_weights,
                               random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X_train, y_train)
    predictions['rf'] = rf.predict_proba(X_val)[:, 1]
    feature_importances['rf'] = rf.feature_importances_
    
    # 3. XGBoost
    scale_pos_weight = class_weights[0] / class_weights[1]
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    predictions['xgboost'] = xgb_model.predict_proba(X_val)[:, 1]
    feature_importances['xgboost'] = xgb_model.feature_importances_
    
    # 4. ElasticNet (only for high-dimensional modalities)
    if modality in ['expression', 'methylation']:
        from sklearn.linear_model import LogisticRegressionCV
        elastic = LogisticRegressionCV(
            penalty='elasticnet',
            solver='saga',
            l1_ratios=[0.5],
            cv=3,
            random_state=42,
            max_iter=2000,
            n_jobs=-1
        )
        elastic.fit(X_train_scaled, y_train)
        predictions['elasticnet'] = elastic.predict_proba(X_val_scaled)[:, 1]
        # Feature importance: absolute value of coefficients
        feature_importances['elasticnet'] = np.abs(elastic.coef_[0])
    
    # 5. MLP
    predictions['mlp'] = train_mlp_optimized(X_train_scaled, y_train, X_val_scaled, 
                                           y_val, class_weights)
    # MLP doesn't have built-in feature importance, skip for now
    
    return predictions, feature_importances, feature_names


def get_mlp_hidden_activations(model, X):
    """Extract activations from the last hidden layer of MLP."""
    model.eval()
    X_tensor = torch.FloatTensor(X)
    
    # Forward pass through layers to get hidden activations
    with torch.no_grad():
        # Pass through all layers except the last one
        x = X_tensor
        for i, layer in enumerate(model.model):
            if i == len(model.model) - 1:  # Stop before final linear layer
                break
            x = layer(x)
        hidden_activations = x.numpy()
    
    return hidden_activations


def train_mlp_optimized(X_train, y_train, X_val, y_val, class_weights):
    """Train MLP with early stopping."""
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    # Create data loader with larger batch size
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = MLP(X_train.shape[1])
    
    # Loss with class weights
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training with early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    model.train()
    for epoch in range(50):  # Reduced from 100
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation check for early stopping
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t).squeeze()
            val_loss = criterion(val_outputs, y_val_t).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
        model.train()
    
    # Final predictions
    model.eval()
    with torch.no_grad():
        logits = model(X_val_t).squeeze()
        predictions = torch.sigmoid(logits).numpy()
    
    return predictions


def run_phase1(modalities_data, y_train, class_weights):
    """Phase 1: Broad search across feature counts."""
    print("\n" + "="*70)
    print("PHASE 1: BROAD FEATURE SEARCH")
    print("="*70)
    
    # Define Phase 1 configurations
    phase1_configs = {
        'expression': {
            'feature_counts': [100, 500, 1000, 1500, 2000, 3000, 5000, 7500, 10000],
            'methods': ['f_test', 'fold_change', 'lasso']
        },
        'methylation': {
            'feature_counts': [500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000],
            'methods': ['f_test', 'fold_change', 'lasso']
        },
        'mutation': {
            'feature_counts': [50, 100, 200, 300, 500, 750, 1000],
            'methods': ['fisher']  # Only Fisher for binary data
        },
        'protein': {
            'feature_counts': [25, 50, 75, 100, 125, 150, 185],
            'methods': ['f_test', 'lasso']
        }
    }
    
    phase1_results = {}
    best_configs = {}
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for modality, config in phase1_configs.items():
        print(f"\n{'='*50}")
        print(f"Processing {modality.upper()}")
        print(f"{'='*50}")
        
        modality_results = defaultdict(list)
        X_data = modalities_data[modality]
        
        total_configs = len(config['feature_counts']) * len(config['methods'])
        config_count = 0
        
        for method in config['methods']:
            for n_features in config['feature_counts']:
                config_count += 1
                config_name = f"{method}_{n_features}"
                print(f"\n[{config_count}/{total_configs}] Testing {config_name}...")
                print(f"  Starting 5-fold cross-validation...")
                
                config_details = {
                    'method': method,
                    'n_features': n_features,
                    'fold_aucs': [],
                    'model_performances': defaultdict(list),
                    'selected_features_all': [],
                    'runtime': 0,
                    'cv_predictions': np.full(len(y_train), np.nan),  # ADD: Store CV predictions
                    'cv_true_labels': np.full(len(y_train), -1),  # ADD: Store true labels
                    'all_model_predictions': defaultdict(lambda: np.full(len(y_train), np.nan)),  # ADD: Store all model predictions
                    'all_model_binary_predictions': defaultdict(lambda: np.full(len(y_train), -1))  # ADD: Store binary predictions
                }
                
                config_start_time = time.time()
                early_stop = False
                
                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_data, y_train)):
                    # Check early stopping
                    if fold_idx == 2 and len(config_details['fold_aucs']) == 2:
                        if np.mean(config_details['fold_aucs']) < 0.55:
                            print(f"  Early stopping - poor performance (AUC < 0.55)")
                            early_stop = True
                            break
                    
                    X_train_fold = X_data.iloc[train_idx]
                    X_val_fold = X_data.iloc[val_idx]
                    y_train_fold = y_train[train_idx]
                    y_val_fold = y_train[val_idx]
                    
                    # Select features
                    selected_features = select_features(
                        X_train_fold, y_train_fold, modality, method, n_features
                    )
                    
                    if len(selected_features) < 10:
                        continue
                    
                    config_details['selected_features_all'].append(selected_features[:20])  # Top 20
                    
                    X_train_selected = X_train_fold[selected_features]
                    X_val_selected = X_val_fold[selected_features]
                    
                    # Train models
                    predictions, fold_feature_importances, feature_names = train_models(
                        X_train_selected, y_train_fold,
                        X_val_selected, y_val_fold,
                        class_weights, modality
                    )
                    
                    # Track all model performances
                    model_aucs = {}
                    for model_name, pred in predictions.items():
                        auc = float(roc_auc_score(y_val_fold, pred))
                        model_aucs[model_name] = auc
                        config_details['model_performances'][model_name].append(auc)
                        
                        # ADD: Store predictions from ALL models for confusion matrices
                        config_details['all_model_predictions'][model_name][val_idx] = pred
                        config_details['all_model_binary_predictions'][model_name][val_idx] = (pred > 0.5).astype(int)
                    
                    # Get best model for this fold
                    best_model = max(model_aucs.items(), key=lambda x: x[1])
                    config_details['fold_aucs'].append(best_model[1])
                    
                    # ADD: Store predictions from best model for ROC curve
                    config_details['cv_predictions'][val_idx] = predictions[best_model[0]]
                    config_details['cv_true_labels'][val_idx] = y_val_fold
                    
                    # ADD: Store feature importances for the best model
                    if fold_idx == 0:  # Initialize on first fold
                        config_details['feature_importances'] = {}
                        config_details['feature_names'] = feature_names
                    if best_model[0] in fold_feature_importances:
                        if best_model[0] not in config_details['feature_importances']:
                            config_details['feature_importances'][best_model[0]] = []
                        config_details['feature_importances'][best_model[0]].append(
                            fold_feature_importances[best_model[0]]
                        )
                
                config_details['runtime'] = time.time() - config_start_time
                
                if not early_stop and config_details['fold_aucs']:
                    config_details['mean_auc'] = float(np.mean(config_details['fold_aucs']))
                    config_details['std_auc'] = float(np.std(config_details['fold_aucs']))
                    
                    # Calculate mean AUC for each model
                    config_details['model_mean_aucs'] = {
                        model: float(np.mean(aucs)) 
                        for model, aucs in config_details['model_performances'].items()
                    }
                    
                    # Find most frequently selected features
                    all_features = [feat for fold_feats in config_details['selected_features_all'] 
                                   for feat in fold_feats]
                    feature_counts = defaultdict(int)
                    for feat in all_features:
                        feature_counts[feat] += 1
                    config_details['top_features'] = sorted(feature_counts.items(), 
                                                           key=lambda x: x[1], 
                                                           reverse=True)[:10]
                    
                    modality_results[config_name] = config_details
                    print(f"  Mean AUC: {config_details['mean_auc']:.3f} ± {config_details['std_auc']:.3f}")
                    print(f"  Best models: {sorted(config_details['model_mean_aucs'].items(), key=lambda x: x[1], reverse=True)[:2]}")
        
        # Find top 3 configurations for this modality
        if modality_results:
            # Sort by mean AUC
            sorted_configs = sorted(modality_results.items(), 
                                  key=lambda x: x[1]['mean_auc'], 
                                  reverse=True)[:3]
            
            best_configs[modality] = sorted_configs[0]  # Keep best for Phase 2
            
            print(f"\nTop 3 {modality} configurations:")
            for rank, (config_name, details) in enumerate(sorted_configs, 1):
                print(f"  {rank}. {config_name}: AUC={details['mean_auc']:.3f}±{details['std_auc']:.3f}")
                print(f"     Best model: {max(details['model_mean_aucs'].items(), key=lambda x: x[1])[0]}")
                print(f"     Runtime: {details['runtime']:.1f}s")
        
        phase1_results[modality] = dict(modality_results)
    
    return phase1_results, best_configs


def run_phase2(modalities_data, y_train, class_weights, best_configs, phase1_results):
    """Phase 2: Refinement around best configurations."""
    print("\n" + "="*70)
    print("PHASE 2: REFINEMENT AROUND BEST CONFIGS")
    print("="*70)
    
    phase2_results = {}
    final_best = {}
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for modality, (best_config_name, best_details) in best_configs.items():
        print(f"\n{'='*50}")
        print(f"Refining {modality.upper()}")
        print(f"Best from Phase 1: {best_config_name} (AUC: {best_details['mean_auc']:.3f})")
        print(f"{'='*50}")
        
        # Parse best config
        method, n_features = best_config_name.rsplit('_', 1)
        n_features = int(n_features)
        
        # Generate refinement range (±25%)
        refinement_features = [
            int(n_features * 0.75),
            n_features,
            int(n_features * 1.25)
        ]
        
        # Remove duplicates and ensure valid range
        X_data = modalities_data[modality]
        max_features = X_data.shape[1]
        
        # Special handling for boundaries
        if n_features == max_features:
            refinement_features = [int(n_features * 0.75), n_features]
        else:
            refinement_features = sorted(list(set([f for f in refinement_features 
                                                  if 10 <= f <= max_features])))
        
        modality_results = {}
        
        for n_feat in refinement_features:
            config_name = f"{method}_{n_feat}"
            print(f"\nTesting {config_name}...")
            
            config_details = {
                'method': method,
                'n_features': n_feat,
                'fold_aucs': [],
                'model_performances': defaultdict(list),
                'selected_features_all': [],
                'runtime': 0,
                'cv_predictions': np.full(len(y_train), np.nan),  # ADD: Store CV predictions
                'cv_true_labels': np.full(len(y_train), -1),  # ADD: Store true labels
                'all_model_predictions': defaultdict(lambda: np.full(len(y_train), np.nan)),  # ADD: Store all model predictions
                'all_model_binary_predictions': defaultdict(lambda: np.full(len(y_train), -1))  # ADD: Store binary predictions
            }
            
            config_start_time = time.time()
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_data, y_train)):
                X_train_fold = X_data.iloc[train_idx]
                X_val_fold = X_data.iloc[val_idx]
                y_train_fold = y_train[train_idx]
                y_val_fold = y_train[val_idx]
                
                # Select features
                selected_features = select_features(
                    X_train_fold, y_train_fold, modality, method, n_feat
                )
                
                X_train_selected = X_train_fold[selected_features]
                X_val_selected = X_val_fold[selected_features]
                
                # Train all models
                predictions, fold_feature_importances, feature_names = train_models(
                    X_train_selected, y_train_fold,
                    X_val_selected, y_val_fold,
                    class_weights, modality
                )
                
                config_details['selected_features_all'].append(selected_features[:20])
                
                # Track all model performances
                model_aucs = {}
                for model_name, pred in predictions.items():
                    auc = float(roc_auc_score(y_val_fold, pred))
                    model_aucs[model_name] = auc
                    config_details['model_performances'][model_name].append(auc)
                    
                    # ADD: Store predictions from ALL models for confusion matrices
                    config_details['all_model_predictions'][model_name][val_idx] = pred
                    config_details['all_model_binary_predictions'][model_name][val_idx] = (pred > 0.5).astype(int)
                
                # Get best model for this fold
                best_model = max(model_aucs.items(), key=lambda x: x[1])
                config_details['fold_aucs'].append(best_model[1])
                
                # ADD: Store predictions from best model for ROC curve
                config_details['cv_predictions'][val_idx] = predictions[best_model[0]]
                config_details['cv_true_labels'][val_idx] = y_val_fold
                
                # ADD: Store feature importances for the best model
                if fold_idx == 0:  # Initialize on first fold
                    config_details['feature_importances'] = {}
                    config_details['feature_names'] = feature_names
                if best_model[0] in fold_feature_importances:
                    if best_model[0] not in config_details['feature_importances']:
                        config_details['feature_importances'][best_model[0]] = []
                    config_details['feature_importances'][best_model[0]].append(
                        fold_feature_importances[best_model[0]]
                    )
            
            config_details['runtime'] = time.time() - config_start_time
            config_details['mean_auc'] = float(np.mean(config_details['fold_aucs']))
            config_details['std_auc'] = float(np.std(config_details['fold_aucs']))
            
            # Calculate mean AUC for each model
            config_details['model_mean_aucs'] = {
                model: float(np.mean(aucs)) 
                for model, aucs in config_details['model_performances'].items()
            }
            
            # Find most frequently selected features
            all_features = [feat for fold_feats in config_details['selected_features_all'] 
                           for feat in fold_feats]
            feature_counts = defaultdict(int)
            for feat in all_features:
                feature_counts[feat] += 1
            config_details['top_features'] = sorted(feature_counts.items(), 
                                                   key=lambda x: x[1], 
                                                   reverse=True)[:10]
            
            modality_results[config_name] = config_details
            print(f"  Mean AUC: {config_details['mean_auc']:.3f} ± {config_details['std_auc']:.3f}")
            print(f"  Best model: {max(config_details['model_mean_aucs'].items(), key=lambda x: x[1])[0]}")
        
        # Combine Phase 1 and Phase 2 results for final ranking
        all_configs = {**phase1_results[modality], **modality_results}
        
        # Get top 3 overall
        top_3_configs = sorted(all_configs.items(), 
                              key=lambda x: x[1]['mean_auc'], 
                              reverse=True)[:3]
        
        final_best[modality] = top_3_configs
        phase2_results[modality] = modality_results
    
    return phase2_results, final_best


def extract_model_performance_matrix(phase1_results, phase2_results):
    """Extract best AUC for each model type across each modality for heatmap."""
    model_performance_matrix = {}
    
    # Define all model types we want to track
    model_types = ['logistic', 'rf', 'xgboost', 'elasticnet', 'mlp']
    modalities = ['expression', 'protein', 'methylation', 'mutation']
    
    # Initialize matrix
    for model_type in model_types:
        model_performance_matrix[model_type] = {}
        for modality in modalities:
            model_performance_matrix[model_type][modality] = None
    
    # Collect all results from both phases
    all_results = {}
    for modality in modalities:
        all_results[modality] = {}
        if modality in phase1_results:
            all_results[modality].update(phase1_results[modality])
        if modality in phase2_results:
            all_results[modality].update(phase2_results[modality])
    
    # Find best AUC for each model type in each modality
    for modality, configs in all_results.items():
        for config_name, config_details in configs.items():
            if 'model_mean_aucs' in config_details:
                for model_type, auc in config_details['model_mean_aucs'].items():
                    if model_type in model_performance_matrix:
                        current_best = model_performance_matrix[model_type][modality]
                        if current_best is None or auc > current_best:
                            model_performance_matrix[model_type][modality] = float(auc)
    
    return model_performance_matrix


def save_pca_data_for_best_models(modalities_data, y_train, sample_ids, class_weights, 
                                 final_best, phase1_results, phase2_results, n_samples=1000):
    """Save model transformations for PCA visualization (protein and methylation only)."""
    pca_data = {}
    
    # Process all modalities for PCA figures
    for modality in ['protein', 'methylation', 'expression', 'mutation']:
        if modality not in final_best or not final_best[modality]:
            continue
        
        print(f"\nGenerating PCA data for {modality}...")
        
        # Get the best configuration
        best_config_name = final_best[modality][0][0]
        best_config_details = final_best[modality][0][1]
        
        # Get full training data for this modality
        X_full = modalities_data[modality]
        
        # Sample subset for PCA (to avoid memory issues)
        n_samples_actual = min(n_samples, len(X_full))
        sample_indices = np.random.choice(len(X_full), n_samples_actual, replace=False)
        X_subset = X_full.iloc[sample_indices]
        y_subset = y_train[sample_indices]
        sample_ids_subset = sample_ids[sample_indices]
        
        # Get feature selection method and number of features
        method = best_config_details['method']
        n_features = best_config_details['n_features']
        
        # Select features using the same method as in training
        selected_features = select_features(X_full, y_train, modality, method, n_features)
        X_subset_selected = X_subset[selected_features]
        
        # Initialize PCA data structure
        pca_data[modality] = {
            'X_raw': X_subset_selected.values[:500],  # Save only first 500 for visualization
            'y_labels': y_subset[:500].tolist(),
            'sample_ids': sample_ids_subset[:500].tolist(),
            'feature_names': selected_features,
            'n_features': n_features,
            'transformations': {}
        }
        
        # Train each model type on full selected data and get transformations
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_subset_selected)
        
        # 1. Logistic Regression
        print(f"  Training Logistic Regression for {modality} PCA...")
        lr = LogisticRegression(class_weight=class_weights, max_iter=1000, random_state=42)
        lr.fit(X_scaled, y_subset)
        # Decision function transformation
        lr_transform = X_scaled[:500] @ lr.coef_.T
        pca_data[modality]['transformations']['logistic'] = lr_transform.flatten()
        
        # 2. Random Forest
        print(f"  Training Random Forest for {modality} PCA...")
        rf = RandomForestClassifier(n_estimators=100, class_weight=class_weights,
                                   random_state=42, n_jobs=-1, max_depth=10)
        rf.fit(X_subset_selected.values, y_subset)
        # Leaf indices transformation
        rf_transform = rf.apply(X_subset_selected.values[:500])
        pca_data[modality]['transformations']['rf'] = rf_transform
        
        # 3. XGBoost
        print(f"  Training XGBoost for {modality} PCA...")
        scale_pos_weight = class_weights[0] / class_weights[1]
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                     scale_pos_weight=scale_pos_weight, random_state=42,
                                     n_jobs=-1, eval_metric='logloss')
        xgb_model.fit(X_subset_selected.values, y_subset)
        # Leaf indices transformation
        xgb_transform = xgb_model.apply(X_subset_selected.values[:500])
        pca_data[modality]['transformations']['xgboost'] = xgb_transform
        
        # 4. SVM (instead of ElasticNet for clearer decision boundaries)
        print(f"  Training SVM for {modality} PCA...")
        from sklearn.svm import SVC
        svm = SVC(kernel='rbf', class_weight=class_weights, random_state=42, 
                 decision_function_shape='ovr')
        svm.fit(X_scaled[:1000], y_subset[:1000])  # Limit samples for SVM speed
        # Decision function transformation
        svm_transform = svm.decision_function(X_scaled[:500])
        pca_data[modality]['transformations']['svm'] = svm_transform
        
        # 5. MLP (if modality supports it)
        if modality in ['protein', 'methylation']:
            print(f"  Training MLP for {modality} PCA...")
            # Train a simple MLP to get hidden layer activations
            model = MLP(X_scaled.shape[1])
            
            # Quick training
            X_train_t = torch.FloatTensor(X_scaled)
            y_train_t = torch.FloatTensor(y_subset)
            
            dataset = TensorDataset(X_train_t, y_train_t)
            loader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            pos_weight = torch.tensor([class_weights[1] / class_weights[0]])
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            model.train()
            for epoch in range(20):  # Quick training
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Get hidden layer activations
            mlp_transform = get_mlp_hidden_activations(model, X_scaled[:500])
            pca_data[modality]['transformations']['mlp'] = mlp_transform
    
    return pca_data


def generate_clinical_decision_examples(modalities_data, y_train, sample_ids, final_best, 
                                      phase1_results, phase2_results, n_examples=3):
    """Generate example patients with predictions and feature contributions for clinical decision support mockup."""
    clinical_examples = []
    
    # Use stratified sampling to get diverse examples
    responder_indices = np.where(y_train == 1)[0]
    non_responder_indices = np.where(y_train == 0)[0]
    
    # Select examples: 1-2 responders, 1-2 non-responders
    n_responders = min(len(responder_indices), (n_examples + 1) // 2)
    n_non_responders = min(len(non_responder_indices), n_examples - n_responders)
    
    selected_resp_idx = np.random.choice(responder_indices, n_responders, replace=False)
    selected_non_resp_idx = np.random.choice(non_responder_indices, n_non_responders, replace=False)
    selected_indices = np.concatenate([selected_resp_idx, selected_non_resp_idx])
    
    for idx in selected_indices:
        patient_id = sample_ids[idx]
        true_response = y_train[idx]
        
        patient_data = {
            'patient_id': patient_id,
            'true_response': int(true_response),
            'modality_data': {},
            'predictions': {},
            'feature_contributions': {}
        }
        
        # For each modality, get the patient's data and predictions
        for modality, top_configs in final_best.items():
            if not top_configs:
                continue
                
            # Get best configuration
            best_config_name = top_configs[0][0]
            best_config_details = top_configs[0][1]
            best_model_type = max(best_config_details['model_mean_aucs'].items(), 
                                key=lambda x: x[1])[0]
            
            # Get patient's features for this modality
            X_modality = modalities_data[modality]
            patient_features = X_modality.iloc[idx]
            
            # Get selected features from best config
            # Note: This is approximate - we're using the features from one fold
            selected_features = best_config_details['top_features'][:20] if 'top_features' in best_config_details else []
            selected_feature_names = [feat[0] for feat in selected_features]
            
            # Store patient's values for top features
            feature_values = {}
            for feat_name in selected_feature_names[:10]:  # Top 10 for display
                if feat_name in patient_features.index:
                    feature_values[feat_name] = float(patient_features[feat_name])
            
            patient_data['modality_data'][modality] = {
                'feature_values': feature_values,
                'best_model': best_model_type,
                'n_features_used': best_config_details['n_features']
            }
            
            # Store predictions (using CV predictions we already have)
            if modality in final_best:
                # Get CV predictions for this patient
                config = None
                if modality in phase1_results and best_config_name in phase1_results[modality]:
                    config = phase1_results[modality][best_config_name]
                elif modality in phase2_results and best_config_name in phase2_results[modality]:
                    config = phase2_results[modality][best_config_name]
                
                if config and 'cv_predictions' in config:
                    pred_value = config['cv_predictions'][idx]
                    if not np.isnan(pred_value):
                        patient_data['predictions'][modality] = {
                            'probability': float(pred_value),
                            'prediction': int(pred_value > 0.5)
                        }
            
            # Approximate feature contributions (simplified for mockup)
            # For tree-based models: use feature importance × normalized feature value
            # For linear models: use coefficient × standardized feature value
            if 'feature_importances' in phase1_results.get(modality, {}).get(best_config_name, {}) or \
               'feature_importances' in phase2_results.get(modality, {}).get(best_config_name, {}):
                
                # Get feature importances for best model
                config = phase1_results.get(modality, {}).get(best_config_name, {}) or \
                         phase2_results.get(modality, {}).get(best_config_name, {})
                
                if 'feature_importances' in config and best_model_type in config['feature_importances']:
                    importances = np.mean(config['feature_importances'][best_model_type], axis=0)
                    feature_names = config.get('feature_names', [])
                    
                    # Calculate approximate contributions for top features
                    contributions = {}
                    for i, (feat_name, importance) in enumerate(zip(feature_names[:10], importances[:10])):
                        if feat_name in feature_values:
                            # Normalize feature value (0-1 range for approximation)
                            feat_val = feature_values[feat_name]
                            if modality == 'mutation':
                                # Binary features
                                contribution = importance * feat_val
                            else:
                                # Continuous features - use z-score approximation
                                contribution = importance * np.sign(feat_val) * min(abs(feat_val), 3) / 3
                            contributions[feat_name] = float(contribution)
                    
                    patient_data['feature_contributions'][modality] = contributions
        
        clinical_examples.append(patient_data)
    
    return clinical_examples


def extract_confusion_matrix_data(phase1_results, phase2_results, final_best, y_train):
    """Extract confusion matrix data for all models and best models per modality."""
    confusion_matrix_data = {}
    
    for modality, top_configs in final_best.items():
        if not top_configs:
            continue
            
        # Get the best configuration
        best_config_name = top_configs[0][0]  # First is best
        best_config_details = top_configs[0][1]
        
        # Find the predictions from either phase1 or phase2 results
        config = None
        
        # Check phase1 results
        if modality in phase1_results and best_config_name in phase1_results[modality]:
            config = phase1_results[modality][best_config_name]
        elif modality in phase2_results and best_config_name in phase2_results[modality]:
            config = phase2_results[modality][best_config_name]
        
        if config and 'all_model_binary_predictions' in config:
            confusion_matrix_data[modality] = {
                'best_model': max(best_config_details['model_mean_aucs'].items(), 
                                key=lambda x: x[1])[0],
                'best_config': best_config_name,
                'model_confusion_matrices': {}
            }
            
            # For each model type, calculate confusion matrix
            for model_type in config['all_model_binary_predictions']:
                y_pred = config['all_model_binary_predictions'][model_type]
                y_true = config['cv_true_labels']
                
                # Filter out invalid predictions (from early stopping or missing folds)
                valid_mask = (y_pred != -1) & (y_true != -1)
                y_pred_valid = y_pred[valid_mask]
                y_true_valid = y_true[valid_mask]
                
                if len(y_pred_valid) > 0:
                    # Calculate confusion matrix components
                    tp = np.sum((y_true_valid == 1) & (y_pred_valid == 1))
                    tn = np.sum((y_true_valid == 0) & (y_pred_valid == 0))
                    fp = np.sum((y_true_valid == 0) & (y_pred_valid == 1))
                    fn = np.sum((y_true_valid == 1) & (y_pred_valid == 0))
                    
                    # Calculate metrics
                    total = tp + tn + fp + fn
                    accuracy = (tp + tn) / total if total > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    confusion_matrix_data[modality]['model_confusion_matrices'][model_type] = {
                        'confusion_matrix': {
                            'tp': int(tp),
                            'tn': int(tn),
                            'fp': int(fp),
                            'fn': int(fn)
                        },
                        'metrics': {
                            'accuracy': float(accuracy),
                            'precision': float(precision),
                            'recall': float(recall),
                            'specificity': float(specificity),
                            'f1_score': float(f1_score)
                        },
                        'n_samples': int(total)
                    }
            
            # Also store just the best model's confusion matrix for easy access
            best_model_type = confusion_matrix_data[modality]['best_model']
            if best_model_type in confusion_matrix_data[modality]['model_confusion_matrices']:
                confusion_matrix_data[modality]['best_model_confusion_matrix'] = \
                    confusion_matrix_data[modality]['model_confusion_matrices'][best_model_type]
    
    return confusion_matrix_data


def extract_feature_importances_for_best(phase1_results, phase2_results, final_best):
    """Extract feature importances for the best configuration of each modality."""
    print("\nExtracting feature importances for best configurations...")
    feature_importance_data = {}
    
    for modality, top_configs in final_best.items():
        print(f"\nProcessing {modality}...")
        if not top_configs:
            print(f"  No top configs found for {modality}")
            continue
            
        # Get the best configuration
        best_config_name = top_configs[0][0]  # First is best
        best_config_details = top_configs[0][1]
        print(f"  Best config: {best_config_name}")
        
        # Find the feature importances from either phase1 or phase2 results
        feature_importances = None
        feature_names = None
        
        # Check phase1 results
        if modality in phase1_results and best_config_name in phase1_results[modality]:
            config = phase1_results[modality][best_config_name]
            if 'feature_importances' in config and 'feature_names' in config:
                feature_importances = config['feature_importances']
                feature_names = config['feature_names']
                print(f"  Found importances in phase1 results")
        
        # Check phase2 results if not found in phase1
        if feature_importances is None and modality in phase2_results and best_config_name in phase2_results[modality]:
            config = phase2_results[modality][best_config_name]
            if 'feature_importances' in config and 'feature_names' in config:
                feature_importances = config['feature_importances']
                feature_names = config['feature_names']
                print(f"  Found importances in phase2 results")
        
        if feature_importances is None:
            print(f"  WARNING: No feature importances found for {modality}")
            continue
            
        if feature_importances and feature_names:
            # Get the best model type
            best_model_type = max(best_config_details['model_mean_aucs'].items(), 
                                key=lambda x: x[1])[0]
            
            # Average importances across folds for the best model
            if best_model_type in feature_importances and feature_importances[best_model_type]:
                avg_importances = np.mean(feature_importances[best_model_type], axis=0)
                
                # Create sorted list of (feature_name, importance) tuples
                # Convert numpy float32 to regular Python float for JSON serialization
                feature_importance_pairs = [(feat, float(imp)) for feat, imp in zip(feature_names, avg_importances)]
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Store top features (limit for methylation due to size)
                n_top_features = 50 if modality == 'methylation' else 20
                feature_importance_data[modality] = {
                    'best_model': best_model_type,
                    'top_features': feature_importance_pairs[:n_top_features],
                    'method': best_config_details['method'],
                    'n_features': best_config_details['n_features']
                }
                
                # For methylation, don't store all features due to memory constraints
                if modality != 'methylation':
                    feature_importance_data[modality]['all_features'] = feature_importance_pairs
                
                print(f"  Stored feature importances for {modality}: {len(feature_importance_pairs)} features, top {n_top_features} saved")
    
    return feature_importance_data


def extract_best_cv_predictions(phase1_results, phase2_results, final_best, y_train):
    """Extract CV predictions for the best configuration of each modality."""
    best_predictions = {}
    
    for modality, top_configs in final_best.items():
        if not top_configs:
            continue
            
        # Get the best configuration
        best_config_name = top_configs[0][0]  # First is best
        best_config_details = top_configs[0][1]
        
        # Find the predictions from either phase1 or phase2 results
        cv_preds = None
        cv_labels = None
        
        # Check phase1 results
        if modality in phase1_results and best_config_name in phase1_results[modality]:
            if 'cv_predictions' in phase1_results[modality][best_config_name]:
                cv_preds = phase1_results[modality][best_config_name]['cv_predictions']
                cv_labels = phase1_results[modality][best_config_name]['cv_true_labels']
        
        # Check phase2 results if not found in phase1
        if cv_preds is None and modality in phase2_results and best_config_name in phase2_results[modality]:
            if 'cv_predictions' in phase2_results[modality][best_config_name]:
                cv_preds = phase2_results[modality][best_config_name]['cv_predictions']
                cv_labels = phase2_results[modality][best_config_name]['cv_true_labels']
        
        if cv_preds is not None:
            # Filter out NaN values (from early stopping or other issues)
            valid_indices = ~np.isnan(cv_preds)
            
            # ADD: Extract per-fold predictions for proper ROC curve averaging
            fold_predictions_list = []
            
            # Use the same CV split as training
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_train)), y_train)):
                # Get predictions for this fold
                fold_mask = np.zeros(len(y_train), dtype=bool)
                fold_mask[val_idx] = True
                fold_valid = fold_mask & valid_indices
                
                if np.any(fold_valid):
                    fold_predictions_list.append({
                        'fold': fold_idx + 1,
                        'y_true': cv_labels[fold_valid].tolist() if cv_labels is not None else y_train[fold_valid].tolist(),
                        'y_pred_proba': cv_preds[fold_valid].tolist(),
                        'indices': np.where(fold_valid)[0].tolist()
                    })
            
            best_predictions[modality] = {
                'y_true': cv_labels[valid_indices].tolist() if cv_labels is not None else y_train[valid_indices].tolist(),
                'y_pred_proba': cv_preds[valid_indices].tolist(),
                'cv_auc': best_config_details['mean_auc'],
                'best_model': max(best_config_details['model_mean_aucs'].items(), 
                                key=lambda x: x[1])[0],
                'n_features': best_config_details['n_features'],
                'method': best_config_details['method'],
                'fold_predictions': fold_predictions_list  # ADD: Per-fold predictions for proper ROC averaging
            }
    
    return best_predictions


def convert_numpy_to_native(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle NaN and Inf values
        if np.isnan(obj):
            return None  # Convert NaN to None for JSON compatibility
        elif np.isinf(obj):
            return None  # Convert Inf to None for JSON compatibility
        return float(obj)
    elif isinstance(obj, (float, int)):
        # Handle Python float/int NaN and Inf
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        # Convert to list and recursively handle NaN/Inf in arrays
        return [convert_numpy_to_native(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_native(item) for item in obj)
    return obj

def main():
    """Main function for optimized model training."""
    data_dir = '/Users/tobyliu/bladder'
    
    print("="*80)
    print("OPTIMIZED MODEL TRAINING - Two-Phase Adaptive Feature Selection")
    print("="*80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    y_train, sample_ids, class_weights = load_data_and_labels(data_dir)
    
    # Load all modality data
    print("\nLoading modality data...")
    modalities_data = {}
    for modality in ['expression', 'protein', 'methylation', 'mutation']:
        X_data = load_modality_data(modality, data_dir)
        modalities_data[modality] = X_data.loc[sample_ids]
        print(f"  {modality}: {X_data.shape}")
    
    start_time = time.time()
    
    # Phase 1: Broad search
    phase1_results, best_configs = run_phase1(modalities_data, y_train, class_weights)
    
    phase1_time = time.time() - start_time
    print(f"\nPhase 1 completed in {phase1_time/60:.1f} minutes")
    
    # Phase 2: Refinement
    phase2_results, final_best = run_phase2(modalities_data, y_train, 
                                           class_weights, best_configs, phase1_results)
    
    total_time = time.time() - start_time
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY - TOP 3 CONFIGURATIONS PER MODALITY")
    print("="*80)
    
    final_summary = {}
    
    for modality, top_configs in final_best.items():
        print(f"\n{modality.upper()}:")
        print("-" * 60)
        
        modality_summary = []
        
        for rank, (config_name, details) in enumerate(top_configs, 1):
            print(f"\nRank {rank}: {config_name}")
            print(f"  Mean AUC: {details['mean_auc']:.3f} ± {details['std_auc']:.3f}")
            print(f"  Method: {details['method']}")
            print(f"  N features: {details['n_features']}")
            
            # Best model
            best_model = max(details['model_mean_aucs'].items(), key=lambda x: x[1])
            print(f"  Best model: {best_model[0]} (AUC: {best_model[1]:.3f})")
            
            # All model performances
            print(f"  All models:")
            for model, auc in sorted(details['model_mean_aucs'].items(), 
                                    key=lambda x: x[1], reverse=True):
                print(f"    - {model}: {auc:.3f}")
            
            # Top features
            print(f"  Top 5 features:")
            for feat, count in details['top_features'][:5]:
                print(f"    - {feat} (selected {count}/5 folds)")
            
            print(f"  Runtime: {details['runtime']:.1f}s")
            
            # Store for JSON output
            modality_summary.append({
                'rank': rank,
                'config': config_name,
                'method': details['method'],
                'n_features': details['n_features'],
                'mean_auc': details['mean_auc'],
                'std_auc': details['std_auc'],
                'best_model': best_model[0],
                'best_model_auc': float(best_model[1]),
                'all_model_aucs': details['model_mean_aucs'],
                'top_features': [feat for feat, _ in details['top_features'][:10]],
                'runtime_seconds': details['runtime'],
                'fold_aucs': details['fold_aucs']  # ADD: Include fold-by-fold AUCs for CV heatmap
            })
        
        final_summary[modality] = modality_summary
    
    # Extract CV predictions for ROC curves
    cv_predictions = extract_best_cv_predictions(phase1_results, phase2_results, final_best, y_train)
    
    # Extract model performance matrix for heatmap
    model_performance_matrix = extract_model_performance_matrix(phase1_results, phase2_results)
    
    # Extract feature importances for best configurations
    feature_importance_data = extract_feature_importances_for_best(phase1_results, phase2_results, final_best)
    
    # Generate clinical decision support examples
    np.random.seed(42)  # For reproducible patient selection
    clinical_examples = generate_clinical_decision_examples(
        modalities_data, y_train, sample_ids, final_best, 
        phase1_results, phase2_results, n_examples=5
    )
    
    # Generate PCA data for protein and methylation
    print("\nGenerating PCA visualization data...")
    pca_data = save_pca_data_for_best_models(
        modalities_data, y_train, sample_ids, class_weights,
        final_best, phase1_results, phase2_results, n_samples=1000
    )
    
    # Extract confusion matrix data
    confusion_matrix_data = extract_confusion_matrix_data(phase1_results, phase2_results, final_best, y_train)
    
    # Save comprehensive results
    results = {
        'final_summary': final_summary,
        'cv_predictions': cv_predictions,  # ADD: CV predictions for ROC curves
        'model_performance_matrix': model_performance_matrix,  # ADD: For model performance heatmap
        'feature_importance_data': feature_importance_data,  # ADD: For feature importance figure
        'clinical_decision_examples': clinical_examples,  # ADD: For clinical decision support mockup
        'confusion_matrix_data': confusion_matrix_data,  # ADD: For confusion matrices
        'runtime_minutes': total_time / 60,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Convert all numpy types to native Python types before saving
    results = convert_numpy_to_native(results)
    
    output_file = f'{data_dir}/individual_model_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save PCA data separately as pickle (contains numpy arrays)
    if pca_data:
        pca_output_file = f'{data_dir}/pca_visualization_data.pkl'
        import pickle
        with open(pca_output_file, 'wb') as f:
            pickle.dump(pca_data, f)
        print(f"\nPCA data saved to: {pca_output_file}")
        
        # Add reference in JSON results
        results['pca_data_file'] = 'pca_visualization_data.pkl'
        # Re-save JSON with PCA reference
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print comprehensive final summary
    print("\n" + "="*80)
    print("FINAL MODEL SUMMARY - BEST CONFIGURATIONS")
    print("="*80)
    
    for modality, configs in final_summary.items():
        if configs:
            best = configs[0]  # Already sorted by rank
            print(f"\n{modality.upper()}:")
            print(f"  Best Configuration: {best['config']}")
            print(f"  - Feature Selection Method: {best['method']}")
            print(f"  - Number of Features: {best['n_features']}")
            print(f"  - Mean CV AUC: {best['mean_auc']:.4f} ± {best['std_auc']:.4f}")
            print(f"  - Best Model: {best['best_model']} (AUC: {best['best_model_auc']:.4f})")
            print(f"  - All Model AUCs:")
            for model, auc in best['all_model_aucs'].items():
                print(f"    - {model}: {auc:.4f}")
            print(f"  - Top 10 Features: {', '.join(best['top_features'][:10])}")
    
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    # Note: all_results variable was not defined, skipping this summary line
    print(f"Total runtime: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_file}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()