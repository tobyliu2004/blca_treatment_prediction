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
    
    # Scale data once
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 1. Logistic Regression
    lr = LogisticRegression(class_weight=class_weights, max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    predictions['logistic'] = lr.predict_proba(X_val_scaled)[:, 1]
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight=class_weights,
                               random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X_train, y_train)
    predictions['rf'] = rf.predict_proba(X_val)[:, 1]
    
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
        predictions['elastic'] = elastic.predict_proba(X_val_scaled)[:, 1]
    
    # 5. MLP
    predictions['mlp'] = train_mlp_optimized(X_train_scaled, y_train, X_val_scaled, 
                                           y_val, class_weights)
    
    return predictions


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
                
                config_details = {
                    'method': method,
                    'n_features': n_features,
                    'fold_aucs': [],
                    'model_performances': defaultdict(list),
                    'selected_features_all': [],
                    'runtime': 0
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
                    predictions = train_models(
                        X_train_selected, y_train_fold,
                        X_val_selected, y_val_fold,
                        class_weights, modality
                    )
                    
                    # Track all model performances
                    model_aucs = {}
                    for model_name, pred in predictions.items():
                        auc = roc_auc_score(y_val_fold, pred)
                        model_aucs[model_name] = auc
                        config_details['model_performances'][model_name].append(auc)
                    
                    # Get best model for this fold
                    best_model = max(model_aucs.items(), key=lambda x: x[1])
                    config_details['fold_aucs'].append(best_model[1])
                
                config_details['runtime'] = time.time() - config_start_time
                
                if not early_stop and config_details['fold_aucs']:
                    config_details['mean_auc'] = np.mean(config_details['fold_aucs'])
                    config_details['std_auc'] = np.std(config_details['fold_aucs'])
                    
                    # Calculate mean AUC for each model
                    config_details['model_mean_aucs'] = {
                        model: np.mean(aucs) 
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
                'runtime': 0
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
                predictions = train_models(
                    X_train_selected, y_train_fold,
                    X_val_selected, y_val_fold,
                    class_weights, modality
                )
                
                config_details['selected_features_all'].append(selected_features[:20])
                
                # Track all model performances
                model_aucs = {}
                for model_name, pred in predictions.items():
                    auc = roc_auc_score(y_val_fold, pred)
                    model_aucs[model_name] = auc
                    config_details['model_performances'][model_name].append(auc)
                
                # Get best model for this fold
                best_model = max(model_aucs.items(), key=lambda x: x[1])
                config_details['fold_aucs'].append(best_model[1])
            
            config_details['runtime'] = time.time() - config_start_time
            config_details['mean_auc'] = np.mean(config_details['fold_aucs'])
            config_details['std_auc'] = np.std(config_details['fold_aucs'])
            
            # Calculate mean AUC for each model
            config_details['model_mean_aucs'] = {
                model: np.mean(aucs) 
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
                'best_model_auc': best_model[1],
                'all_model_aucs': details['model_mean_aucs'],
                'top_features': [feat for feat, _ in details['top_features'][:10]],
                'runtime_seconds': details['runtime']
            })
        
        final_summary[modality] = modality_summary
    
    # Save comprehensive results
    results = {
        'final_summary': final_summary,
        'runtime_minutes': total_time / 60,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    output_file = f'{data_dir}/optimized_model_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTotal runtime: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_file}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()