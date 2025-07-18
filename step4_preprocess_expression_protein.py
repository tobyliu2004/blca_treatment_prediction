#!/usr/bin/env python3
"""
Step 4a: Preprocess expression and protein datasets
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


def load_train_test_samples(data_dir='/Users/tobyliu/bladder'):
    """Load train and test sample IDs."""
    print("Loading train/test splits...")
    train_samples = pd.read_csv(f'{data_dir}/train_samples_fixed.csv')['sampleID'].tolist()
    test_samples = pd.read_csv(f'{data_dir}/test_samples_fixed.csv')['sampleID'].tolist()
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Test samples: {len(test_samples)}")
    return train_samples, test_samples


#remove genes that are 0 (not expressed) in >80% of samples
# calculates variance of each gene (how much each gene varies across samples), removes bottom 5% variance genes
# standardizes the data to have mean=0, std=1, ensures no genes dominate
# start with 20,653 end with 17,689
def preprocess_expression(data_dir, train_samples, test_samples, variance_percentile=10):
    """Preprocess expression data with unsupervised methods only."""
    print(f"\n{'='*60}")
    print(f"Preprocessing Expression Data")
    print(f"  variance_percentile={variance_percentile}%")
    print(f"{'='*60}")
    
    # Load data
    print("Loading expression data...")
    data = pd.read_csv(f'{data_dir}/BLCA_expression', sep='\t', index_col=0)
    print(f"  Original shape: {data.shape}")
    
    # Split train/test
    train_data = data[train_samples].T
    test_data = data[test_samples].T
    
    # Remove genes with >80% zeros (using training set) - LESS CONSERVATIVE
    print("\nRemoving genes with >80% zeros...")
    zero_pct = (train_data == 0).sum() / len(train_data)
    keep_genes = zero_pct[zero_pct <= 0.8].index
    train_data = train_data[keep_genes]
    test_data = test_data[keep_genes]
    print(f"  Kept {len(keep_genes)} genes (removed {len(data) - len(keep_genes)})")
    
    # Remove bottom variance_percentile% genes (using training set)
    print(f"\nRemoving bottom {variance_percentile}% variance genes...")
    variances = train_data.var()
    threshold = np.percentile(variances, variance_percentile)
    keep_genes = variances[variances > threshold].index
    train_data = train_data[keep_genes]
    test_data = test_data[keep_genes]
    print(f"  Kept {len(keep_genes)} genes (variance > {threshold:.3f})")
    
    # Standardize
    print("\nStandardizing...")
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_data),
        index=train_data.index,
        columns=train_data.columns
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_data),
        index=test_data.index,
        columns=test_data.columns
    )
    
    print(f"\nFinal shape: {train_scaled.shape[1]} features")
    
    return train_scaled, test_scaled, {
        'scaler': scaler, 
        'features': list(train_scaled.columns),
        'variance_percentile': variance_percentile
    }


# removes proteins missing in >25% of samples, imputes remaining missing values with median
# removes bottom 5% variance proteins, standardizes the data to have mean=0 and std=1
# start with 245 proteins, end with 196
def preprocess_protein(data_dir, train_samples, test_samples, missing_threshold=0.25, variance_percentile=5):
    """Preprocess protein expression data."""
    print(f"\n{'='*60}")
    print("Preprocessing Protein Expression Data")
    print(f"  missing_threshold={missing_threshold*100:.0f}%")
    print(f"  variance_percentile={variance_percentile}%")
    print(f"{'='*60}")
    
    # Load data
    print("Loading protein expression data...")
    data = pd.read_csv(f'{data_dir}/BLCA_protein_expression', sep='\t', index_col=0)
    print(f"  Original shape: {data.shape}")
    
    # Remove proteins with >25% missing values - LESS CONSERVATIVE
    print(f"\nRemoving proteins with >{missing_threshold*100:.0f}% missing values...")
    missing_pct = data.isna().sum(axis=1) / len(data.columns)
    keep_proteins = missing_pct[missing_pct <= missing_threshold].index
    data = data.loc[keep_proteins]
    print(f"  Kept {len(keep_proteins)} proteins (removed {245 - len(keep_proteins)})")
    
    # Impute remaining missing values with median
    print("\nImputing remaining missing values with median...")
    imputer = SimpleImputer(strategy='median')
    data_imputed = pd.DataFrame(
        imputer.fit_transform(data.T).T,
        index=data.index,
        columns=data.columns
    )
    
    # Split train/test
    train_data = data_imputed[train_samples].T
    test_data = data_imputed[test_samples].T
    
    # Remove low variance proteins (using training set)
    print(f"\nRemoving bottom {variance_percentile}% variance proteins...")
    variances = train_data.var()
    threshold = np.percentile(variances, variance_percentile)
    keep_proteins = variances[variances > threshold].index
    train_data = train_data[keep_proteins]
    test_data = test_data[keep_proteins]
    print(f"  Kept {len(keep_proteins)} proteins (variance > {threshold:.3f})")
    
    # Standardize
    print("\nStandardizing...")
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_data),
        index=train_data.index,
        columns=train_data.columns
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_data),
        index=test_data.index,
        columns=test_data.columns
    )
    
    print(f"\nFinal shape: {train_scaled.shape[1]} features")
    
    return train_scaled, test_scaled, {
        'scaler': scaler, 
        'features': list(train_scaled.columns),
        'imputer': imputer,
        'missing_threshold': missing_threshold,
        'variance_percentile': variance_percentile
    }


def save_preprocessed_data(train_data, test_data, params, modality, output_dir):
    """Save preprocessed data and parameters to organized folders."""
    # Create modality-specific folder
    modality_dir = f'{output_dir}/preprocessed_data/{modality}'
    import os
    os.makedirs(modality_dir, exist_ok=True)
    
    # Save train data
    train_file = f'{modality_dir}/{modality}_train_preprocessed.csv'
    train_data.to_csv(train_file)
    print(f"  Saved training data: {train_file}")
    
    # Save test data
    test_file = f'{modality_dir}/{modality}_test_preprocessed.csv'
    test_data.to_csv(test_file)
    print(f"  Saved test data: {test_file}")
    
    # Save parameters
    params_file = f'{modality_dir}/{modality}_preprocessing_params.pkl'
    with open(params_file, 'wb') as f:
        pickle.dump(params, f)
    print(f"  Saved parameters: {params_file}")


def main():
    """Main preprocessing function."""
    data_dir = '/Users/tobyliu/bladder'
    output_dir = data_dir
    
    # Load train/test splits
    train_samples, test_samples = load_train_test_samples(data_dir)
    
    # Create summary dictionary
    summary = {
        'train_samples': len(train_samples),
        'test_samples': len(test_samples),
        'features_per_modality': {}
    }
    
    try:
        # 1. Preprocess Expression
        print("\n" + "="*70)
        print("PROCESSING EXPRESSION DATA")
        print("="*70)
        
        train_expr, test_expr, expr_params = preprocess_expression(
            data_dir, train_samples, test_samples, variance_percentile=5
        )
        save_preprocessed_data(train_expr, test_expr, expr_params, 'expression', output_dir)
        summary['features_per_modality']['expression'] = train_expr.shape[1]
          
        # 3. Preprocess Protein
        print("\n" + "="*70)
        print("PROCESSING PROTEIN EXPRESSION DATA")
        print("="*70)
        
        train_prot, test_prot, prot_params = preprocess_protein(
            data_dir, train_samples, test_samples
        )
        save_preprocessed_data(train_prot, test_prot, prot_params, 'protein', output_dir)
        summary['features_per_modality']['protein'] = train_prot.shape[1]
        
        # Print summary
        print("\n" + "="*70)
        print("PREPROCESSING SUMMARY (Expression, Mutation, Protein)")
        print("="*70)
        print(f"Train samples: {summary['train_samples']}")
        print(f"Test samples: {summary['test_samples']}")
        print("\nFeatures per modality:")
        for modality, n_features in summary['features_per_modality'].items():
            print(f"  {modality}: {n_features} features")
        
        print("\nNote: Methylation data will be processed separately due to size")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    main()