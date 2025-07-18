#!/usr/bin/env python3
"""
Step 4b: Preprocess methylation data separately (due to large size)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')


# remove CpGs that are missing in >20% of samples
# calculates variance of each CpG, removes bottom 90% variance CpGs (keeps top 10%)
# process 50k CpGs at a time to fit in memory
# 3 passes:
# 1. for each chunk find CpGs with acceptable missing values (<=20%)
# 2. calculate variances for those CpGs, keep top 10% variance CpGs
# 3. combine all chunks, impute missing values using KNN k=5, standardize using StandardScaler mean=0, std=1
# start with 495k CpGs, end with 39,575k CpGs after filtering
def preprocess_methylation_chunked(data_dir, train_samples, test_samples, missing_threshold=0.2, 
                                   variance_percentile=90):
    """
    Preprocess methylation data in chunks to handle large file.
    
    Parameters:
    - missing_threshold: Remove CpGs with >20% missing values
    - variance_percentile: Remove bottom 90% variance CpGs (keep top 10%)
    """
    print(f"Preprocessing Methylation Data")
    print(f"  missing_threshold={missing_threshold*100:.0f}%")
    print(f"  variance_percentile={variance_percentile}%")
    print("="*60)
    
    # First pass: identify CpGs with acceptable missing values
    print(f"Pass 1: Identifying CpGs with <={missing_threshold*100:.0f}% missing values...")
    chunk_size = 50000
    acceptable_cpgs = []
    total_cpgs = 0
    
    for i, chunk in enumerate(pd.read_csv(f'{data_dir}/BLCA_methylation', sep='\t', 
                                         index_col=0, chunksize=chunk_size)):
        if i % 5 == 0:
            print(f"  Processing chunk {i+1} ({i*chunk_size} CpGs processed)")
        
        total_cpgs += len(chunk)
        # Find CpGs with acceptable missing values
        missing_pct = chunk.isna().sum(axis=1) / len(chunk.columns)
        acceptable = missing_pct[missing_pct <= missing_threshold].index.tolist()
        acceptable_cpgs.extend(acceptable)
    
    print(f"\nTotal CpGs: {total_cpgs}")
    print(f"CpGs with <={missing_threshold*100:.0f}% missing: {len(acceptable_cpgs)} ({len(acceptable_cpgs)/total_cpgs*100:.1f}%)")
    
    # Second pass: calculate variances for filtering
    print("\nPass 2: Calculating variances...")
    all_variances = []
    cpg_variance_map = {}
    
    # Process in chunks again, but only keep acceptable CpGs
    for i, chunk in enumerate(pd.read_csv(f'{data_dir}/BLCA_methylation', sep='\t', 
                                         index_col=0, chunksize=chunk_size)):
        if i % 5 == 0:
            print(f"  Processing chunk {i+1}")
        
        # Filter to acceptable CpGs in this chunk
        chunk_acceptable = chunk.loc[chunk.index.intersection(acceptable_cpgs)]
        
        if len(chunk_acceptable) > 0:
            # Calculate variance using only training samples
            train_data = chunk_acceptable[train_samples]
            variances = train_data.var(axis=1)
            
            # Store variances for percentile calculation
            for cpg, var in variances.items():
                if not pd.isna(var):  # Skip NaN variances
                    all_variances.append(var)
                    cpg_variance_map[cpg] = var
    
    # Calculate variance threshold based on percentile
    variance_threshold = np.percentile(all_variances, variance_percentile)
    print(f"\nVariance threshold (top {100-variance_percentile}%): {variance_threshold:.4f}")
    
    # Keep only high variance CpGs
    keep_cpgs = [cpg for cpg, var in cpg_variance_map.items() if var > variance_threshold]
    print(f"Kept {len(keep_cpgs)} CpGs with variance > {variance_threshold:.4f}")
    
    # Third pass: load data
    print("\nPass 3: Loading data...")
    
    # Read only the CpGs we want to keep
    filtered_data = []
    for i, chunk in enumerate(pd.read_csv(f'{data_dir}/BLCA_methylation', sep='\t', 
                                         index_col=0, chunksize=chunk_size)):
        chunk_filtered = chunk.loc[chunk.index.intersection(keep_cpgs)]
        if len(chunk_filtered) > 0:
            filtered_data.append(chunk_filtered)
    
    # Combine all chunks
    data = pd.concat(filtered_data)
    print(f"Loaded {len(data)} CpGs after variance filtering")
    
    # Split train/test
    train_data = data[train_samples].T
    test_data = data[test_samples].T
    
    # Impute missing values using KNN (better for methylation)
    print("\nImputing missing values using KNN...")
    imputer = KNNImputer(n_neighbors=5)
    train_imputed = pd.DataFrame(
        imputer.fit_transform(train_data),
        index=train_data.index,
        columns=train_data.columns
    )
    test_imputed = pd.DataFrame(
        imputer.transform(test_data),
        index=test_data.index,
        columns=test_data.columns
    )
    
    # Use imputed data directly (no fold change filter)
    train_data = train_imputed
    test_data = test_imputed
    print(f"\nUsing {train_data.shape[1]} variance-filtered CpGs")
    
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
 
def main():
    data_dir = '/Users/tobyliu/bladder'
    
    # Load train/test samples
    train_samples = pd.read_csv(f'{data_dir}/train_samples_fixed.csv')['sampleID'].tolist()
    test_samples = pd.read_csv(f'{data_dir}/test_samples_fixed.csv')['sampleID'].tolist()
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print()
    
    # Process methylation
    train_meth, test_meth, meth_params = preprocess_methylation_chunked(
        data_dir, train_samples, test_samples, missing_threshold=0.2, 
        variance_percentile=90
    )
    
    # Save results
    print("\nSaving preprocessed methylation data...")
    import os
    output_dir = f'{data_dir}/preprocessed_data/methylation'
    os.makedirs(output_dir, exist_ok=True)
    
    train_meth.to_csv(f'{output_dir}/methylation_train_preprocessed.csv')
    test_meth.to_csv(f'{output_dir}/methylation_test_preprocessed.csv')
    
    with open(f'{output_dir}/methylation_preprocessing_params.pkl', 'wb') as f:
        pickle.dump(meth_params, f)
    
    print("Methylation preprocessing completed!")


if __name__ == "__main__":
    main()