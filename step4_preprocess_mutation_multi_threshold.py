#!/usr/bin/env python3
"""
Step 4: Preprocess mutation data with unsupervised filtering
Author: ML Engineers Team
Date: 2025-01-15

Goal: Create mutation features using unsupervised methods:
- Frequency thresholds (different for known cancer genes)
- Biological knowledge (bladder cancer genes)
- Pathway-level features
- Mutation burden calculation
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


# Define bladder cancer genes from literature
BLADDER_CANCER_GENES = [
    'TP53', 'RB1', 'FGFR3', 'PIK3CA', 'TSC1', 'ERBB2', 'ERCC2', 'EP300',
    'CREBBP', 'CDKN2A', 'CDKN1A', 'STAG2', 'KDM6A', 'ARID1A', 'KMT2D',
    'KMT2C', 'KMT2A', 'RBM10', 'ELF3', 'NFE2L2', 'TXNIP', 'RHOB', 'FOXA1',
    'PAIP1', 'BTG2', 'ZFP36L1', 'RHOA', 'FBXW7', 'SPTAN1', 'NRAS', 'KRAS',
    'HRAS', 'AKT1', 'PTEN', 'TSC2', 'MTOR', 'PPP1R3A', 'RXRA', 'ZBTB7C',
    'EZH2', 'KANSL1', 'ASXL2', 'MBD1', 'BPTF', 'CHD6', 'SRCAP', 'ATM',
    'ERBB3', 'GNA13', 'C3orf70', 'FAT1', 'FAM47C', 'PSIP1', 'MKL1', 'TRRAP',
    'MGA', 'IRAK4', 'DDX3X', 'RB1CC1', 'ANK3', 'RANBP2', 'PLXNB2'
]

# Define pathways
PATHWAYS = {
    'RTK_RAS': ['FGFR3', 'ERBB2', 'ERBB3', 'EGFR', 'NRAS', 'KRAS', 'HRAS', 'NF1'],
    'PI3K_AKT': ['PIK3CA', 'PTEN', 'AKT1', 'TSC1', 'TSC2', 'MTOR', 'PIK3R1'],
    'Cell_Cycle_TP53': ['TP53', 'RB1', 'CDKN2A', 'CDKN1A', 'MDM2', 'MDM4', 'CCND1'],
    'Chromatin_Remodeling': ['KDM6A', 'CREBBP', 'EP300', 'ARID1A', 'KMT2D', 'KMT2C', 'KMT2A'],
    'DNA_Repair': ['ERCC2', 'ATM', 'BRCA1', 'BRCA2', 'MLH1', 'MSH2'],
}

# general threshold = 0.03, for regular genes, must be mutated in >= 3% of samples
# for cancer genes, must be mutated in >= 1% of samples
# uses 61 genes from bladder cancer literature selected above
# pathways group together genes, different genes in same pathway often have similar effects, 
# instead of 8 individual features we have 1 summary feature
# ex.
#  Patient 1: KRAS=1, FGFR3=0, ERBB2=0... → RTK_RAS_pathway=1, Patient 2: KRAS=0, FGFR3=1, ERBB2=0... → RTK_RAS_pathway=1, Patient 3: KRAS=0, FGFR3=0, ERBB2=0... → RTK_RAS_pathway=0
# 
def preprocess_mutation_hybrid(data, train_samples, test_samples, general_threshold=0.03, cancer_gene_threshold=0.01):
    """
    Preprocess mutation data with unsupervised frequency-based approach.
    
    Parameters:
    - data: mutation matrix (genes x samples)
    - train_samples: list of training sample IDs
    - test_samples: list of test sample IDs  
    - general_threshold: minimum mutation frequency for general genes (3%)
    - cancer_gene_threshold: minimum mutation frequency for known cancer genes (1%)
    
    Returns:
    - train_data, test_data, params dictionary
    """
    # Split train/test
    train_data = data[train_samples].T
    test_data = data[test_samples].T
    
    # Calculate mutation frequency using ONLY training data
    mutation_freq = train_data.sum() / len(train_data)
    
    # Apply different thresholds based on biological knowledge
    keep_genes = []
    for gene in mutation_freq.index:
        if gene in BLADDER_CANCER_GENES:
            # Lower threshold for known cancer genes
            if mutation_freq[gene] >= cancer_gene_threshold:
                keep_genes.append(gene)
        else:
            # Standard threshold for other genes
            if mutation_freq[gene] >= general_threshold:
                keep_genes.append(gene)
    
    print(f"  After frequency filtering: {len(keep_genes)} genes")
    
    # Use frequency-filtered genes
    final_genes = keep_genes
    train_filtered = train_data[final_genes].copy()
    test_filtered = test_data[final_genes].copy()
    
    # Add mutation burden features
    # 1. Selected genes burden (using only filtered genes)
    train_filtered['selected_mutation_burden'] = train_filtered[final_genes].sum(axis=1)
    test_filtered['selected_mutation_burden'] = test_filtered[final_genes].sum(axis=1)
    
    # 2. Total mutation burden (all genes)
    train_filtered['total_mutation_burden'] = train_data.sum(axis=1)
    test_filtered['total_mutation_burden'] = test_data.sum(axis=1)
    
    # 3. Add pathway-level features
    print("  Adding pathway features...")
    for pathway_name, pathway_genes in PATHWAYS.items():
        # Find genes in this pathway that we have
        pathway_genes_present = [g for g in pathway_genes if g in final_genes]
        if pathway_genes_present:
            # Create pathway feature: 1 if ANY gene in pathway is mutated
            train_filtered[f'{pathway_name}_pathway'] = train_filtered[pathway_genes_present].max(axis=1)
            test_filtered[f'{pathway_name}_pathway'] = test_filtered[pathway_genes_present].max(axis=1)
    
    # Calculate statistics
    n_genes = len(final_genes)
    n_total_features = train_filtered.shape[1]
    sparsity = (train_filtered == 0).sum().sum() / (train_filtered.shape[0] * train_filtered.shape[1])
    
    params = {
        'general_threshold': general_threshold,
        'cancer_gene_threshold': cancer_gene_threshold,
        'n_genes_selected': n_genes,
        'n_pathway_features': n_total_features - n_genes - 2,  # subtract genes and 2 burden features
        'total_features': n_total_features,
        'sparsity': sparsity,
        'features': list(train_filtered.columns),
        'selected_genes': final_genes
    }
    
    return train_filtered, test_filtered, params


def analyze_mutation_distribution(train_data, y_train, top_n=10):
    """
    Analyze which mutations differ between responders and non-responders.
    """
    results = []
    
    for gene in train_data.columns:
        if gene == 'mutation_burden':
            continue
            
        # Calculate mutation rates in each group
        responders = train_data[y_train == 1][gene]
        non_responders = train_data[y_train == 0][gene]
        
        resp_rate = responders.mean()
        non_resp_rate = non_responders.mean()
        
        # Calculate difference and ratio
        diff = resp_rate - non_resp_rate
        ratio = resp_rate / (non_resp_rate + 0.0001)  # avoid division by zero
        
        results.append({
            'gene': gene,
            'responder_rate': resp_rate,
            'non_responder_rate': non_resp_rate,
            'difference': diff,
            'ratio': ratio
        })
    
    # Sort by absolute difference
    results_df = pd.DataFrame(results)
    results_df['abs_diff'] = results_df['difference'].abs()
    results_df = results_df.sort_values('abs_diff', ascending=False)
    
    return results_df.head(top_n)


def main():
    """Main function to preprocess mutations with hybrid approach."""
    data_dir = '/Users/tobyliu/bladder'
    output_dir = f'{data_dir}/preprocessed_data/mutation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load train/test splits
    print("Loading train/test splits...")
    train_samples = pd.read_csv(f'{data_dir}/train_samples_fixed.csv')['sampleID'].tolist()
    test_samples = pd.read_csv(f'{data_dir}/test_samples_fixed.csv')['sampleID'].tolist()
    
    # Load labels for analysis
    labels_df = pd.read_csv(f'{data_dir}/overlap_treatment_response_fixed.csv')
    train_labels = pd.DataFrame({'sampleID': train_samples}).merge(labels_df, on='sampleID')
    y_train = train_labels['treatment_response'].values
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    # Load mutation data
    print("\nLoading mutation data...")
    data = pd.read_csv(f'{data_dir}/BLCA_mutation', sep='\t', index_col=0)
    print(f"Original shape: {data.shape}")
    
    print("\n" + "="*70)
    print("UNSUPERVISED MUTATION PREPROCESSING")
    print("="*70)
    print(f"General gene threshold: 3%")
    print(f"Cancer gene threshold: 1%")
    
    # Preprocess with hybrid approach
    train_data, test_data, params = preprocess_mutation_hybrid(
        data, train_samples, test_samples, 
        general_threshold=0.03, 
        cancer_gene_threshold=0.01
    )
    
    # Print statistics
    print(f"\nFinal statistics:")
    print(f"  Genes selected: {params['n_genes_selected']}")
    print(f"  Pathway features: {params['n_pathway_features']}")
    print(f"  Total features: {params['total_features']}")
    print(f"  Sparsity: {params['sparsity']:.1%}")
    
    # Analyze top discriminative mutations
    print(f"\nTop 20 discriminative mutations:")
    top_mutations = analyze_mutation_distribution(train_data, y_train, top_n=20)
    print(top_mutations.to_string(index=False))
    
    # Save preprocessed data (single output)
    train_file = f'{output_dir}/mutation_train_preprocessed.csv'
    test_file = f'{output_dir}/mutation_test_preprocessed.csv'
    params_file = f'{output_dir}/mutation_preprocessing_params.pkl'
    
    train_data.to_csv(train_file)
    test_data.to_csv(test_file)
    
    # Save comprehensive params including analysis
    params['top_mutations'] = top_mutations.to_dict('records')
    params['bladder_cancer_genes_used'] = [g for g in params['selected_genes'] if g in BLADDER_CANCER_GENES]
    
    with open(params_file, 'wb') as f:
        pickle.dump(params, f)
    
    print(f"\nSaved files:")
    print(f"  {train_file}")
    print(f"  {test_file}")
    print(f"  {params_file}")
    
    # Save additional analysis
    analysis_file = f'{output_dir}/mutation_statistical_analysis.csv'
    
    # Create analysis dataframe
    analysis_data = []
    for gene in params['selected_genes']:
        is_cancer_gene = gene in BLADDER_CANCER_GENES
        
        analysis_data.append({
            'gene': gene,
            'is_bladder_cancer_gene': is_cancer_gene,
            'mutation_freq': train_data[gene].mean() if gene in train_data.columns else 0
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    analysis_df = analysis_df.sort_values('mutation_freq', ascending=False)
    analysis_df.to_csv(analysis_file, index=False)
    print(f"  {analysis_file}")
    
    print("\nPreprocessing complete!")
    print(f"\nUsing frequency-based approach found:")
    print(f"  {len([g for g in params['selected_genes'] if g in BLADDER_CANCER_GENES])} known bladder cancer genes")
    print(f"  {len([g for g in params['selected_genes'] if g not in BLADDER_CANCER_GENES])} other genes meeting threshold")
    print(f"  {params['n_pathway_features']} pathway features added")


if __name__ == "__main__":
    main()