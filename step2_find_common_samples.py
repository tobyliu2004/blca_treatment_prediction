#!/usr/bin/env python3
"""
Step 2: Find common samples across all datasets

Goal: Identify samples that have data in all 4 molecular datasets AND valid treatment outcomes
This ensures we can use all modalities for our multimodal model.
create "common_samples.txt" with common sample IDs
create "sample_presence_matrix.csv" showing presence of samples in each dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_sample_ids_from_dataset(file_path, dataset_name):
    """
    Load sample IDs from a dataset file.
    
    Parameters:
    -----------
    file_path : str
        Path to the dataset file
    dataset_name : str
        Name of the dataset for logging
        
    Returns:
    --------
    sample_ids : set
        Set of sample IDs from this dataset
    """
    print(f"\nLoading {dataset_name} samples...")
    
    # Read just the header row to get sample IDs
    with open(file_path, 'r') as f:
        header = f.readline().strip().split('\t')
    
    # Remove the first column (gene/feature names)
    sample_ids = set(header[1:])
    print(f"  Found {len(sample_ids)} samples in {dataset_name}")
    
    return sample_ids


def load_treatment_outcomes(file_path):
    """
    Load sample IDs from treatment outcomes file.
    
    Parameters:
    -----------
    file_path : str
        Path to treatment_outcomes.csv
        
    Returns:
    --------
    sample_ids : set
        Set of sample IDs with valid treatment outcomes
    """
    print(f"\nLoading treatment outcome samples...")
    df = pd.read_csv(file_path)
    sample_ids = set(df['sampleID'].values)
    print(f"  Found {len(sample_ids)} samples with valid treatment outcomes")
    
    return sample_ids


def find_common_samples(data_dir='/Users/tobyliu/bladder'):
    """
    Find samples common to all datasets.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing all dataset files
        
    Returns:
    --------
    common_samples : list
        Sorted list of common sample IDs
    dataset_samples : dict
        Dictionary mapping dataset names to their sample sets
    """
    
    print("Step 2: Finding common samples across all datasets")
    print("=" * 60)
    
    # Define dataset files
    datasets = {
        'expression': f'{data_dir}/BLCA_expression',
        'methylation': f'{data_dir}/BLCA_methylation',
        'mutation': f'{data_dir}/BLCA_mutation',
        'protein_expression': f'{data_dir}/BLCA_protein_expression'
    }
    
    # Load sample IDs from each molecular dataset
    dataset_samples = {}
    for name, file_path in datasets.items():
        dataset_samples[name] = load_sample_ids_from_dataset(file_path, name)
    
    # Load treatment outcome samples
    treatment_samples = load_treatment_outcomes(f'{data_dir}/treatment_outcomes.csv')
    dataset_samples['treatment_outcomes'] = treatment_samples
    
    # Find intersection of all datasets
    print("\nFinding common samples across all datasets...")
    common_samples = dataset_samples['expression'].copy()
    
    for name, samples in dataset_samples.items():
        common_samples = common_samples.intersection(samples)
        print(f"  After intersecting with {name}: {len(common_samples)} samples remain")
    
    # Convert to sorted list for consistency
    common_samples = sorted(list(common_samples))
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Total common samples: {len(common_samples)}")
    print(f"  Percentage of treatment outcome samples retained: {len(common_samples)/len(treatment_samples)*100:.1f}%")
    
    # Show which samples are lost from each dataset
    print("\nSamples lost from each dataset:")
    for name, samples in dataset_samples.items():
        lost = len(samples) - len(common_samples)
        pct_lost = lost / len(samples) * 100
        print(f"  {name}: {lost} samples lost ({pct_lost:.1f}%)")
    
    return common_samples, dataset_samples


def create_sample_presence_matrix(common_samples, dataset_samples, output_file):
    """
    Create a matrix showing which samples are present in each dataset.
    
    Parameters:
    -----------
    common_samples : list
        List of common sample IDs
    dataset_samples : dict
        Dictionary mapping dataset names to their sample sets
    output_file : str
        Path to save the presence matrix
    """
    print(f"\nCreating sample presence matrix...")
    
    # Get all unique samples across all datasets
    all_samples = set()
    for samples in dataset_samples.values():
        all_samples.update(samples)
    all_samples = sorted(list(all_samples))
    
    # Create presence matrix
    presence_data = {'sampleID': all_samples}
    for name, samples in dataset_samples.items():
        presence_data[name] = [1 if sample in samples else 0 for sample in all_samples]
    
    # Add a column for whether sample is in common set
    presence_data['is_common'] = [1 if sample in common_samples else 0 for sample in all_samples]
    
    # Create DataFrame and save
    presence_df = pd.DataFrame(presence_data)
    presence_df.to_csv(output_file, index=False)
    print(f"  Saved to: {output_file}")
    
    # Show some statistics
    print(f"  Total unique samples across all datasets: {len(all_samples)}")
    print(f"  Samples in all datasets (common): {len(common_samples)}")
    print(f"  Samples missing from at least one dataset: {len(all_samples) - len(common_samples)}")


def save_common_samples(common_samples, output_file):
    """
    Save common sample IDs to a text file.
    
    Parameters:
    -----------
    common_samples : list
        List of common sample IDs
    output_file : str
        Path to save the sample list
    """
    with open(output_file, 'w') as f:
        for sample in common_samples:
            f.write(f"{sample}\n")
    
    print(f"\nSaved {len(common_samples)} common sample IDs to: {output_file}")


def verify_treatment_outcomes(common_samples, data_dir):
    """
    Verify the distribution of treatment outcomes in common samples.
    
    Parameters:
    -----------
    common_samples : list
        List of common sample IDs
    data_dir : str
        Directory containing treatment_outcomes.csv
    """
    print("\nVerifying treatment outcome distribution in common samples...")
    
    # Load treatment outcomes
    outcomes_df = pd.read_csv(f'{data_dir}/treatment_outcomes.csv')
    
    # Filter to common samples
    common_outcomes = outcomes_df[outcomes_df['sampleID'].isin(common_samples)]
    
    # Calculate statistics
    responders = (common_outcomes['treatment_response'] == 1).sum()
    non_responders = (common_outcomes['treatment_response'] == 0).sum()
    total = len(common_outcomes)
    
    print(f"  Common samples with outcomes: {total}")
    print(f"  Responders (1): {responders} ({responders/total*100:.1f}%)")
    print(f"  Non-responders (0): {non_responders} ({non_responders/total*100:.1f}%)")
    print(f"  Class imbalance ratio: {responders/non_responders:.2f}:1")


def main():
    """Main function to execute Step 2"""
    
    # Set up paths
    data_dir = '/Users/tobyliu/bladder'
    common_samples_file = f'{data_dir}/common_samples.txt'
    presence_matrix_file = f'{data_dir}/sample_presence_matrix.csv'
    
    try:
        # Find common samples
        common_samples, dataset_samples = find_common_samples(data_dir)
        
        if len(common_samples) == 0:
            raise ValueError("No common samples found across all datasets!")
        
        # Save common samples list
        save_common_samples(common_samples, common_samples_file)
        
        # Create sample presence matrix
        create_sample_presence_matrix(common_samples, dataset_samples, presence_matrix_file)
        
        # Verify treatment outcome distribution
        verify_treatment_outcomes(common_samples, data_dir)
        
        # Show first few common samples
        print("\nFirst 10 common samples:")
        for i, sample in enumerate(common_samples[:10]):
            print(f"  {i+1}. {sample}")
        
        print("\nStep 2 completed successfully!")
        print(f"Common samples saved to: {common_samples_file}")
        print(f"Sample presence matrix saved to: {presence_matrix_file}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    main()