#!/usr/bin/env python3
"""
Step 1: Load phenotype data and map primary therapy outcomes to binary values

Goal: Map treatment outcomes (PHENOTYPE) to binary values
- 1 (Responder): Complete Remission/Response, Partial Remission/Response  
- 0 (Non-responder): Progressive Disease, Stable Disease
- Exclude: Missing values, [Discrepancy]
- Save the results to a CSV file "treatment_outcomes.csv" with columns sampleID and treatment_response(0 or 1)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_and_map_treatment_outcomes(phenotype_file):
    """
    Load phenotype data and map treatment outcomes to binary values.
    
    Parameters:
    -----------
    phenotype_file : str
        Path to the phenotype TSV file
        
    Returns:
    --------
    outcome_df : DataFrame
        DataFrame with sampleID and treatment_response columns
    stats : dict
        Statistics about the mapping
    """
    
    print("Step 1: Loading phenotype data and mapping treatment outcomes")
    print("=" * 60)
    
    # Load the phenotype data
    print(f"\nLoading phenotype data from: {phenotype_file}")
    phenotype_df = pd.read_csv(phenotype_file, sep='\t')
    print(f"Loaded {len(phenotype_df)} samples with {len(phenotype_df.columns)} features")
    
    # Check if primary_therapy_outcome_success column exists
    if 'primary_therapy_outcome_success' not in phenotype_df.columns:
        raise ValueError("Column 'primary_therapy_outcome_success' not found in phenotype data!")
    
    # Get the treatment outcome column
    outcomes = phenotype_df[['sampleID', 'primary_therapy_outcome_success']].copy()
    
    # Show unique values and their counts
    print("\nUnique treatment outcome values:")
    value_counts = outcomes['primary_therapy_outcome_success'].value_counts(dropna=False)
    for value, count in value_counts.items():
        print(f"  '{value}': {count}")
    
    # Define mapping
    response_mapping = {
        'Complete Remission/Response': 1,  # Responder
        'Partial Remission/Response': 1,    # Responder
        'Progressive Disease': 0,           # Non-responder
        'Stable Disease': 0                 # Non-responder
    }
    
    # Apply mapping
    print("\nApplying mapping:")
    print("  1 (Responder): Complete/Partial Remission/Response")
    print("  0 (Non-responder): Progressive/Stable Disease")
    print("  Exclude: Missing values, [Discrepancy]")
    
    # Create new column with mapped values
    outcomes['treatment_response'] = outcomes['primary_therapy_outcome_success'].map(response_mapping)
    
    # Filter out samples with unmapped values (NaN after mapping)
    original_count = len(outcomes)
    outcomes_clean = outcomes.dropna(subset=['treatment_response']).copy()
    excluded_count = original_count - len(outcomes_clean)
    
    # Convert to int type
    outcomes_clean['treatment_response'] = outcomes_clean['treatment_response'].astype(int)
    
    # Calculate statistics
    responders = (outcomes_clean['treatment_response'] == 1).sum()
    non_responders = (outcomes_clean['treatment_response'] == 0).sum()
    total = len(outcomes_clean)
    
    stats = {
        'total_samples': original_count,
        'usable_samples': total,
        'excluded_samples': excluded_count,
        'responders': responders,
        'non_responders': non_responders,
        'responder_rate': responders / total * 100,
        'non_responder_rate': non_responders / total * 100
    }
    
    # Print summary statistics
    print(f"\nMapping Summary:")
    print(f"  Total samples in phenotype data: {stats['total_samples']}")
    print(f"  Excluded samples (missing/invalid): {stats['excluded_samples']}")
    print(f"  Usable samples: {stats['usable_samples']}")
    print(f"\nClass Distribution:")
    print(f"  Responders (1): {stats['responders']} ({stats['responder_rate']:.1f}%)")
    print(f"  Non-responders (0): {stats['non_responders']} ({stats['non_responder_rate']:.1f}%)")
    print(f"  Class imbalance ratio: {stats['responders']/stats['non_responders']:.2f}:1")
    
    # Return only the relevant columns
    outcome_df = outcomes_clean[['sampleID', 'treatment_response']]
    
    return outcome_df, stats


def save_treatment_outcomes(outcome_df, output_file='treatment_outcomes.csv'):
    """
    Save the treatment outcome mapping to a CSV file.
    
    Parameters:
    -----------
    outcome_df : DataFrame
        DataFrame with sampleID and treatment_response columns
    output_file : str
        Path to save the output CSV file
    """
    outcome_df.to_csv(output_file, index=False)
    print(f"\nSaved treatment outcome mapping to: {output_file}")
    print(f"File contains {len(outcome_df)} samples with binary treatment response values")


def main():
    """Main function to execute Step 1"""
    
    # File paths
    phenotype_file = '/Users/tobyliu/bladder/BLCA_phenotype'
    output_file = '/Users/tobyliu/bladder/treatment_outcomes.csv'
    
    try:
        # Load and map treatment outcomes
        outcome_df, stats = load_and_map_treatment_outcomes(phenotype_file)
        
        # Save the results
        save_treatment_outcomes(outcome_df, output_file)
        
        # Show a few examples
        print("\nFirst 10 samples:")
        print(outcome_df.head(10).to_string(index=False))
        
        print("\nStep 1 completed successfully!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    main()