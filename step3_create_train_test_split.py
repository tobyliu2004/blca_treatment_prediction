#!/usr/bin/env python3
"""
Step 3: Create 80/20 train-test split with stratification
Author: ML Engineers Team
Date: 2025-01-11

Goal: Split the common samples into training and test sets
- Maintain class balance using stratification
- 80% training, 20% test
- Save sample lists for both sets
- creates "train_samples_fixed.csv" and "test_samples_fixed.csv"
- Also saves "treatment_response_fixed.csv" with all common samples and their outcomes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_common_samples_with_outcomes(data_dir='/Users/tobyliu/bladder'):
    """
    Load common samples and their treatment outcomes.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data files
        
    Returns:
    --------
    samples_df : DataFrame
        DataFrame with sampleID and treatment_response columns
    """
    print("Loading common samples and treatment outcomes...")
    
    # Load common samples
    with open(f'{data_dir}/common_samples.txt', 'r') as f:
        common_samples = [line.strip() for line in f.readlines()]
    
    print(f"  Loaded {len(common_samples)} common samples")
    
    # Load treatment outcomes
    outcomes_df = pd.read_csv(f'{data_dir}/treatment_outcomes.csv')
    
    # Filter to only common samples
    samples_df = outcomes_df[outcomes_df['sampleID'].isin(common_samples)].copy()
    
    # Verify we have all common samples
    if len(samples_df) != len(common_samples):
        print(f"  WARNING: Expected {len(common_samples)} samples but found {len(samples_df)}")
        missing = set(common_samples) - set(samples_df['sampleID'])
        if missing:
            print(f"  Missing samples: {missing}")
    
    return samples_df


def create_stratified_split(samples_df, test_size=0.2, random_state=42):
    """
    Create stratified train-test split.
    
    Parameters:
    -----------
    samples_df : DataFrame
        DataFrame with sampleID and treatment_response columns
    test_size : float
        Proportion of samples for test set (default: 0.2)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    train_samples : list
        List of training sample IDs
    test_samples : list
        List of test sample IDs
    split_stats : dict
        Statistics about the split
    """
    print(f"\nCreating stratified {int((1-test_size)*100)}/{int(test_size*100)} train-test split...")
    print(f"  Random seed: {random_state}")
    
    # Get sample IDs and labels
    X = samples_df['sampleID'].values
    y = samples_df['treatment_response'].values
    
    # Create stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y,  # This ensures stratification
        random_state=random_state
    )
    
    # Calculate statistics
    train_responders = (y_train == 1).sum()
    train_non_responders = (y_train == 0).sum()
    test_responders = (y_test == 1).sum()
    test_non_responders = (y_test == 0).sum()
    
    split_stats = {
        'total_samples': len(samples_df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_responders': train_responders,
        'train_non_responders': train_non_responders,
        'test_responders': test_responders,
        'test_non_responders': test_non_responders,
        'train_responder_rate': train_responders / len(X_train) * 100,
        'test_responder_rate': test_responders / len(X_test) * 100
    }
    
    # Print statistics
    print(f"\nSplit Statistics:")
    print(f"  Total samples: {split_stats['total_samples']}")
    print(f"  Training samples: {split_stats['train_samples']} ({split_stats['train_samples']/split_stats['total_samples']*100:.1f}%)")
    print(f"  Test samples: {split_stats['test_samples']} ({split_stats['test_samples']/split_stats['total_samples']*100:.1f}%)")
    
    print(f"\nClass Distribution:")
    print(f"  Training set:")
    print(f"    Responders: {split_stats['train_responders']} ({split_stats['train_responder_rate']:.1f}%)")
    print(f"    Non-responders: {split_stats['train_non_responders']} ({100-split_stats['train_responder_rate']:.1f}%)")
    print(f"  Test set:")
    print(f"    Responders: {split_stats['test_responders']} ({split_stats['test_responder_rate']:.1f}%)")
    print(f"    Non-responders: {split_stats['test_non_responders']} ({100-split_stats['test_responder_rate']:.1f}%)")
    
    # Verify no overlap
    train_set = set(X_train)
    test_set = set(X_test)
    overlap = train_set.intersection(test_set)
    if overlap:
        print(f"\n  ERROR: Found {len(overlap)} overlapping samples!")
    else:
        print(f"\n  ✓ Verified: No overlap between train and test sets")
    
    return list(X_train), list(X_test), split_stats


def save_splits(train_samples, test_samples, samples_df, data_dir='/Users/tobyliu/bladder'):
    """
    Save the train-test split to files.
    
    Parameters:
    -----------
    train_samples : list
        List of training sample IDs
    test_samples : list
        List of test sample IDs
    samples_df : DataFrame
        DataFrame with all common samples and outcomes
    data_dir : str
        Directory to save files
    """
    print(f"\nSaving split files...")
    
    # Create train samples DataFrame
    train_df = pd.DataFrame({'sampleID': train_samples})
    train_file = f'{data_dir}/train_samples_fixed.csv'
    train_df.to_csv(train_file, index=False)
    print(f"  Saved {len(train_samples)} training samples to: {train_file}")
    
    # Create test samples DataFrame
    test_df = pd.DataFrame({'sampleID': test_samples})
    test_file = f'{data_dir}/test_samples_fixed.csv'
    test_df.to_csv(test_file, index=False)
    print(f"  Saved {len(test_samples)} test samples to: {test_file}")
    
    # Save all common samples with outcomes (for reference)
    response_file = f'{data_dir}/overlap_treatment_response_fixed.csv'
    samples_df.to_csv(response_file, index=False)
    print(f"  Saved {len(samples_df)} samples with outcomes to: {response_file}")


def verify_split_files(data_dir='/Users/tobyliu/bladder'):
    """
    Verify the created split files.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the split files
    """
    print(f"\nVerifying saved files...")
    
    # Load saved files
    train_df = pd.read_csv(f'{data_dir}/train_samples_fixed.csv')
    test_df = pd.read_csv(f'{data_dir}/test_samples_fixed.csv')
    response_df = pd.read_csv(f'{data_dir}/overlap_treatment_response_fixed.csv')
    
    print(f"  train_samples_fixed.csv: {len(train_df)} samples")
    print(f"  test_samples_fixed.csv: {len(test_df)} samples")
    print(f"  overlap_treatment_response_fixed.csv: {len(response_df)} samples")
    
    # Verify total matches
    if len(train_df) + len(test_df) == len(response_df):
        print(f"  ✓ Train + Test = Total samples")
    else:
        print(f"  ERROR: Train + Test != Total samples")
    
    # Show first few samples from each set
    print(f"\nFirst 5 training samples:")
    for i, sample in enumerate(train_df['sampleID'].head(5)):
        outcome = response_df[response_df['sampleID'] == sample]['treatment_response'].values[0]
        print(f"  {i+1}. {sample} (outcome: {outcome})")
    
    print(f"\nFirst 5 test samples:")
    for i, sample in enumerate(test_df['sampleID'].head(5)):
        outcome = response_df[response_df['sampleID'] == sample]['treatment_response'].values[0]
        print(f"  {i+1}. {sample} (outcome: {outcome})")


def main():
    """Main function to execute Step 3"""
    
    # Set random seed for numpy as well (for complete reproducibility)
    np.random.seed(42)
    
    data_dir = '/Users/tobyliu/bladder'
    
    try:
        print("Step 3: Creating stratified train-test split")
        print("=" * 60)
        
        # Load common samples with outcomes
        samples_df = load_common_samples_with_outcomes(data_dir)
        
        # Create stratified split
        train_samples, test_samples, split_stats = create_stratified_split(
            samples_df, 
            test_size=0.2, 
            random_state=42
        )
        
        # Save splits
        save_splits(train_samples, test_samples, samples_df, data_dir)
        
        # Verify files
        verify_split_files(data_dir)
        
        print("\nStep 3 completed successfully!")
        print("Files created:")
        print("  - train_samples_fixed.csv")
        print("  - test_samples_fixed.csv")
        print("  - overlap_treatment_response_fixed.csv")
        print("  - (Using existing common_samples.txt from Step 2)")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    main()