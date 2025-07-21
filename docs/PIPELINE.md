# Pipeline Overview

This document describes the complete pipeline from raw TCGA data to final predictions.

## Pipeline Stages

### Stage 1: Data Preparation (01_data_preparation/)

#### Step 1: Map Treatment Outcomes
**Script**: `step1_map_treatment_outcomes.py`
- Extracts treatment response from clinical data
- Maps patient IDs to binary outcomes (responder/non-responder)
- Output: `treatment_outcomes.csv`

#### Step 2: Find Common Samples
**Script**: `step2_find_common_samples.py`
- Identifies samples present across all 4 modalities
- Ensures consistent sample sets
- Output: `common_samples.txt`

#### Step 3: Create Train/Test Split
**Script**: `step3_create_train_test_split.py`
- Stratified 80/20 split maintaining class balance
- Fixed random seed for reproducibility
- Outputs: `train_samples_fixed.csv`, `test_samples_fixed.csv`

#### Step 4: Preprocess Each Modality

**Expression & Protein** (`step4_preprocess_expression_protein.py`):
- Expression: Remove zero-expression genes, variance filter (top 95%)
- Protein: Remove high-missing proteins (>25%), variance filter
- Both: Z-score normalization

**Methylation** (`step4_preprocess_methylation.py`):
- 3-pass chunked processing for memory efficiency
- Remove high-missing CpGs (>20%)
- Keep top 10% variance CpGs
- KNN imputation (k=5) for remaining missing values

**Mutation** (`step4_preprocess_mutation_multi_threshold.py`):
- Dual threshold: ≥3% general, ≥1% for known bladder genes
- Feature engineering: mutation burden, pathway aggregation
- Binary encoding (1=mutated, 0=wild-type)

### Stage 2: Model Training (02_model_training/)

#### Step 6: Individual Modality Models
**Script**: `step6_model_training_COMPREHENSIVE.py`
- Tests 8 different ML algorithms per modality
- Feature selection inside CV folds (no data leakage)
- Comprehensive hyperparameter tuning
- Outputs individual modality performances

### Stage 3: Multimodal Fusion (03_fusion_approaches/)

#### Step 7: Fusion Approaches

**FOCUSED Script** (`step7_fusion_FOCUSED.py`) - Best Performance:
- Smart grid search over feature counts
- Tests 4 fusion methods: simple average, optimized weights, rank-based, stacking
- Stable feature selection (multiple iterations)
- Early stopping for efficiency
- **Result**: 0.712 AUC

**Optimized Script** (`step7_fusion_optimized.py`) - Faster Alternative:
- Tests 6 pre-defined configurations
- Performance-weighted averaging
- 30-minute runtime
- **Result**: 0.687 AUC

### Stage 4: Analysis & Visualization (04_analysis_viz/)

- `compare_expression_vs_embedded.py`: Compares raw vs embedded features
- `create_poster_figures.py`: Generates publication-ready figures
- `verify_figures.py`: Validates results for accuracy

## Data Flow Diagram

```
Raw TCGA Data (BLCA_*)
       ↓
[Data Preparation]
   - Filter samples
   - Create splits
   - Preprocess
       ↓
Preprocessed Data/
   - expression/
   - methylation/
   - protein/
   - mutation/
       ↓
[Model Training]
   - Individual models
   - Cross-validation
       ↓
[Fusion]
   - Late fusion
   - Grid search
       ↓
Final Predictions
(0.712 AUC)
```

## Key Parameters

### Feature Counts (Best Configuration)
- Expression: 1,500 genes
- Methylation: 5,500 CpGs
- Protein: Variable (100-175)
- Mutation: Variable (200-300)

### Cross-Validation
- 5-fold stratified CV
- Feature selection inside folds
- Class weights for imbalance

### Hyperparameters
Tailored by modality dimensionality:
- High-dim (expr/meth): Shallow trees, high regularization
- Low-dim (prot/mut): Deeper trees, moderate regularization

## Running the Complete Pipeline

```bash
# 1. Data Preparation (run in order)
cd 01_data_preparation
python step1_map_treatment_outcomes.py
python step2_find_common_samples.py
python step3_create_train_test_split.py
python step4_preprocess_expression_protein.py
python step4_preprocess_methylation.py
python step4_preprocess_mutation_multi_threshold.py

# 2. Model Training (optional - for individual performances)
cd ../02_model_training
python step6_model_training_COMPREHENSIVE.py

# 3. Final Fusion Model
cd ..
python step7_fusion_FOCUSED.py  # Best performance (6.8 hours)
# OR
python 03_fusion_approaches/step7_fusion_optimized.py  # Faster (30 min)
```

## Computational Requirements

- Memory: 16GB RAM minimum (32GB recommended for methylation)
- Processing: Multi-core CPU beneficial for Random Forest/XGBoost
- Time: Full pipeline ~8 hours, quick version ~2 hours