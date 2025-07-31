# Bladder Cancer Treatment Response Prediction

This project implements a multi-modal machine learning approach to predict treatment response in bladder cancer patients using genomic data from The Cancer Genome Atlas (TCGA).

## Project Overview

This pipeline integrates four genomic modalities:
- Gene Expression (RNA-seq)
- DNA Methylation
- Protein Expression (RPPA)
- Somatic Mutations

The goal is to predict treatment response (responder vs non-responder) using various feature selection and fusion strategies.

## Requirements

- Python 3.8+
- Required packages are listed in each script
- TCGA data files (included in the project)

## How to Run the Complete Pipeline

Follow these steps in order to reproduce the analysis from start to finish:

### Step 1: Data Preparation (01_data_preparation/)

Run the scripts in this exact order:

1. **Split phenotype patients dataset into responder(1) and non-responder(0):**
   ```bash
   python 01_data_preparation/step1_map_treatment_outcomes.py
   ```

2. **Find common samples among all raw datasets:**
   ```bash
   python 01_data_preparation/step2_find_common_samples.py
   ```

3. **Create 80/20 train/test split datasets:**
   ```bash
   python 01_data_preparation/step3_create_train_test_split.py
   ```

4. **Preprocess all raw datasets (different scripts/methods for different datasets):**
   ```bash
   python 01_data_preparation/step4_preprocess_expression_protein.py
   python 01_data_preparation/step4_preprocess_methylation.py
   python 01_data_preparation/step4_preprocess_mutation_multi_threshold.py
   ```

### Step 2: Individual Model Training (02_model_training/)

After data preparation is complete, run the individual model training script (takes a 2-3 hours to run):

1. **Train individual modality models:**
   ```bash
   python 02_model_training/step6_individual_model_training.py
   ```

### Step 3: Fusion Model Training (03_fusion_approaches/)

Main script for our final results of fusion models (takes 2-3 hours to run):

```bash
python 03_fusion_approaches/step7_fusion_advanced.py
```

### Step 4: Visualization and Analysis (04_analysis_viz/)

Generate figures and visualizations:

1. **Compare our preprocessing vs scFoundation preprocessing of expression dataset:**
   ```bash
   python 04_analysis_viz/compare_expression_vs_embedded.py
   ```

2. **Create main analysis figures (not all figures generated are used in poster nor important):**
   ```bash
   python 04_analysis_viz/final_figures.py
   python 04_analysis_viz/create_phd_poster_figures_updated.py
   ```

## Expected Output

Each step will generate output files in their respective directories:

- `01_data_preparation/`: Processed data files for each modality (in preprocessed_data folder)
- `02_model_training/`: Individual model results (in individual_model_results.json)
- `03_expression_comparison/`: Fusion model results (in advanced_fusion_results/advanced_fusion_results.json)
- `04_analysis_viz/`: Publication-ready figures (all in 04_analysis_viz/final_figures_folder or 04_analysis_viz/phd_poster_figures_updated)

## Important Notes

1. **Keep the directory structure as is** - The scripts rely on relative paths
2. **Run scripts in order** - Each step depends on outputs from previous steps
3. **Check output messages** - Scripts will print progress and confirm successful completion
4. **Data files** - The TCGA data files should already be in the project directory

## Results Summary

The pipeline will produce:
- Individual modality performance metrics
- Feature-selected subsets for each modality
- Fusion model results comparing different strategies
- Visualizations comparing performance across approaches
- Literature comparison figures

## Troubleshooting

If you encounter errors:
1. Ensure all required packages are installed
2. Check that data files exist in the expected locations
3. Verify that previous steps completed successfully
4. Look for error messages in the console output