# Bladder Cancer Treatment Response Prediction

A multimodal machine learning pipeline that predicts bladder cancer treatment response by integrating four genomic data types.

## 🎯 Project Overview

This project develops a predictive model for bladder cancer treatment response using:
- **Gene Expression** (RNA-seq)
- **DNA Methylation** (450k array) 
- **Somatic Mutations** (binary status)
- **Protein Expression** (RPPA)

**Best Performance**: 0.712 AUC using late fusion with smart grid search

## 📊 Dataset

- **Training samples**: 227 patients
  - 159 responders (70%)
  - 68 non-responders (30%)
- **Source**: TCGA BLCA (Bladder Urothelial Carcinoma)

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas scikit-learn xgboost
```

### Running the Best Model
```bash
# Run the focused fusion model (best performance)
python step7_fusion_FOCUSED.py
```

Note: This script takes ~6.8 hours to complete due to extensive grid search.

### Running the Faster Alternative
```bash
# Run the optimized fusion model (30 minutes, 0.687 AUC)
python 03_fusion_approaches/step7_fusion_optimized.py
```

## 📁 Project Structure

```
bladder/
├── 01_data_preparation/     # Data preprocessing scripts
├── 02_model_training/       # Individual modality models
├── 03_fusion_approaches/    # Multimodal fusion scripts
├── 04_analysis_viz/         # Analysis and visualization
├── data/                    # Raw TCGA data files (BLCA_*)
├── preprocessed_data/       # Processed features by modality
├── fusion_focused_results/  # Best model results
├── docs/                    # Detailed documentation
└── step7_fusion_FOCUSED.py  # Main script (best model)
```

## 🔬 Key Methods

### Feature Selection (per modality)
- **Expression/Methylation**: Fold change selection
- **Protein**: F-statistic (ANOVA)
- **Mutation**: Fisher's exact test + pathway aggregation

### Models Used
- XGBoost (tailored hyperparameters by dimensionality)
- Random Forest
- Logistic Regression (ElasticNet for high-dim)
- Multi-Layer Perceptron

### Fusion Strategy
Late fusion with performance-weighted averaging outperformed early fusion (0.712 vs 0.640 AUC)

## 📈 Results Summary

| Configuration | Expression | Methylation | Protein | Mutation | AUC |
|--------------|------------|-------------|---------|----------|-----|
| Best (dense_1500_5500) | 1500 | 5500 | - | - | **0.712** |
| High features | 3000 | 3000 | 185 | 300 | 0.687 |
| Standard | 2000 | 2000 | 150 | 200 | 0.667 |

## 🏆 Key Insights

1. **Late fusion superior**: Preserves modality-specific signals
2. **Sample size limitation**: 227 samples limits model complexity
3. **Proper validation critical**: Feature selection inside CV folds prevents overfitting
4. **Simple methods competitive**: Well-tuned classical ML matches complex approaches

## 📚 Documentation

- `docs/project_writeup.md` - Comprehensive technical details
- `docs/PIPELINE.md` - Step-by-step pipeline explanation
- `docs/RESULTS.md` - Detailed performance analysis

## 👥 Authors

ML Engineers Team - Bladder Cancer Prediction Project

## 📄 License

This project uses publicly available TCGA data. Please cite TCGA when using this work.