(ml-env) tobyliu@OSXLAP11545 bladder % caffeinate -dims python 03_fusion_approaches/step7_fusion_advanced.py
================================================================================
ADVANCED MULTI-MODAL FUSION - TARGETING 0.75+ AUC
================================================================================

Started at: 2025-07-24 20:17:40
Loading data and labels...
  Loaded 227 training samples
  Class distribution: 159 responders, 68 non-responders

Loading modality data...
  expression: (227, 17689)
  protein: (227, 185)
  methylation: (227, 39575)
  mutation: (227, 1725)

======================================================================
ADVANCED FEATURE OPTIMIZATION
======================================================================

==================================================
STRATEGY: MINIMAL
==================================================

EXPRESSION (minimal):
  300 features: 0.659
  375 features: 0.650
  500 features: 0.653
  750 features: 0.653

METHYLATION (minimal):
  400 features: 0.696
  600 features: 0.670
  800 features: 0.674
  1000 features: 0.670

PROTEIN (minimal):
  75 features: 0.679
  90 features: 0.701
  100 features: 0.700
  110 features: 0.737

MUTATION (minimal):
  200 features: 0.585
  300 features: 0.596
  400 features: 0.622
  500 features: 0.605

==================================================
STRATEGY: DIVERSE
==================================================

EXPRESSION (diverse):
  2000 features: 0.639
  3000 features: 0.628
  4000 features: 0.638
  5000 features: 0.651
  6000 features: 0.659

METHYLATION (diverse):
  1000 features: 0.670
  1500 features: 0.667
  2000 features: 0.664
  2500 features: 0.659

PROTEIN (diverse):
  110 features: 0.737
  130 features: 0.689
  150 features: 0.690
  185 features: 0.675

MUTATION (diverse):
  600 features: 0.608
  800 features: 0.605
  1000 features: 0.613

==================================================
STRATEGY: MIXED_1
==================================================

EXPRESSION (mixed_1):
  300 features: 0.659
  375 features: 0.650
  500 features: 0.653

METHYLATION (mixed_1):
  400 features: 0.696
  600 features: 0.670
  800 features: 0.674

PROTEIN (mixed_1):
  130 features: 0.689
  150 features: 0.690
  185 features: 0.675

MUTATION (mixed_1):
  800 features: 0.605
  1000 features: 0.613
  1250 features: 0.599

==================================================
STRATEGY: MIXED_2
==================================================

EXPRESSION (mixed_2):
  300 features: 0.659
  500 features: 0.653
  750 features: 0.653

METHYLATION (mixed_2):
  1500 features: 0.667
  2000 features: 0.664
  2500 features: 0.659

PROTEIN (mixed_2):
  110 features: 0.737
  150 features: 0.690
  185 features: 0.675

MUTATION (mixed_2):
  800 features: 0.605
  1000 features: 0.613

Best configurations for minimal strategy:
  expression: 300 features (AUC: 0.659)
  methylation: 400 features (AUC: 0.696)
  protein: 110 features (AUC: 0.737)
  mutation: 400 features (AUC: 0.622)

Best configurations for diverse strategy:
  expression: 6000 features (AUC: 0.659)
  methylation: 1000 features (AUC: 0.670)
  protein: 110 features (AUC: 0.737)
  mutation: 1000 features (AUC: 0.613)

Best configurations for mixed_1 strategy:
  expression: 300 features (AUC: 0.659)
  methylation: 400 features (AUC: 0.696)
  protein: 150 features (AUC: 0.690)
  mutation: 1000 features (AUC: 0.613)

Best configurations for mixed_2 strategy:
  expression: 300 features (AUC: 0.659)
  methylation: 1500 features (AUC: 0.667)
  protein: 110 features (AUC: 0.737)
  mutation: 1000 features (AUC: 0.613)

======================================================================
TESTING ADVANCED FUSION METHODS WITH BOTH STRATEGIES
======================================================================

==================================================
Testing MINIMAL Strategy
==================================================

Using configurations for minimal:
  expression: 300 features
  methylation: 400 features
  protein: 110 features
  mutation: 400 features

Fold 1/5:
  expression: rf (AUC: 0.699, 300 features)
  methylation: rf (AUC: 0.812, 400 features)
  protein: rf (AUC: 0.804, 110 features)
  mutation: rf (AUC: 0.562, 400 features)
  Ensemble fusion (rank): 0.871

Fold 2/5:
  expression: logistic (AUC: 0.674, 300 features)
  methylation: xgboost (AUC: 0.623, 400 features)
  protein: logistic (AUC: 0.737, 110 features)
  mutation: xgboost (AUC: 0.647, 400 features)
  Ensemble fusion (rank): 0.743

Fold 3/5:
  expression: extra_trees (AUC: 0.623, 300 features)
  methylation: rf (AUC: 0.675, 400 features)
  protein: extra_trees (AUC: 0.731, 110 features)
  mutation: xgboost (AUC: 0.584, 400 features)
  Ensemble fusion (rank): 0.721

Fold 4/5:
  expression: xgboost (AUC: 0.700, 300 features)
  methylation: logistic (AUC: 0.769, 400 features)
  protein: xgboost (AUC: 0.728, 110 features)
  mutation: logistic (AUC: 0.620, 400 features)
  Ensemble fusion (rank): 0.762

Fold 5/5:
  expression: logistic (AUC: 0.599, 300 features)
  methylation: logistic (AUC: 0.599, 400 features)
  protein: xgboost (AUC: 0.687, 110 features)
  mutation: xgboost (AUC: 0.696, 400 features)
  Ensemble fusion (rank): 0.733

MINIMAL Strategy - FUSION METHOD SUMMARY:
ensemble: 0.766 ± 0.054
weighted: 0.732 ± 0.058
rank: 0.766 ± 0.054
geometric: 0.719 ± 0.059
stacking: 0.710 ± 0.058

==================================================
Testing DIVERSE Strategy
==================================================

Using configurations for diverse:
  expression: 6000 features
  methylation: 1000 features
  protein: 110 features
  mutation: 1000 features

Fold 1/5:
  expression: rf (AUC: 0.743, 6000 features)
  methylation: rf (AUC: 0.801, 1000 features)
  protein: rf (AUC: 0.804, 110 features)
  mutation: extra_trees (AUC: 0.558, 1000 features)
  Ensemble fusion (rank): 0.864

Fold 2/5:
  expression: xgboost (AUC: 0.578, 6000 features)
  methylation: extra_trees (AUC: 0.625, 1000 features)
  protein: logistic (AUC: 0.737, 110 features)
  mutation: xgboost (AUC: 0.594, 1000 features)
  Ensemble fusion (weighted_avg): 0.763

Fold 3/5:
  expression: rf (AUC: 0.716, 6000 features)
  methylation: logistic (AUC: 0.623, 1000 features)
  protein: extra_trees (AUC: 0.731, 110 features)
  mutation: rf (AUC: 0.601, 1000 features)
  Ensemble fusion (trimmed_mean): 0.745

Fold 4/5:
  expression: rf (AUC: 0.685, 6000 features)
  methylation: logistic (AUC: 0.762, 1000 features)
  protein: xgboost (AUC: 0.728, 110 features)
  mutation: extra_trees (AUC: 0.606, 1000 features)
  Ensemble fusion (rank): 0.776

Fold 5/5:
  expression: logistic (AUC: 0.574, 6000 features)
  methylation: logistic (AUC: 0.539, 1000 features)
  protein: xgboost (AUC: 0.687, 110 features)
  mutation: xgboost (AUC: 0.707, 1000 features)
  Ensemble fusion (trimmed_mean): 0.707

DIVERSE Strategy - FUSION METHOD SUMMARY:
ensemble: 0.771 ± 0.052
weighted: 0.747 ± 0.054
rank: 0.758 ± 0.060
geometric: 0.724 ± 0.049
stacking: 0.621 ± 0.095

==================================================
Testing MIXED_1 Strategy
==================================================

Using configurations for mixed_1:
  expression: 300 features
  methylation: 400 features
  protein: 150 features
  mutation: 1000 features

Fold 1/5:
  expression: rf (AUC: 0.699, 300 features)
  methylation: rf (AUC: 0.812, 400 features)
  protein: xgboost (AUC: 0.739, 150 features)
  mutation: extra_trees (AUC: 0.558, 1000 features)
  Ensemble fusion (rank): 0.837

Fold 2/5:
  expression: logistic (AUC: 0.674, 300 features)
  methylation: xgboost (AUC: 0.623, 400 features)
  protein: logistic (AUC: 0.647, 150 features)
  mutation: xgboost (AUC: 0.594, 1000 features)
  Ensemble fusion (rank): 0.708

Fold 3/5:
  expression: extra_trees (AUC: 0.623, 300 features)
  methylation: rf (AUC: 0.675, 400 features)
  protein: extra_trees (AUC: 0.656, 150 features)
  mutation: rf (AUC: 0.601, 1000 features)
  Ensemble fusion (weighted_avg): 0.685

Fold 4/5:
  expression: xgboost (AUC: 0.700, 300 features)
  methylation: logistic (AUC: 0.769, 400 features)
  protein: xgboost (AUC: 0.690, 150 features)
  mutation: extra_trees (AUC: 0.606, 1000 features)
  Ensemble fusion (rank): 0.781

Fold 5/5:
  expression: logistic (AUC: 0.599, 300 features)
  methylation: logistic (AUC: 0.599, 400 features)
  protein: xgboost (AUC: 0.717, 150 features)
  mutation: xgboost (AUC: 0.707, 1000 features)
  Ensemble fusion (rank): 0.758

MIXED_1 Strategy - FUSION METHOD SUMMARY:
ensemble: 0.754 ± 0.054
weighted: 0.716 ± 0.047
rank: 0.753 ± 0.055
geometric: 0.696 ± 0.048
stacking: 0.610 ± 0.073

==================================================
Testing MIXED_2 Strategy
==================================================

Using configurations for mixed_2:
  expression: 300 features
  methylation: 1500 features
  protein: 110 features
  mutation: 1000 features

Fold 1/5:
  expression: rf (AUC: 0.699, 300 features)
  methylation: rf (AUC: 0.821, 1500 features)
  protein: rf (AUC: 0.804, 110 features)
  mutation: extra_trees (AUC: 0.558, 1000 features)
  Ensemble fusion (rank): 0.850

Fold 2/5:
  expression: logistic (AUC: 0.674, 300 features)
  methylation: logistic (AUC: 0.569, 1500 features)
  protein: logistic (AUC: 0.737, 110 features)
  mutation: xgboost (AUC: 0.594, 1000 features)
  Ensemble fusion (weighted_avg): 0.719

Fold 3/5:
  expression: extra_trees (AUC: 0.623, 300 features)
  methylation: logistic (AUC: 0.603, 1500 features)
  protein: extra_trees (AUC: 0.731, 110 features)
  mutation: rf (AUC: 0.601, 1000 features)
  Ensemble fusion (rank): 0.697

Fold 4/5:
  expression: xgboost (AUC: 0.700, 300 features)
  methylation: logistic (AUC: 0.752, 1500 features)
  protein: xgboost (AUC: 0.728, 110 features)
  mutation: extra_trees (AUC: 0.606, 1000 features)
  Ensemble fusion (rank): 0.774

Fold 5/5:
  expression: logistic (AUC: 0.599, 300 features)
  methylation: logistic (AUC: 0.590, 1500 features)
  protein: xgboost (AUC: 0.687, 110 features)
  mutation: xgboost (AUC: 0.707, 1000 features)
  Ensemble fusion (trimmed_mean): 0.749

MIXED_2 Strategy - FUSION METHOD SUMMARY:
ensemble: 0.758 ± 0.053
weighted: 0.727 ± 0.052
rank: 0.751 ± 0.056
geometric: 0.694 ± 0.058
stacking: 0.588 ± 0.106

======================================================================
TOP 10 FUSION CONFIGURATIONS
======================================================================

Rank 1:
  Strategy: diverse
  Fusion Method: ensemble
  Mean AUC: 0.7712 ± 0.0518
  Feature Counts:
    expression: 6000
    methylation: 1000
    protein: 110
    mutation: 1000

Rank 2:
  Strategy: minimal
  Fusion Method: ensemble
  Mean AUC: 0.7659 ± 0.0540
  Feature Counts:
    expression: 300
    methylation: 400
    protein: 110
    mutation: 400

Rank 3:
  Strategy: minimal
  Fusion Method: rank
  Mean AUC: 0.7659 ± 0.0540
  Feature Counts:
    expression: 300
    methylation: 400
    protein: 110
    mutation: 400

Rank 4:
  Strategy: mixed_2
  Fusion Method: ensemble
  Mean AUC: 0.7578 ± 0.0532
  Feature Counts:
    expression: 300
    methylation: 1500
    protein: 110
    mutation: 1000

Rank 5:
  Strategy: diverse
  Fusion Method: rank
  Mean AUC: 0.7577 ± 0.0596
  Feature Counts:
    expression: 6000
    methylation: 1000
    protein: 110
    mutation: 1000

Rank 6:
  Strategy: mixed_1
  Fusion Method: ensemble
  Mean AUC: 0.7538 ± 0.0540
  Feature Counts:
    expression: 300
    methylation: 400
    protein: 150
    mutation: 1000

Rank 7:
  Strategy: mixed_1
  Fusion Method: rank
  Mean AUC: 0.7528 ± 0.0552
  Feature Counts:
    expression: 300
    methylation: 400
    protein: 150
    mutation: 1000

Rank 8:
  Strategy: mixed_2
  Fusion Method: rank
  Mean AUC: 0.7515 ± 0.0563
  Feature Counts:
    expression: 300
    methylation: 1500
    protein: 110
    mutation: 1000

Rank 9:
  Strategy: diverse
  Fusion Method: weighted
  Mean AUC: 0.7469 ± 0.0537
  Feature Counts:
    expression: 6000
    methylation: 1000
    protein: 110
    mutation: 1000
  Modality Weights:
    expression: 0.246
    methylation: 0.250
    protein: 0.275
    mutation: 0.229

Rank 10:
  Strategy: minimal
  Fusion Method: weighted
  Mean AUC: 0.7316 ± 0.0581
  Feature Counts:
    expression: 300
    methylation: 400
    protein: 110
    mutation: 400
  Modality Weights:
    expression: 0.243
    methylation: 0.256
    protein: 0.272
    mutation: 0.229

======================================================================
SAVING ROC DATA FOR TOP CONFIGURATIONS
======================================================================

Generating full predictions for best minimal configuration...
  Method: ensemble
  Mean AUC: 0.7659

Generating full predictions for best diverse configuration...
  Method: ensemble
  Mean AUC: 0.7712
No clinical examples found in individual results

======================================================================
BOOTSTRAP CONFIDENCE INTERVALS
======================================================================
  Bootstrap iteration 0/200...
  Bootstrap iteration 40/200...
  Bootstrap iteration 80/200...
  Bootstrap iteration 120/200...
  Bootstrap iteration 160/200...

Bootstrap AUC: 0.728 [95% CI: 0.538-0.905]
Based on 200 successful iterations

================================================================================
FINAL ADVANCED FUSION SUMMARY
================================================================================

BEST OVERALL CONFIGURATION:
  Strategy: diverse
  Fusion Method: ensemble
  Mean AUC: 0.7712 ± 0.0518
  Feature Counts:
    expression: 6000
    methylation: 1000
    protein: 110
    mutation: 1000

STRATEGY COMPARISON:
  Best minimal: ensemble (AUC: 0.766)
  Best diverse: ensemble (AUC: 0.771)
  Best mixed_1: ensemble (AUC: 0.754)
    Feature philosophy: Minimal: expression, methylation; Diverse: protein, mutation
  Best mixed_2: ensemble (AUC: 0.758)
    Feature philosophy: Minimal: expression, protein; Diverse: methylation, mutation

KEY IMPROVEMENTS:
  - Original fusion (5000 expr): 0.739
  - Aligned fusion (300 expr): 0.725
  - Advanced fusion: 0.771
  - Improvement over original: +0.032
  - Best individual modality: 0.737
  - Fusion improvement: +0.034

Bootstrap 95% CI: [0.538, 0.905]

Total runtime: 210.9 minutes
Results saved to: /Users/tobyliu/bladder/advanced_fusion_results

🎯 TARGET ACHIEVED! Fusion AUC ≥ 0.75

Completed at: 2025-07-24 23:48:39





