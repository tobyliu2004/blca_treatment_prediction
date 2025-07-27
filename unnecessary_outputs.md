(ml-env) tobyliu@OSXLAP11545 bladder % caffeinate -dims python /Users/tobyliu/bladder/02_model_training/step6_individual_model_training.py
================================================================================
OPTIMIZED MODEL TRAINING - Two-Phase Adaptive Feature Selection
================================================================================

Started at: 2025-07-23 23:50:50
Loading data and labels...
  Loaded 227 training samples
  Class distribution: 159 responders, 68 non-responders

Loading modality data...
  expression: (227, 17689)
  protein: (227, 185)
  methylation: (227, 39575)
  mutation: (227, 1725)

======================================================================
PHASE 1: BROAD FEATURE SEARCH
======================================================================

==================================================
Processing EXPRESSION
==================================================

[1/27] Testing f_test_100...
  Mean AUC: 0.661 ± 0.061
  Best models: [('rf', 0.6428072935129386), ('xgboost', 0.6373692839418645)]

[2/27] Testing f_test_500...
  Mean AUC: 0.664 ± 0.059
  Best models: [('xgboost', 0.6537287309464729), ('rf', 0.6258313762850053)]

[3/27] Testing f_test_1000...
  Mean AUC: 0.659 ± 0.055
  Best models: [('xgboost', 0.6475529510811768), ('rf', 0.6244954138603331)]

[4/27] Testing f_test_1500...
  Mean AUC: 0.643 ± 0.072
  Best models: [('xgboost', 0.6263647642679901), ('rf', 0.6071002082594825)]

[5/27] Testing f_test_2000...
  Mean AUC: 0.642 ± 0.088
  Best models: [('rf', 0.6260102800425381), ('xgboost', 0.6109989808578519)]

[6/27] Testing f_test_3000...
  Mean AUC: 0.639 ± 0.073
  Best models: [('xgboost', 0.6296249113789436), ('rf', 0.6226448954271535)]

[7/27] Testing f_test_5000...
  Mean AUC: 0.629 ± 0.077
  Best models: [('rf', 0.6145316377171216), ('xgboost', 0.6126861042183622)]

[8/27] Testing f_test_7500...
  Mean AUC: 0.638 ± 0.082
  Best models: [('rf', 0.6172174096065225), ('xgboost', 0.6147144186458703)]

[9/27] Testing f_test_10000...
  Mean AUC: 0.642 ± 0.072
  Best models: [('rf', 0.601445077100319), ('xgboost', 0.5982840747961716)]

[10/27] Testing fold_change_100...
  Mean AUC: 0.640 ± 0.064
  Best models: [('rf', 0.6188768388869196), ('mlp', 0.6143909517901454)]

[11/27] Testing fold_change_500...
  Mean AUC: 0.674 ± 0.048
  Best models: [('xgboost', 0.6582672367954625), ('rf', 0.6152594381425027)]

[12/27] Testing fold_change_1000...
  Mean AUC: 0.659 ± 0.078
  Best models: [('xgboost', 0.6381901364764268), ('rf', 0.6223823555476781)]

[13/27] Testing fold_change_1500...
  Mean AUC: 0.631 ± 0.069
  Best models: [('xgboost', 0.6234712867777384), ('rf', 0.6087934243176178)]

[14/27] Testing fold_change_2000...
  Mean AUC: 0.634 ± 0.080
  Best models: [('rf', 0.6074679856433889), ('xgboost', 0.6066842431761786)]

[15/27] Testing fold_change_3000...
  Mean AUC: 0.638 ± 0.088
  Best models: [('rf', 0.6184442573555476), ('mlp', 0.6035093938319745)]

[16/27] Testing fold_change_5000...
  Mean AUC: 0.629 ± 0.066
  Best models: [('xgboost', 0.6101349255583126), ('rf', 0.603947514179369)]

[17/27] Testing fold_change_7500...
  Mean AUC: 0.651 ± 0.057
  Best models: [('xgboost', 0.6270704094292804), ('rf', 0.6079963886919533)]

[18/27] Testing fold_change_10000...
  Mean AUC: 0.635 ± 0.070
  Best models: [('rf', 0.6252686325771003), ('xgboost', 0.6068448688408365)]

[19/27] Testing lasso_100...
  Mean AUC: 0.612 ± 0.080
  Best models: [('rf', 0.5931673165544133), ('mlp', 0.5867877082594826)]

[20/27] Testing lasso_500...
  Mean AUC: 0.636 ± 0.064
  Best models: [('rf', 0.6024060616802552), ('logistic', 0.5693825327897908)]

[21/27] Testing lasso_1000...
  Mean AUC: 0.623 ± 0.063
  Best models: [('rf', 0.6052402738390642), ('xgboost', 0.5795285359801489)]

[22/27] Testing lasso_1500...
  Mean AUC: 0.628 ± 0.063
  Best models: [('rf', 0.5988446029776675), ('xgboost', 0.5974798387096774)]

[23/27] Testing lasso_2000...
  Mean AUC: 0.649 ± 0.088
  Best models: [('rf', 0.6373642990074442), ('mlp', 0.5969126639489543)]

[24/27] Testing lasso_3000...
  Mean AUC: 0.628 ± 0.066
  Best models: [('mlp', 0.5977157922722439), ('rf', 0.5939998006026231)]

[25/27] Testing lasso_5000...
  Mean AUC: 0.621 ± 0.072
  Best models: [('xgboost', 0.6207328961361218), ('logistic', 0.5840426710386388)]

[26/27] Testing lasso_7500...
  Mean AUC: 0.642 ± 0.047
  Best models: [('rf', 0.6102728420772775), ('xgboost', 0.6056074973413683)]

[27/27] Testing lasso_10000...
  Mean AUC: 0.645 ± 0.034
  Best models: [('rf', 0.6073256380716059), ('xgboost', 0.5954116448068061)]

Top 3 expression configurations:
  1. fold_change_500: AUC=0.674±0.048
     Best model: xgboost
     Runtime: 29.8s
  2. f_test_500: AUC=0.664±0.059
     Best model: xgboost
     Runtime: 18.7s
  3. f_test_100: AUC=0.661±0.061
     Best model: rf
     Runtime: 5.0s

==================================================
Processing METHYLATION
==================================================

[1/27] Testing f_test_500...
  Mean AUC: 0.623 ± 0.077
  Best models: [('rf', 0.6026663860333215), ('mlp', 0.5977711804324708)]

[2/27] Testing f_test_1000...
  Mean AUC: 0.602 ± 0.081
  Best models: [('mlp', 0.5955933179723502), ('rf', 0.5660243486352358)]

[3/27] Testing f_test_2000...
  Mean AUC: 0.627 ± 0.069
  Best models: [('mlp', 0.6104639312300602), ('rf', 0.5850523971995747)]

[4/27] Testing f_test_3000...
  Mean AUC: 0.640 ± 0.077
  Best models: [('mlp', 0.5988102623183268), ('rf', 0.5979927330733782)]

[5/27] Testing f_test_5000...
  Mean AUC: 0.652 ± 0.057
  Best models: [('elastic', 0.6150500708968452), ('mlp', 0.5995247695852535)]

[6/27] Testing f_test_7500...
  Mean AUC: 0.639 ± 0.079
  Best models: [('rf', 0.5900916120170152), ('mlp', 0.5800203828429635)]

[7/27] Testing f_test_10000...
  Mean AUC: 0.632 ± 0.089
  Best models: [('mlp', 0.5951180875576036), ('rf', 0.5829066598723857)]

[8/27] Testing f_test_15000...
  Mean AUC: 0.612 ± 0.071
  Best models: [('mlp', 0.5859856876993974), ('logistic', 0.5813884704005672)]

[9/27] Testing f_test_20000...
  Mean AUC: 0.647 ± 0.083
  Best models: [('mlp', 0.6006768433179724), ('rf', 0.5930316155618576)]

[10/27] Testing fold_change_500...
  Mean AUC: 0.617 ± 0.077
  Best models: [('mlp', 0.6047567352002836), ('rf', 0.5663854794399148)]

[11/27] Testing fold_change_1000...
  Mean AUC: 0.619 ± 0.089
  Best models: [('rf', 0.5938042803970223), ('mlp', 0.5768477490251683)]

[12/27] Testing fold_change_2000...
  Mean AUC: 0.654 ± 0.088
  Best models: [('elastic', 0.6001063452676356), ('xgboost', 0.5966279688053882)]

[13/27] Testing fold_change_3000...
  Mean AUC: 0.650 ± 0.101
  Best models: [('mlp', 0.6011675824175824), ('rf', 0.5838227800425381)]

[14/27] Testing fold_change_5000...
  Mean AUC: 0.652 ± 0.064
  Best models: [('elastic', 0.6114066377171217), ('mlp', 0.6035315490960651)]

[15/27] Testing fold_change_7500...
  Mean AUC: 0.650 ± 0.099
  Best models: [('rf', 0.5957672367954625), ('xgboost', 0.5863700815313718)]

[16/27] Testing fold_change_10000...
  Mean AUC: 0.636 ± 0.083
  Best models: [('mlp', 0.5930953119461184), ('logistic', 0.5832860687699397)]

[17/27] Testing fold_change_15000...
  Mean AUC: 0.627 ± 0.083
  Best models: [('mlp', 0.6145981035093938), ('logistic', 0.5854462070187877)]

[18/27] Testing fold_change_20000...
  Mean AUC: 0.650 ± 0.089
  Best models: [('logistic', 0.598354971641262), ('rf', 0.5920783853243531)]

[19/27] Testing lasso_500...
  Mean AUC: 0.612 ± 0.075
  Best models: [('xgboost', 0.5862847837646225), ('mlp', 0.5773960918114145)]

[20/27] Testing lasso_1000...
  Mean AUC: 0.637 ± 0.089
  Best models: [('xgboost', 0.5848136742289969), ('rf', 0.584187234136831)]

[21/27] Testing lasso_2000...
  Mean AUC: 0.647 ± 0.054
  Best models: [('logistic', 0.6038550159517901), ('xgboost', 0.5896136121942573)]

[22/27] Testing lasso_3000...
  Mean AUC: 0.631 ± 0.074
  Best models: [('xgboost', 0.6184198865650479), ('logistic', 0.5903292272243885)]

[23/27] Testing lasso_5000...
  Mean AUC: 0.650 ± 0.069
  Best models: [('logistic', 0.6070132488479263), ('xgboost', 0.6056063895781637)]

[24/27] Testing lasso_7500...
  Mean AUC: 0.631 ± 0.069
  Best models: [('xgboost', 0.6039868397731302), ('logistic', 0.5961372297057781)]

[25/27] Testing lasso_10000...
  Mean AUC: 0.627 ± 0.058
  Best models: [('xgboost', 0.5915865384615384), ('logistic', 0.5879896313364055)]

[26/27] Testing lasso_15000...
  Mean AUC: 0.631 ± 0.068
  Best models: [('xgboost', 0.5998792538107054), ('logistic', 0.5946295639844026)]

[27/27] Testing lasso_20000...
  Mean AUC: 0.646 ± 0.068
  Best models: [('xgboost', 0.6104240517546969), ('mlp', 0.5937112282878412)]

Top 3 methylation configurations:
  1. fold_change_2000: AUC=0.654±0.088
     Best model: elastic
     Runtime: 132.1s
  2. fold_change_5000: AUC=0.652±0.064
     Best model: elastic
     Runtime: 230.5s
  3. f_test_5000: AUC=0.652±0.057
     Best model: elastic
     Runtime: 227.1s

==================================================
Processing MUTATION
==================================================

[1/7] Testing fisher_50...
  Mean AUC: 0.611 ± 0.026
  Best models: [('xgboost', 0.563623271889401), ('rf', 0.5318138514711095)]

[2/7] Testing fisher_100...
  Early stopping - poor performance (AUC < 0.55)

[3/7] Testing fisher_200...
  Mean AUC: 0.603 ± 0.053
  Best models: [('xgboost', 0.5908609535625665), ('rf', 0.5141749379652606)]

[4/7] Testing fisher_300...
  Mean AUC: 0.614 ± 0.038
  Best models: [('xgboost', 0.5963410581354129), ('rf', 0.541111308046792)]

[5/7] Testing fisher_500...
  Mean AUC: 0.608 ± 0.047
  Best models: [('xgboost', 0.578761963842609), ('rf', 0.5162076834455867)]

[6/7] Testing fisher_750...
  Mean AUC: 0.617 ± 0.058
  Best models: [('xgboost', 0.5953839507266926), ('mlp', 0.5559442573555478)]

[7/7] Testing fisher_1000...
  Mean AUC: 0.626 ± 0.067
  Best models: [('xgboost', 0.602150168380007), ('mlp', 0.5452399415101028)]

Top 3 mutation configurations:
  1. fisher_1000: AUC=0.626±0.067
     Best model: xgboost
     Runtime: 13.3s
  2. fisher_750: AUC=0.617±0.058
     Best model: xgboost
     Runtime: 11.4s
  3. fisher_300: AUC=0.614±0.038
     Best model: xgboost
     Runtime: 11.2s

==================================================
Processing PROTEIN
==================================================

[1/14] Testing f_test_25...
  Mean AUC: 0.629 ± 0.094
  Best models: [('xgboost', 0.5826468894009216), ('logistic', 0.5814914923785892)]

[2/14] Testing f_test_50...
  Mean AUC: 0.653 ± 0.047
  Best models: [('logistic', 0.616944345976604), ('rf', 0.5940053394186459)]

[3/14] Testing f_test_75...
  Mean AUC: 0.690 ± 0.040
  Best models: [('rf', 0.6176256203473945), ('xgboost', 0.6122884172279333)]

[4/14] Testing f_test_100...
  Mean AUC: 0.689 ± 0.039
  Best models: [('xgboost', 0.6211394452321871), ('logistic', 0.5991459145693017)]

[5/14] Testing f_test_125...
  Mean AUC: 0.702 ± 0.039
  Best models: [('xgboost', 0.6048520028358738), ('logistic', 0.5941853509393832)]

[6/14] Testing f_test_150...
  Mean AUC: 0.670 ± 0.045
  Best models: [('xgboost', 0.6223546614675646), ('rf', 0.5813951169797944)]

[7/14] Testing f_test_185...
  Mean AUC: 0.684 ± 0.067
  Best models: [('xgboost', 0.6433678216944345), ('rf', 0.608144275079759)]

[8/14] Testing lasso_25...
  Mean AUC: 0.660 ± 0.057
  Best models: [('rf', 0.6408858782346686), ('xgboost', 0.622955069124424)]

[9/14] Testing lasso_50...
  Mean AUC: 0.673 ± 0.070
  Best models: [('xgboost', 0.6398462424672102), ('rf', 0.6058290499822757)]

[10/14] Testing lasso_75...
  Mean AUC: 0.666 ± 0.057
  Best models: [('xgboost', 0.6371876107763205), ('rf', 0.6296620214462957)]

[11/14] Testing lasso_100...
  Mean AUC: 0.656 ± 0.083
  Best models: [('xgboost', 0.6214053084012762), ('rf', 0.6100230414746544)]

[12/14] Testing lasso_125...
  Mean AUC: 0.671 ± 0.046
  Best models: [('xgboost', 0.6266815845444877), ('rf', 0.599241182204892)]

[13/14] Testing lasso_150...
  Mean AUC: 0.673 ± 0.068
  Best models: [('xgboost', 0.6331265508684865), ('rf', 0.6002924494859979)]

[14/14] Testing lasso_185...
  Mean AUC: 0.683 ± 0.052
  Best models: [('xgboost', 0.6404200638071607), ('rf', 0.5795739542715349)]

Top 3 protein configurations:
  1. f_test_125: AUC=0.702±0.039
     Best model: xgboost
     Runtime: 12.0s
  2. f_test_75: AUC=0.690±0.040
     Best model: rf
     Runtime: 6.5s
  3. f_test_100: AUC=0.689±0.039
     Best model: xgboost
     Runtime: 8.9s

Phase 1 completed in 203.2 minutes

======================================================================
PHASE 2: REFINEMENT AROUND BEST CONFIGS
======================================================================

==================================================
Refining EXPRESSION
Best from Phase 1: fold_change_500 (AUC: 0.674)
==================================================

Testing fold_change_375...
  Mean AUC: 0.670 ± 0.058
  Best model: xgboost

Testing fold_change_500...
  Mean AUC: 0.672 ± 0.045
  Best model: xgboost

Testing fold_change_625...
  Mean AUC: 0.659 ± 0.063
  Best model: xgboost

==================================================
Refining METHYLATION
Best from Phase 1: fold_change_2000 (AUC: 0.654)
==================================================

Testing fold_change_1500...
  Mean AUC: 0.628 ± 0.089
  Best model: rf

Testing fold_change_2000...
  Mean AUC: 0.654 ± 0.088
  Best model: elastic

Testing fold_change_2500...
  Mean AUC: 0.633 ± 0.067
  Best model: mlp

==================================================
Refining MUTATION
Best from Phase 1: fisher_1000 (AUC: 0.626)
==================================================

Testing fisher_750...
  Mean AUC: 0.614 ± 0.059
  Best model: xgboost

Testing fisher_1000...
  Mean AUC: 0.625 ± 0.067
  Best model: xgboost

Testing fisher_1250...
  Mean AUC: 0.631 ± 0.042
  Best model: xgboost

==================================================
Refining PROTEIN
Best from Phase 1: f_test_125 (AUC: 0.702)
==================================================

Testing f_test_93...
  Mean AUC: 0.704 ± 0.048
  Best model: xgboost

Testing f_test_125...
  Mean AUC: 0.687 ± 0.037
  Best model: xgboost

Testing f_test_156...
  Mean AUC: 0.693 ± 0.034
  Best model: xgboost

================================================================================
FINAL RESULTS SUMMARY - TOP 3 CONFIGURATIONS PER MODALITY
================================================================================

EXPRESSION:
------------------------------------------------------------

Rank 1: fold_change_500
  Mean AUC: 0.672 ± 0.045
  Method: fold_change
  N features: 500
  Best model: xgboost (AUC: 0.658)
  All models:
    - xgboost: 0.658
    - rf: 0.615
    - mlp: 0.609
    - elastic: 0.565
    - logistic: 0.539
  Top 5 features:
    - ZFP64 (selected 5/5 folds)
    - AADACL2 (selected 3/5 folds)
    - ZNF69 (selected 3/5 folds)
    - TXNRD1 (selected 3/5 folds)
    - PLXNB1 (selected 3/5 folds)
  Runtime: 37.7s

Rank 2: fold_change_375
  Mean AUC: 0.670 ± 0.058
  Method: fold_change
  N features: 375
  Best model: xgboost (AUC: 0.642)
  All models:
    - xgboost: 0.642
    - rf: 0.625
    - mlp: 0.618
    - elastic: 0.563
    - logistic: 0.538
  Top 5 features:
    - ZFP64 (selected 5/5 folds)
    - AADACL2 (selected 3/5 folds)
    - ZNF69 (selected 3/5 folds)
    - TXNRD1 (selected 3/5 folds)
    - PLXNB1 (selected 3/5 folds)
  Runtime: 33.8s

Rank 3: f_test_500
  Mean AUC: 0.664 ± 0.059
  Method: f_test
  N features: 500
  Best model: xgboost (AUC: 0.654)
  All models:
    - xgboost: 0.654
    - rf: 0.626
    - mlp: 0.616
    - elastic: 0.562
    - logistic: 0.537
  Top 5 features:
    - SQLE (selected 5/5 folds)
    - WASH3P (selected 5/5 folds)
    - ZNF193 (selected 5/5 folds)
    - OPA1 (selected 4/5 folds)
    - C14orf118 (selected 4/5 folds)
  Runtime: 18.7s

METHYLATION:
------------------------------------------------------------

Rank 1: fold_change_2000
  Mean AUC: 0.654 ± 0.088
  Method: fold_change
  N features: 2000
  Best model: elastic (AUC: 0.600)
  All models:
    - elastic: 0.600
    - mlp: 0.597
    - xgboost: 0.597
    - logistic: 0.590
    - rf: 0.559
  Top 5 features:
    - cg11751117 (selected 5/5 folds)
    - cg16453474 (selected 4/5 folds)
    - cg16218705 (selected 4/5 folds)
    - cg27504861 (selected 3/5 folds)
    - cg09599027 (selected 3/5 folds)
  Runtime: 83.4s

Rank 2: fold_change_5000
  Mean AUC: 0.652 ± 0.064
  Method: fold_change
  N features: 5000
  Best model: elastic (AUC: 0.611)
  All models:
    - elastic: 0.611
    - mlp: 0.604
    - xgboost: 0.588
    - rf: 0.586
    - logistic: 0.579
  Top 5 features:
    - cg11751117 (selected 5/5 folds)
    - cg16453474 (selected 4/5 folds)
    - cg16218705 (selected 4/5 folds)
    - cg27504861 (selected 3/5 folds)
    - cg09599027 (selected 3/5 folds)
  Runtime: 230.5s

Rank 3: f_test_5000
  Mean AUC: 0.652 ± 0.057
  Method: f_test
  N features: 5000
  Best model: elastic (AUC: 0.615)
  All models:
    - elastic: 0.615
    - mlp: 0.600
    - xgboost: 0.587
    - logistic: 0.577
    - rf: 0.574
  Top 5 features:
    - cg23818351 (selected 5/5 folds)
    - cg26366091 (selected 5/5 folds)
    - cg09988853 (selected 5/5 folds)
    - cg20637688 (selected 5/5 folds)
    - cg05822335 (selected 5/5 folds)
  Runtime: 227.1s

MUTATION:
------------------------------------------------------------

Rank 1: fisher_1250
  Mean AUC: 0.631 ± 0.042
  Method: fisher
  N features: 1250
  Best model: xgboost (AUC: 0.590)
  All models:
    - xgboost: 0.590
    - mlp: 0.560
    - rf: 0.540
    - logistic: 0.517
  Top 5 features:
    - GPC5 (selected 4/5 folds)
    - RAI14 (selected 4/5 folds)
    - ACTN2 (selected 4/5 folds)
    - CDKN2A (selected 3/5 folds)
    - LRRC7 (selected 3/5 folds)
  Runtime: 11.9s

Rank 2: fisher_1000
  Mean AUC: 0.625 ± 0.067
  Method: fisher
  N features: 1000
  Best model: xgboost (AUC: 0.602)
  All models:
    - xgboost: 0.602
    - logistic: 0.535
    - mlp: 0.532
    - rf: 0.520
  Top 5 features:
    - GPC5 (selected 4/5 folds)
    - RAI14 (selected 4/5 folds)
    - ACTN2 (selected 4/5 folds)
    - CDKN2A (selected 3/5 folds)
    - LRRC7 (selected 3/5 folds)
  Runtime: 11.7s

Rank 3: fisher_750
  Mean AUC: 0.614 ± 0.059
  Method: fisher
  N features: 750
  Best model: xgboost (AUC: 0.595)
  All models:
    - xgboost: 0.595
    - rf: 0.548
    - logistic: 0.546
    - mlp: 0.539
  Top 5 features:
    - GPC5 (selected 4/5 folds)
    - RAI14 (selected 4/5 folds)
    - ACTN2 (selected 4/5 folds)
    - CDKN2A (selected 3/5 folds)
    - LRRC7 (selected 3/5 folds)
  Runtime: 12.1s

PROTEIN:
------------------------------------------------------------

Rank 1: f_test_93
  Mean AUC: 0.704 ± 0.048
  Method: f_test
  N features: 93
  Best model: xgboost (AUC: 0.614)
  All models:
    - xgboost: 0.614
    - rf: 0.602
    - logistic: 0.583
    - mlp: 0.537
  Top 5 features:
    - 4E-BP1-R-V (selected 5/5 folds)
    - AMPK_alpha-R-C (selected 5/5 folds)
    - ASNS-R-V (selected 5/5 folds)
    - beta-Catenin-R-V (selected 5/5 folds)
    - Bim-R-V (selected 5/5 folds)
  Runtime: 6.0s

Rank 2: f_test_156
  Mean AUC: 0.693 ± 0.034
  Method: f_test
  N features: 156
  Best model: xgboost (AUC: 0.649)
  All models:
    - xgboost: 0.649
    - rf: 0.621
    - logistic: 0.549
    - mlp: 0.514
  Top 5 features:
    - 4E-BP1-R-V (selected 5/5 folds)
    - 4E-BP1_pT37_T46-R-V (selected 5/5 folds)
    - 53BP1-R-E (selected 5/5 folds)
    - AMPK_alpha-R-C (selected 5/5 folds)
    - ASNS-R-V (selected 5/5 folds)
  Runtime: 3.9s

Rank 3: f_test_75
  Mean AUC: 0.690 ± 0.040
  Method: f_test
  N features: 75
  Best model: rf (AUC: 0.618)
  All models:
    - rf: 0.618
    - xgboost: 0.612
    - logistic: 0.605
    - mlp: 0.542
  Top 5 features:
    - 4E-BP1-R-V (selected 5/5 folds)
    - ASNS-R-V (selected 5/5 folds)
    - Bim-R-V (selected 5/5 folds)
    - 4E-BP1_pT37_T46-R-V (selected 4/5 folds)
    - AMPK_alpha-R-C (selected 4/5 folds)
  Runtime: 6.5s

Total runtime: 210.2 minutes
Results saved to: /Users/tobyliu/bladder/individual_model_results.json

Completed at: 2025-07-24 03:21:08










(ml-env) tobyliu@OSXLAP11545 bladder % caffeinate -dims python /Users/tobyliu/bladder/02_model_training/step6_individual_model_training_v2.py
================================================================================
OPTIMIZED MODEL TRAINING - Two-Phase Adaptive Feature Selection
================================================================================

Started at: 2025-07-23 23:51:14
Loading data and labels...
  Loaded 227 training samples
  Class distribution: 159 responders, 68 non-responders

Loading modality data...
  expression: (227, 17689)
  protein: (227, 185)
  methylation: (227, 39575)
  mutation: (227, 1725)

======================================================================
PHASE 1: BROAD FEATURE SEARCH
======================================================================

==================================================
Processing EXPRESSION
==================================================

[1/27] Testing f_test_100...
  Starting 5-fold cross-validation...
  Mean AUC: 0.664 ± 0.063
  Best models: [('rf', 0.6428072935129386), ('xgboost', 0.6373692839418645)]

[2/27] Testing f_test_500...
  Starting 5-fold cross-validation...
  Mean AUC: 0.673 ± 0.073
  Best models: [('xgboost', 0.6537287309464729), ('rf', 0.6258313762850053)]

[3/27] Testing f_test_1000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.659 ± 0.055
  Best models: [('xgboost', 0.6475529510811768), ('rf', 0.6244954138603331)]

[4/27] Testing f_test_1500...
  Starting 5-fold cross-validation...
  Mean AUC: 0.648 ± 0.077
  Best models: [('xgboost', 0.6263647642679901), ('rf', 0.6071002082594825)]

[5/27] Testing f_test_2000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.642 ± 0.088
  Best models: [('rf', 0.6260102800425381), ('xgboost', 0.6109989808578519)]

[6/27] Testing f_test_3000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.642 ± 0.070
  Best models: [('xgboost', 0.6296249113789436), ('rf', 0.6228852800425382)]

[7/27] Testing f_test_5000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.630 ± 0.068
  Best models: [('rf', 0.6145316377171216), ('xgboost', 0.6126861042183622)]

[8/27] Testing f_test_7500...
  Starting 5-fold cross-validation...
  Mean AUC: 0.633 ± 0.081
  Best models: [('rf', 0.6172174096065225), ('xgboost', 0.6147144186458703)]

[9/27] Testing f_test_10000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.640 ± 0.071
  Best models: [('rf', 0.601445077100319), ('mlp', 0.6003511609358383)]

[10/27] Testing fold_change_100...
  Starting 5-fold cross-validation...
  Mean AUC: 0.643 ± 0.066
  Best models: [('rf', 0.6188768388869196), ('xgboost', 0.6101792360864942)]

[11/27] Testing fold_change_500...
  Starting 5-fold cross-validation...
  Mean AUC: 0.671 ± 0.042
  Best models: [('xgboost', 0.6582672367954625), ('rf', 0.6152594381425027)]

[12/27] Testing fold_change_1000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.659 ± 0.078
  Best models: [('xgboost', 0.6381901364764268), ('rf', 0.6223823555476781)]

[13/27] Testing fold_change_1500...
  Starting 5-fold cross-validation...
  Mean AUC: 0.633 ± 0.071
  Best models: [('xgboost', 0.6234712867777384), ('mlp', 0.6112415809996455)]

[14/27] Testing fold_change_2000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.634 ± 0.080
  Best models: [('rf', 0.6076984003899326), ('xgboost', 0.6066842431761786)]

[15/27] Testing fold_change_3000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.632 ± 0.081
  Best models: [('rf', 0.6184442573555476), ('mlp', 0.6001639489542715)]

[16/27] Testing fold_change_5000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.631 ± 0.065
  Best models: [('xgboost', 0.6101349255583126), ('rf', 0.603947514179369)]

[17/27] Testing fold_change_7500...
  Starting 5-fold cross-validation...
  Mean AUC: 0.645 ± 0.070
  Best models: [('xgboost', 0.6270704094292804), ('rf', 0.6079963886919533)]

[18/27] Testing fold_change_10000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.635 ± 0.070
  Best models: [('rf', 0.6252686325771003), ('xgboost', 0.6068448688408365)]

[19/27] Testing lasso_100...
  Starting 5-fold cross-validation...
  Mean AUC: 0.608 ± 0.076
  Best models: [('rf', 0.5931673165544133), ('xgboost', 0.567311015597306)]

[20/27] Testing lasso_500...
  Starting 5-fold cross-validation...
  Mean AUC: 0.641 ± 0.072
  Best models: [('rf', 0.6024060616802552), ('mlp', 0.5819988479262672)]

[21/27] Testing lasso_1000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.616 ± 0.068
  Best models: [('rf', 0.6052402738390642), ('xgboost', 0.5795285359801489)]

[22/27] Testing lasso_1500...
  Starting 5-fold cross-validation...
  Mean AUC: 0.628 ± 0.063
  Best models: [('rf', 0.5988446029776675), ('xgboost', 0.5974798387096774)]

[23/27] Testing lasso_2000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.649 ± 0.088
  Best models: [('rf', 0.6373642990074442), ('mlp', 0.593084234314073)]

[24/27] Testing lasso_3000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.627 ± 0.065
  Best models: [('rf', 0.5939998006026231), ('mlp', 0.5913561237149947)]

[25/27] Testing lasso_5000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.621 ± 0.072
  Best models: [('xgboost', 0.6207328961361218), ('logistic', 0.5840426710386388)]

[26/27] Testing lasso_7500...
  Starting 5-fold cross-validation...
  Mean AUC: 0.645 ± 0.055
  Best models: [('rf', 0.6102728420772775), ('xgboost', 0.6056074973413683)]

[27/27] Testing lasso_10000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.646 ± 0.035
  Best models: [('rf', 0.6073256380716059), ('xgboost', 0.5954116448068061)]

Top 3 expression configurations:
  1. f_test_500: AUC=0.673±0.073
     Best model: xgboost
     Runtime: 24.6s
  2. fold_change_500: AUC=0.671±0.042
     Best model: xgboost
     Runtime: 32.2s
  3. f_test_100: AUC=0.664±0.063
     Best model: rf
     Runtime: 6.9s

==================================================
Processing METHYLATION
==================================================

[1/27] Testing f_test_500...
  Starting 5-fold cross-validation...
  Mean AUC: 0.623 ± 0.077
  Best models: [('rf', 0.6026663860333215), ('mlp', 0.5894552020560085)]

[2/27] Testing f_test_1000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.601 ± 0.083
  Best models: [('mlp', 0.571846198156682), ('rf', 0.5660243486352358)]

[3/27] Testing f_test_2000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.625 ± 0.072
  Best models: [('mlp', 0.5900866270825949), ('rf', 0.5850523971995747)]

[4/27] Testing f_test_3000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.637 ± 0.078
  Best models: [('rf', 0.5979927330733782), ('mlp', 0.5921271269053527)]

[5/27] Testing f_test_5000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.645 ± 0.057
  Best models: [('elastic', 0.6150500708968452), ('xgboost', 0.5872695852534562)]

[6/27] Testing f_test_7500...
  Starting 5-fold cross-validation...
  Mean AUC: 0.639 ± 0.079
  Best models: [('rf', 0.5900916120170152), ('xgboost', 0.5781117068415456)]

[7/27] Testing f_test_10000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.636 ± 0.089
  Best models: [('mlp', 0.6015863169088975), ('rf', 0.5829066598723857)]

[8/27] Testing f_test_15000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.611 ± 0.072
  Best models: [('mlp', 0.5879807692307693), ('logistic', 0.5813884704005672)]

[9/27] Testing f_test_20000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.643 ± 0.088
  Best models: [('rf', 0.5930316155618576), ('logistic', 0.5901564161644806)]

[10/27] Testing fold_change_500...
  Starting 5-fold cross-validation...
  Mean AUC: 0.616 ± 0.072
  Best models: [('mlp', 0.5861119727047146), ('rf', 0.5663854794399148)]

[11/27] Testing fold_change_1000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.624 ± 0.085
  Best models: [('mlp', 0.5985011963842609), ('rf', 0.5938042803970223)]

[12/27] Testing fold_change_2000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.654 ± 0.088
  Best models: [('elastic', 0.6001063452676356), ('xgboost', 0.5966279688053882)]

[13/27] Testing fold_change_3000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.653 ± 0.101
  Best models: [('mlp', 0.592042936901808), ('rf', 0.5838227800425381)]

[14/27] Testing fold_change_5000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.650 ± 0.064
  Best models: [('elastic', 0.6114066377171217), ('mlp', 0.6057758773484581)]

[15/27] Testing fold_change_7500...
  Starting 5-fold cross-validation...
  Mean AUC: 0.650 ± 0.099
  Best models: [('mlp', 0.6016372740163062), ('rf', 0.5957672367954625)]

[16/27] Testing fold_change_10000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.633 ± 0.068
  Best models: [('mlp', 0.6049350850762141), ('logistic', 0.5832860687699397)]

[17/27] Testing fold_change_15000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.633 ± 0.091
  Best models: [('mlp', 0.6254840925203828), ('logistic', 0.5854462070187877)]

[18/27] Testing fold_change_20000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.662 ± 0.072
  Best models: [('mlp', 0.6245048298475717), ('logistic', 0.598354971641262)]

[19/27] Testing lasso_500...
  Starting 5-fold cross-validation...
  Mean AUC: 0.609 ± 0.071
  Best models: [('xgboost', 0.5862847837646225), ('mlp', 0.568113036157391)]

[20/27] Testing lasso_1000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.637 ± 0.089
  Best models: [('xgboost', 0.5848136742289969), ('rf', 0.584187234136831)]

[21/27] Testing lasso_2000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.648 ± 0.055
  Best models: [('logistic', 0.6038550159517901), ('xgboost', 0.5896136121942573)]

[22/27] Testing lasso_3000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.631 ± 0.074
  Best models: [('xgboost', 0.6184198865650479), ('logistic', 0.5903292272243885)]

[23/27] Testing lasso_5000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.650 ± 0.069
  Best models: [('logistic', 0.6070132488479263), ('xgboost', 0.6056063895781637)]

[24/27] Testing lasso_7500...
  Starting 5-fold cross-validation...
  Mean AUC: 0.627 ± 0.063
  Best models: [('xgboost', 0.6039868397731302), ('logistic', 0.5961372297057781)]

[25/27] Testing lasso_10000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.625 ± 0.056
  Best models: [('xgboost', 0.5915865384615384), ('logistic', 0.5879896313364055)]

[26/27] Testing lasso_15000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.631 ± 0.068
  Best models: [('xgboost', 0.5998792538107054), ('logistic', 0.5946295639844026)]

[27/27] Testing lasso_20000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.641 ± 0.067
  Best models: [('xgboost', 0.6104240517546969), ('logistic', 0.5906970046082949)]

Top 3 methylation configurations:
  1. fold_change_20000: AUC=0.662±0.072
     Best model: mlp
     Runtime: 695.5s
  2. fold_change_2000: AUC=0.654±0.088
     Best model: elastic
     Runtime: 135.7s
  3. fold_change_3000: AUC=0.653±0.101
     Best model: mlp
     Runtime: 138.6s

==================================================
Processing MUTATION
==================================================

[1/7] Testing fisher_50...
  Starting 5-fold cross-validation...
  Mean AUC: 0.607 ± 0.029
  Best models: [('xgboost', 0.563623271889401), ('mlp', 0.536108649415101)]

[2/7] Testing fisher_100...
  Starting 5-fold cross-validation...
  Early stopping - poor performance (AUC < 0.55)

[3/7] Testing fisher_200...
  Starting 5-fold cross-validation...
  Mean AUC: 0.611 ± 0.053
  Best models: [('xgboost', 0.5908609535625665), ('rf', 0.5141749379652606)]

[4/7] Testing fisher_300...
  Starting 5-fold cross-validation...
  Mean AUC: 0.614 ± 0.038
  Best models: [('xgboost', 0.5963410581354129), ('rf', 0.541111308046792)]

[5/7] Testing fisher_500...
  Starting 5-fold cross-validation...
  Mean AUC: 0.617 ± 0.052
  Best models: [('xgboost', 0.578761963842609), ('rf', 0.5162076834455867)]

[6/7] Testing fisher_750...
  Starting 5-fold cross-validation...
  Mean AUC: 0.608 ± 0.062
  Best models: [('xgboost', 0.5953839507266926), ('rf', 0.5477235466146756)]

[7/7] Testing fisher_1000...
  Starting 5-fold cross-validation...
  Mean AUC: 0.631 ± 0.068
  Best models: [('xgboost', 0.602150168380007), ('mlp', 0.538668690180787)]

Top 3 mutation configurations:
  1. fisher_1000: AUC=0.631±0.068
     Best model: xgboost
     Runtime: 19.1s
  2. fisher_500: AUC=0.617±0.052
     Best model: xgboost
     Runtime: 11.9s
  3. fisher_300: AUC=0.614±0.038
     Best model: xgboost
     Runtime: 10.5s

==================================================
Processing PROTEIN
==================================================

[1/14] Testing f_test_25...
  Starting 5-fold cross-validation...
  Mean AUC: 0.637 ± 0.104
  Best models: [('xgboost', 0.5826468894009216), ('logistic', 0.5814914923785892)]

[2/14] Testing f_test_50...
  Starting 5-fold cross-validation...
  Mean AUC: 0.670 ± 0.073
  Best models: [('logistic', 0.616944345976604), ('rf', 0.5940053394186459)]

[3/14] Testing f_test_75...
  Starting 5-fold cross-validation...
  Mean AUC: 0.690 ± 0.040
  Best models: [('rf', 0.6176256203473945), ('xgboost', 0.6122884172279333)]

[4/14] Testing f_test_100...
  Starting 5-fold cross-validation...
  Mean AUC: 0.693 ± 0.044
  Best models: [('xgboost', 0.6211394452321871), ('logistic', 0.5991459145693017)]

[5/14] Testing f_test_125...
  Starting 5-fold cross-validation...
  Mean AUC: 0.705 ± 0.048
  Best models: [('xgboost', 0.6048520028358738), ('logistic', 0.5941853509393832)]

[6/14] Testing f_test_150...
  Starting 5-fold cross-validation...
  Mean AUC: 0.672 ± 0.043
  Best models: [('xgboost', 0.6223546614675646), ('rf', 0.5813951169797944)]

[7/14] Testing f_test_185...
  Starting 5-fold cross-validation...
  Mean AUC: 0.670 ± 0.046
  Best models: [('xgboost', 0.6433678216944345), ('rf', 0.608144275079759)]

[8/14] Testing lasso_25...
  Starting 5-fold cross-validation...
  Mean AUC: 0.655 ± 0.047
  Best models: [('rf', 0.6408858782346686), ('xgboost', 0.622955069124424)]

[9/14] Testing lasso_50...
  Starting 5-fold cross-validation...
  Mean AUC: 0.671 ± 0.067
  Best models: [('xgboost', 0.6398462424672102), ('rf', 0.6058290499822757)]

[10/14] Testing lasso_75...
  Starting 5-fold cross-validation...
  Mean AUC: 0.665 ± 0.056
  Best models: [('xgboost', 0.6371876107763205), ('rf', 0.6296620214462957)]

[11/14] Testing lasso_100...
  Starting 5-fold cross-validation...
  Mean AUC: 0.653 ± 0.082
  Best models: [('xgboost', 0.6214053084012762), ('rf', 0.6100230414746545)]

[12/14] Testing lasso_125...
  Starting 5-fold cross-validation...
  Mean AUC: 0.670 ± 0.045
  Best models: [('xgboost', 0.6266815845444877), ('rf', 0.599241182204892)]

[13/14] Testing lasso_150...
  Starting 5-fold cross-validation...
  Mean AUC: 0.673 ± 0.068
  Best models: [('xgboost', 0.6331265508684865), ('rf', 0.6002924494859979)]

[14/14] Testing lasso_185...
  Starting 5-fold cross-validation...
  Mean AUC: 0.683 ± 0.052
  Best models: [('xgboost', 0.6404200638071607), ('rf', 0.5795739542715349)]

Top 3 protein configurations:
  1. f_test_125: AUC=0.705±0.048
     Best model: xgboost
     Runtime: 9.2s
  2. f_test_100: AUC=0.693±0.044
     Best model: xgboost
     Runtime: 9.6s
  3. f_test_75: AUC=0.690±0.040
     Best model: rf
     Runtime: 13.0s

Phase 1 completed in 203.4 minutes

======================================================================
PHASE 2: REFINEMENT AROUND BEST CONFIGS
======================================================================

==================================================
Refining EXPRESSION
Best from Phase 1: f_test_500 (AUC: 0.673)
==================================================

Testing f_test_375...
  Mean AUC: 0.658 ± 0.067
  Best model: xgboost

Testing f_test_500...
  Mean AUC: 0.661 ± 0.054
  Best model: xgboost

Testing f_test_625...
  Mean AUC: 0.670 ± 0.055
  Best model: xgboost

==================================================
Refining METHYLATION
Best from Phase 1: fold_change_20000 (AUC: 0.662)
==================================================

Testing fold_change_15000...
  Mean AUC: 0.633 ± 0.080
  Best model: mlp

Testing fold_change_20000...
  Mean AUC: 0.650 ± 0.089
  Best model: logistic

Testing fold_change_25000...
  Mean AUC: 0.633 ± 0.073
  Best model: mlp

==================================================
Refining MUTATION
Best from Phase 1: fisher_1000 (AUC: 0.631)
==================================================

Testing fisher_750...
  Mean AUC: 0.615 ± 0.059
  Best model: xgboost

Testing fisher_1000...
  Mean AUC: 0.619 ± 0.069
  Best model: xgboost

Testing fisher_1250...
  Mean AUC: 0.620 ± 0.043
  Best model: xgboost

==================================================
Refining PROTEIN
Best from Phase 1: f_test_125 (AUC: 0.705)
==================================================

Testing f_test_93...
  Mean AUC: 0.706 ± 0.050
  Best model: xgboost

Testing f_test_125...
  Mean AUC: 0.685 ± 0.040
  Best model: xgboost

Testing f_test_156...
  Mean AUC: 0.693 ± 0.034
  Best model: xgboost

================================================================================
FINAL RESULTS SUMMARY - TOP 3 CONFIGURATIONS PER MODALITY
================================================================================

EXPRESSION:
------------------------------------------------------------

Rank 1: fold_change_500
  Mean AUC: 0.671 ± 0.042
  Method: fold_change
  N features: 500
  Best model: xgboost (AUC: 0.658)
  All models:
    - xgboost: 0.658
    - rf: 0.615
    - mlp: 0.610
    - elastic: 0.565
    - logistic: 0.539
  Top 5 features:
    - ZFP64 (selected 5/5 folds)
    - AADACL2 (selected 3/5 folds)
    - ZNF69 (selected 3/5 folds)
    - TXNRD1 (selected 3/5 folds)
    - PLXNB1 (selected 3/5 folds)
  Runtime: 32.2s

Rank 2: f_test_625
  Mean AUC: 0.670 ± 0.055
  Method: f_test
  N features: 625
  Best model: xgboost (AUC: 0.643)
  All models:
    - xgboost: 0.643
    - rf: 0.625
    - mlp: 0.613
    - elastic: 0.568
    - logistic: 0.543
  Top 5 features:
    - SQLE (selected 5/5 folds)
    - C14orf118 (selected 5/5 folds)
    - LOC441208 (selected 5/5 folds)
    - WASH3P (selected 5/5 folds)
    - ZNF193 (selected 5/5 folds)
  Runtime: 42.2s

Rank 3: f_test_100
  Mean AUC: 0.664 ± 0.063
  Method: f_test
  N features: 100
  Best model: rf (AUC: 0.643)
  All models:
    - rf: 0.643
    - xgboost: 0.637
    - mlp: 0.622
    - elastic: 0.547
    - logistic: 0.532
  Top 5 features:
    - ZFP64 (selected 5/5 folds)
    - SQLE (selected 4/5 folds)
    - FCF1 (selected 4/5 folds)
    - CAPS (selected 4/5 folds)
    - C14orf118 (selected 3/5 folds)
  Runtime: 6.9s

METHYLATION:
------------------------------------------------------------

Rank 1: fold_change_2000
  Mean AUC: 0.654 ± 0.088
  Method: fold_change
  N features: 2000
  Best model: elastic (AUC: 0.600)
  All models:
    - elastic: 0.600
    - xgboost: 0.597
    - mlp: 0.597
    - logistic: 0.590
    - rf: 0.559
  Top 5 features:
    - cg11751117 (selected 5/5 folds)
    - cg16453474 (selected 4/5 folds)
    - cg16218705 (selected 4/5 folds)
    - cg27504861 (selected 3/5 folds)
    - cg09599027 (selected 3/5 folds)
  Runtime: 135.7s

Rank 2: fold_change_3000
  Mean AUC: 0.653 ± 0.101
  Method: fold_change
  N features: 3000
  Best model: mlp (AUC: 0.592)
  All models:
    - mlp: 0.592
    - rf: 0.584
    - elastic: 0.582
    - logistic: 0.576
    - xgboost: 0.574
  Top 5 features:
    - cg11751117 (selected 5/5 folds)
    - cg16453474 (selected 4/5 folds)
    - cg16218705 (selected 4/5 folds)
    - cg27504861 (selected 3/5 folds)
    - cg09599027 (selected 3/5 folds)
  Runtime: 138.6s

Rank 3: fold_change_20000
  Mean AUC: 0.650 ± 0.089
  Method: fold_change
  N features: 20000
  Best model: logistic (AUC: 0.598)
  All models:
    - logistic: 0.598
    - rf: 0.592
    - xgboost: 0.565
    - mlp: 0.546
    - elastic: 0.511
  Top 5 features:
    - cg11751117 (selected 5/5 folds)
    - cg16453474 (selected 4/5 folds)
    - cg16218705 (selected 4/5 folds)
    - cg27504861 (selected 3/5 folds)
    - cg09599027 (selected 3/5 folds)
  Runtime: 484.9s

MUTATION:
------------------------------------------------------------

Rank 1: fisher_1250
  Mean AUC: 0.620 ± 0.043
  Method: fisher
  N features: 1250
  Best model: xgboost (AUC: 0.590)
  All models:
    - xgboost: 0.590
    - rf: 0.540
    - logistic: 0.517
    - mlp: 0.516
  Top 5 features:
    - GPC5 (selected 4/5 folds)
    - RAI14 (selected 4/5 folds)
    - ACTN2 (selected 4/5 folds)
    - CDKN2A (selected 3/5 folds)
    - LRRC7 (selected 3/5 folds)
  Runtime: 10.8s

Rank 2: fisher_1000
  Mean AUC: 0.619 ± 0.069
  Method: fisher
  N features: 1000
  Best model: xgboost (AUC: 0.602)
  All models:
    - xgboost: 0.602
    - logistic: 0.535
    - rf: 0.520
    - mlp: 0.511
  Top 5 features:
    - GPC5 (selected 4/5 folds)
    - RAI14 (selected 4/5 folds)
    - ACTN2 (selected 4/5 folds)
    - CDKN2A (selected 3/5 folds)
    - LRRC7 (selected 3/5 folds)
  Runtime: 7.8s

Rank 3: fisher_500
  Mean AUC: 0.617 ± 0.052
  Method: fisher
  N features: 500
  Best model: xgboost (AUC: 0.579)
  All models:
    - xgboost: 0.579
    - rf: 0.516
    - logistic: 0.514
    - mlp: 0.481
  Top 5 features:
    - GPC5 (selected 4/5 folds)
    - RAI14 (selected 4/5 folds)
    - ACTN2 (selected 4/5 folds)
    - CDKN2A (selected 3/5 folds)
    - LRRC7 (selected 3/5 folds)
  Runtime: 11.9s

PROTEIN:
------------------------------------------------------------

Rank 1: f_test_93
  Mean AUC: 0.706 ± 0.050
  Method: f_test
  N features: 93
  Best model: xgboost (AUC: 0.614)
  All models:
    - xgboost: 0.614
    - rf: 0.602
    - logistic: 0.583
    - mlp: 0.543
  Top 5 features:
    - 4E-BP1-R-V (selected 5/5 folds)
    - AMPK_alpha-R-C (selected 5/5 folds)
    - ASNS-R-V (selected 5/5 folds)
    - beta-Catenin-R-V (selected 5/5 folds)
    - Bim-R-V (selected 5/5 folds)
  Runtime: 4.3s

Rank 2: f_test_156
  Mean AUC: 0.693 ± 0.034
  Method: f_test
  N features: 156
  Best model: xgboost (AUC: 0.649)
  All models:
    - xgboost: 0.649
    - rf: 0.621
    - mlp: 0.550
    - logistic: 0.549
  Top 5 features:
    - 4E-BP1-R-V (selected 5/5 folds)
    - 4E-BP1_pT37_T46-R-V (selected 5/5 folds)
    - 53BP1-R-E (selected 5/5 folds)
    - AMPK_alpha-R-C (selected 5/5 folds)
    - ASNS-R-V (selected 5/5 folds)
  Runtime: 4.5s

Rank 3: f_test_100
  Mean AUC: 0.693 ± 0.044
  Method: f_test
  N features: 100
  Best model: xgboost (AUC: 0.621)
  All models:
    - xgboost: 0.621
    - logistic: 0.599
    - rf: 0.556
    - mlp: 0.539
  Top 5 features:
    - 4E-BP1-R-V (selected 5/5 folds)
    - AMPK_alpha-R-C (selected 5/5 folds)
    - ASNS-R-V (selected 5/5 folds)
    - beta-Catenin-R-V (selected 5/5 folds)
    - Bim-R-V (selected 5/5 folds)
  Runtime: 9.6s

================================================================================
FINAL MODEL SUMMARY - BEST CONFIGURATIONS
================================================================================

EXPRESSION:
  Best Configuration: fold_change_500
  - Feature Selection Method: fold_change
  - Number of Features: 500
  - Mean CV AUC: 0.6707 ± 0.0421
  - Best Model: xgboost (AUC: 0.6583)
  - All Model AUCs:
    - logistic: 0.5393
    - rf: 0.6153
    - xgboost: 0.6583
    - elastic: 0.5648
    - mlp: 0.6097
  - Top 10 Features: ZFP64, AADACL2, ZNF69, TXNRD1, PLXNB1, UBXN7, AACS, SF3B3, CPSF2, ATP6V1A

METHYLATION:
  Best Configuration: fold_change_2000
  - Feature Selection Method: fold_change
  - Number of Features: 2000
  - Mean CV AUC: 0.6536 ± 0.0882
  - Best Model: elastic (AUC: 0.6001)
  - All Model AUCs:
    - logistic: 0.5897
    - rf: 0.5594
    - xgboost: 0.5966
    - elastic: 0.6001
    - mlp: 0.5965
  - Top 10 Features: cg11751117, cg16453474, cg16218705, cg27504861, cg09599027, cg11089646, cg13065507, cg18331249, cg19284108, cg11316296

MUTATION:
  Best Configuration: fisher_1250
  - Feature Selection Method: fisher
  - Number of Features: 1250
  - Mean CV AUC: 0.6196 ± 0.0432
  - Best Model: xgboost (AUC: 0.5905)
  - All Model AUCs:
    - logistic: 0.5174
    - rf: 0.5401
    - xgboost: 0.5905
    - mlp: 0.5162
  - Top 10 Features: GPC5, RAI14, ACTN2, CDKN2A, LRRC7, ZFYVE16, PKHD1, MAGEC1, ABCC11, DZIP1

PROTEIN:
  Best Configuration: f_test_93
  - Feature Selection Method: f_test
  - Number of Features: 93
  - Mean CV AUC: 0.7060 ± 0.0496
  - Best Model: xgboost (AUC: 0.6139)
  - All Model AUCs:
    - logistic: 0.5835
    - rf: 0.6018
    - xgboost: 0.6139
    - mlp: 0.5434
  - Top 10 Features: 4E-BP1-R-V, AMPK_alpha-R-C, ASNS-R-V, beta-Catenin-R-V, Bim-R-V, CDK1-R-V, 4E-BP1_pT37_T46-R-V, Bcl-xL-R-V, C-Raf-R-V, CD31-M-V

================================================================================
OVERALL SUMMARY
================================================================================
Total runtime: 231.5 minutes
Results saved to: /Users/tobyliu/bladder/v2_model_results_individual.json
Completed at: 2025-07-24 03:42:51










(ml-env) tobyliu@OSXLAP11545 bladder % caffeinate -dims python /Users/tobyliu/bladder/03_fusion_approaches/step7_fusion_aligned.py
================================================================================
STRATEGIC MULTI-MODAL FUSION
================================================================================

Started at: 2025-07-23 23:51:53
Loading data and labels...
  Loaded 227 training samples
  Class distribution: 159 responders, 68 non-responders

Loading modality data...
  expression: (227, 17689)
  protein: (227, 185)
  methylation: (227, 39575)
  mutation: (227, 1725)

======================================================================
PHASE 1: STRATEGIC COARSE SEARCH
======================================================================

Testing individual modalities...

EXPRESSION:
  Testing 375 features... AUC: 0.641
  Testing 500 features... AUC: 0.640
  Testing 250 features... AUC: 0.647
  Testing 450 features... AUC: 0.648
  Testing 300 features... AUC: 0.663
  Testing 600 features... AUC: 0.641

METHYLATION:
  Testing 2000 features... AUC: 0.670
  Testing 1500 features... AUC: 0.661
  Testing 2500 features... AUC: 0.642
  Testing 1000 features... AUC: 0.683
  Testing 3000 features... AUC: 0.660
  Testing 1250 features... AUC: 0.678

PROTEIN:
  Testing 100 features... AUC: 0.685
  Testing 75 features... AUC: 0.690
  Testing 125 features... AUC: 0.685
  Testing 90 features... AUC: 0.700
  Testing 110 features... AUC: 0.710
  Testing 85 features... AUC: 0.701

MUTATION:
  Testing 1000 features... AUC: 0.616
  Testing 750 features... AUC: 0.600
  Testing 1250 features... AUC: 0.608
  Testing 500 features... AUC: 0.608
  Testing 300 features... AUC: 0.614
  Testing 800 features... AUC: 0.611

Best expression: 300 features (AUC: 0.663)

Best methylation: 1000 features (AUC: 0.683)

Best protein: 110 features (AUC: 0.710)

Best mutation: 1000 features (AUC: 0.616)

Phase 1 completed in 38.9 minutes

======================================================================
PHASE 2: GOLDEN RATIO REFINEMENT
======================================================================

EXPRESSION - Refining around 300:
  185 features: 0.650
  300 features: 0.663 (from Phase 1)
  485 features: 0.641

METHYLATION - Refining around 1000:
  618 features: 0.656
  1000 features: 0.683 (from Phase 1)
  1618 features: 0.673

PROTEIN - Refining around 110:
  67 features: 0.669
  110 features: 0.710 (from Phase 1)
  177 features: 0.669

MUTATION - Refining around 1000:
  618 features: 0.602
  1000 features: 0.616 (from Phase 1)
  1618 features: 0.608

==================================================
FINAL BEST CONFIGURATION:
expression: 300 features (AUC: 0.663)
methylation: 1000 features (AUC: 0.683)
protein: 110 features (AUC: 0.710)
mutation: 1000 features (AUC: 0.616)

======================================================================
TESTING FUSION METHODS
======================================================================

Fold 1/5:
  expression: xgboost (AUC: 0.674)
  methylation: rf (AUC: 0.828)
  protein: xgboost (AUC: 0.712)
  mutation: xgboost (AUC: 0.547)
  Weighted fusion: 0.779
  Simple average: 0.763

Fold 2/5:
  expression: lr (AUC: 0.674)
  methylation: lr (AUC: 0.596)
  protein: lr (AUC: 0.737)
  mutation: xgboost (AUC: 0.647)
  Weighted fusion: 0.728
  Simple average: 0.723

Fold 3/5:
  expression: rf (AUC: 0.656)
  methylation: xgboost (AUC: 0.690)
  protein: rf (AUC: 0.726)
  mutation: xgboost (AUC: 0.572)
  Weighted fusion: 0.673
  Simple average: 0.668

Fold 4/5:
  expression: rf (AUC: 0.710)
  methylation: lr (AUC: 0.762)
  protein: xgboost (AUC: 0.704)
  mutation: lr (AUC: 0.572)
  Weighted fusion: 0.745
  Simple average: 0.738

Fold 5/5:
  expression: lr (AUC: 0.599)
  methylation: lr (AUC: 0.539)
  protein: xgboost (AUC: 0.671)
  mutation: xgboost (AUC: 0.740)
  Weighted fusion: 0.700
  Simple average: 0.682

--------------------------------------------------
FUSION METHOD SUMMARY:
weighted: 0.725 ± 0.036
simple_average: 0.715 ± 0.035
stacking: 0.701 ± 0.041

======================================================================
ABLATION STUDY
======================================================================

Full model (all modalities)...
Full model AUC: 0.725

Removing expression...
  AUC without expression: 0.732 (impact: -0.007)

Removing methylation...
  AUC without methylation: 0.701 (impact: +0.024)

Removing protein...
  AUC without protein: 0.684 (impact: +0.041)

Removing mutation...
  AUC without mutation: 0.721 (impact: +0.004)

======================================================================
BOOTSTRAP CONFIDENCE INTERVALS
======================================================================
  Bootstrap iteration 0/100...
  Bootstrap iteration 20/100...
  Bootstrap iteration 40/100...
  Bootstrap iteration 60/100...
  Bootstrap iteration 80/100...

Bootstrap AUC: 0.707 [95% CI: 0.506-0.889]

================================================================================
FINAL FUSION MODEL SUMMARY
================================================================================

BEST FEATURE CONFIGURATIONS:
  EXPRESSION: 300 features (Individual AUC: 0.663)
  METHYLATION: 1000 features (Individual AUC: 0.683)
  PROTEIN: 110 features (Individual AUC: 0.710)
  MUTATION: 1000 features (Individual AUC: 0.616)

FUSION METHODS COMPARISON:
  weighted: 0.725 ± 0.036
  simple_average: 0.715 ± 0.035
  stacking: 0.701 ± 0.041

BEST FUSION: weighted (AUC: 0.725)

MODALITY WEIGHTS (approximate from performance):
  expression: 24.8%
  methylation: 25.6%
  protein: 26.6%
  mutation: 23.0%

ABLATION STUDY - MODALITY CONTRIBUTIONS:
  Full model: 0.725
  expression contribution: +-0.007 AUC
  methylation contribution: +0.024 AUC
  protein contribution: +0.041 AUC
  mutation contribution: +0.004 AUC

BOOTSTRAP CONFIDENCE INTERVALS:
  Mean: 0.707
  95% CI: [0.506, 0.889]

PERFORMANCE SUMMARY:
  Best fusion AUC: 0.725
  Improvement over best individual: +0.015

Total runtime: 185.7 minutes
Results saved to: /Users/tobyliu/bladder/v2_fusion_results/v2_fusion_results.json
Completed at: 2025-07-24 02:57:43










(ml-env) tobyliu@OSXLAP11545 bladder % caffeinate -dims python /Users/tobyliu/bladder/03_fusion_approaches/step7_fusion_multi_modal.py
================================================================================
STRATEGIC MULTI-MODAL FUSION
================================================================================

Started at: 2025-07-23 23:52:14
Loading data and labels...
  Loaded 227 training samples
  Class distribution: 159 responders, 68 non-responders

Loading modality data...
  expression: (227, 17689)
  protein: (227, 185)
  methylation: (227, 39575)
  mutation: (227, 1725)

======================================================================
PHASE 1: STRATEGIC COARSE SEARCH
======================================================================

Testing individual modalities...

EXPRESSION:
  1500 features: 0.620
  2000 features: 0.641
  3000 features: 0.636
  1000 features: 0.608
  5000 features: 0.657
  500 features: 0.640
  7500 features: 0.634
  100 features: 0.611

METHYLATION:
  5000 features: 0.634
  3000 features: 0.660
  7500 features: 0.637
  2000 features: 0.670
  10000 features: 0.634
  1000 features: 0.683
  500 features: 0.668

PROTEIN:
  120 features: 0.667
  90 features: 0.700
  150 features: 0.670
  60 features: 0.634
  185 features: 0.669
  30 features: 0.645

MUTATION:
  300 features: 0.614
  200 features: 0.603
  500 features: 0.608
  100 features: Early stopped (AUC < 0.55)
  100 features: 0.549
  750 features: 0.600
  50 features: 0.589
  1000 features: 0.616

Best expression: 5000 features (AUC: 0.657)

Best methylation: 1000 features (AUC: 0.683)

Best protein: 90 features (AUC: 0.700)

Best mutation: 1000 features (AUC: 0.616)

Phase 1 completed in 49.9 minutes

======================================================================
PHASE 2: GOLDEN RATIO REFINEMENT
======================================================================

EXPRESSION - Refining around 5000:
  3090 features: 0.640
  5000 features: 0.657 (from Phase 1)
  8090 features: 0.645

METHYLATION - Refining around 1000:
  618 features: 0.656
  1000 features: 0.683 (from Phase 1)
  1618 features: 0.673

PROTEIN - Refining around 90:
  55 features: 0.653
  90 features: 0.700 (from Phase 1)
  145 features: 0.687

MUTATION - Refining around 1000:
  618 features: 0.602
  1000 features: 0.616 (from Phase 1)
  1618 features: 0.608

==================================================
FINAL BEST CONFIGURATION:
expression: 5000 features (AUC: 0.657)
methylation: 1000 features (AUC: 0.683)
protein: 90 features (AUC: 0.700)
mutation: 1000 features (AUC: 0.616)

======================================================================
TESTING FUSION METHODS
======================================================================

Fold 1/5:
  expression: rf (AUC: 0.757)
  methylation: rf (AUC: 0.828)
  protein: rf (AUC: 0.750)
  mutation: xgboost (AUC: 0.547)
  Weighted fusion: 0.855
  Simple average: 0.842

Fold 2/5:
  expression: xgboost (AUC: 0.614)
  methylation: lr (AUC: 0.596)
  protein: lr (AUC: 0.663)
  mutation: xgboost (AUC: 0.647)
  Weighted fusion: 0.688
  Simple average: 0.683

Fold 3/5:
  expression: rf (AUC: 0.665)
  methylation: xgboost (AUC: 0.690)
  protein: rf (AUC: 0.768)
  mutation: xgboost (AUC: 0.572)
  Weighted fusion: 0.697
  Simple average: 0.700

Fold 4/5:
  expression: rf (AUC: 0.633)
  methylation: lr (AUC: 0.762)
  protein: xgboost (AUC: 0.647)
  mutation: lr (AUC: 0.572)
  Weighted fusion: 0.719
  Simple average: 0.704

Fold 5/5:
  expression: xgboost (AUC: 0.615)
  methylation: lr (AUC: 0.539)
  protein: rf (AUC: 0.672)
  mutation: xgboost (AUC: 0.740)
  Weighted fusion: 0.735
  Simple average: 0.719

--------------------------------------------------
FUSION METHOD SUMMARY:
weighted: 0.739 ± 0.060
simple_average: 0.729 ± 0.057
stacking: 0.710 ± 0.045

======================================================================
ABLATION STUDY
======================================================================

Full model (all modalities)...
Full model AUC: 0.739

Removing expression...
  AUC without expression: 0.730 (impact: +0.008)

Removing methylation...
  AUC without methylation: 0.722 (impact: +0.016)

Removing protein...
  AUC without protein: 0.708 (impact: +0.031)

Removing mutation...
  AUC without mutation: 0.734 (impact: +0.004)

======================================================================
BOOTSTRAP CONFIDENCE INTERVALS
======================================================================
  Bootstrap iteration 0/100...
  Bootstrap iteration 20/100...
  Bootstrap iteration 40/100...
  Bootstrap iteration 60/100...
  Bootstrap iteration 80/100...

Bootstrap AUC: 0.687 [95% CI: 0.490-0.868]

================================================================================
FINAL RESULTS SUMMARY
================================================================================

Best fusion method: weighted
Best AUC: 0.739
Bootstrap 95% CI: [0.490, 0.868]

Total runtime: 205.3 minutes
Results saved to: /Users/tobyliu/bladder/fusion_results

Completed at: 2025-07-24 03:17:39










(ml-env) tobyliu@OSXLAP11545 bladder % caffeinate -dims python /Users/tobyliu/bladder/03_fusion_approaches/step7_fusion_VALID_advanced.py
================================================================================
VALID ADVANCED MULTI-MODAL FUSION
================================================================================

This combines:
- All advanced methods from step7_fusion_advanced.py
- VALID nested cross-validation (no data leakage)
- Multiple strategies and fusion methods
- Goal: Maximize AUC while maintaining validity
Loading data and labels...
  Loaded 227 training samples
  Class distribution: 159 responders, 68 non-responders

Loading modality data...
  expression: (227, 17689)
  protein: (227, 185)
  methylation: (227, 39575)
  mutation: (227, 1725)

======================================================================
VALID NESTED CV WITH ADVANCED FUSION METHODS
======================================================================

==================================================
Testing MINIMAL Strategy
==================================================

Outer Fold 1/5:
  expression: rf (AUC: 0.757)
  methylation: rf (AUC: 0.812)
  protein: rf (AUC: 0.804)
  mutation: extra_trees (AUC: 0.556)
  Fusion (rank): 0.864

Outer Fold 2/5:
  expression: logistic (AUC: 0.603)
  methylation: xgboost (AUC: 0.623)
  protein: logistic (AUC: 0.748)
  mutation: xgboost (AUC: 0.621)
  Fusion (rank): 0.752

Outer Fold 3/5:
  expression: logistic (AUC: 0.608)
  methylation: xgboost (AUC: 0.647)
  protein: extra_trees (AUC: 0.731)
  mutation: xgboost (AUC: 0.567)
  Fusion (rank): 0.702

Outer Fold 4/5:
  expression: extra_trees (AUC: 0.659)
  methylation: logistic (AUC: 0.736)
  protein: xgboost (AUC: 0.688)
  mutation: logistic (AUC: 0.596)
  Fusion (rank): 0.738

Outer Fold 5/5:
  expression: logistic (AUC: 0.599)
  methylation: logistic (AUC: 0.562)
  protein: xgboost (AUC: 0.647)
  mutation: xgboost (AUC: 0.620)
  Fusion (trimmed_mean): 0.687

minimal Mean AUC: 0.749 ± 0.062

==================================================
Testing DIVERSE Strategy
==================================================

Outer Fold 1/5:
  expression: xgboost (AUC: 0.728)
  methylation: extra_trees (AUC: 0.801)
  protein: xgboost (AUC: 0.777)
  mutation: extra_trees (AUC: 0.558)
  Fusion (rank): 0.839

Outer Fold 2/5:
  expression: xgboost (AUC: 0.565)
  methylation: extra_trees (AUC: 0.625)
  protein: logistic (AUC: 0.737)
  mutation: xgboost (AUC: 0.623)
  Fusion (weighted_avg): 0.743

Outer Fold 3/5:
  expression: rf (AUC: 0.642)
  methylation: logistic (AUC: 0.623)
  protein: xgboost (AUC: 0.685)
  mutation: rf (AUC: 0.577)
  Fusion (trimmed_mean): 0.709

Outer Fold 4/5:
  expression: logistic (AUC: 0.627)
  methylation: logistic (AUC: 0.726)
  protein: xgboost (AUC: 0.690)
  mutation: extra_trees (AUC: 0.606)
  Fusion (rank): 0.738

Outer Fold 5/5:
  expression: extra_trees (AUC: 0.578)
  methylation: logistic (AUC: 0.569)
  protein: xgboost (AUC: 0.700)
  mutation: xgboost (AUC: 0.682)
  Fusion (rank): 0.714

diverse Mean AUC: 0.749 ± 0.047

==================================================
Testing MIXED_1 Strategy
==================================================

Outer Fold 1/5:
  expression: rf (AUC: 0.757)
  methylation: rf (AUC: 0.812)
  protein: xgboost (AUC: 0.777)
  mutation: extra_trees (AUC: 0.558)
  Fusion (rank): 0.859

Outer Fold 2/5:
  expression: logistic (AUC: 0.603)
  methylation: xgboost (AUC: 0.623)
  protein: logistic (AUC: 0.647)
  mutation: xgboost (AUC: 0.623)
  Fusion (rank): 0.705

Outer Fold 3/5:
  expression: logistic (AUC: 0.608)
  methylation: xgboost (AUC: 0.647)
  protein: xgboost (AUC: 0.685)
  mutation: rf (AUC: 0.577)
  Fusion (rank): 0.690

Outer Fold 4/5:
  expression: extra_trees (AUC: 0.659)
  methylation: logistic (AUC: 0.736)
  protein: xgboost (AUC: 0.690)
  mutation: extra_trees (AUC: 0.606)
  Fusion (rank): 0.732

Outer Fold 5/5:
  expression: logistic (AUC: 0.599)
  methylation: logistic (AUC: 0.562)
  protein: xgboost (AUC: 0.700)
  mutation: xgboost (AUC: 0.707)
  Fusion (trimmed_mean): 0.760

mixed_1 Mean AUC: 0.749 ± 0.060

==================================================
Testing MIXED_2 Strategy
==================================================

Outer Fold 1/5:
  expression: rf (AUC: 0.757)
  methylation: extra_trees (AUC: 0.801)
  protein: xgboost (AUC: 0.777)
  mutation: extra_trees (AUC: 0.558)
  Fusion (rank): 0.839

Outer Fold 2/5:
  expression: xgboost (AUC: 0.576)
  methylation: logistic (AUC: 0.569)
  protein: logistic (AUC: 0.737)
  mutation: xgboost (AUC: 0.623)
  Fusion (weighted_avg): 0.746

Outer Fold 3/5:
  expression: rf (AUC: 0.608)
  methylation: logistic (AUC: 0.603)
  protein: xgboost (AUC: 0.685)
  mutation: rf (AUC: 0.577)
  Fusion (rank): 0.666

Outer Fold 4/5:
  expression: extra_trees (AUC: 0.659)
  methylation: logistic (AUC: 0.726)
  protein: xgboost (AUC: 0.690)
  mutation: extra_trees (AUC: 0.606)
  Fusion (rank): 0.736

Outer Fold 5/5:
  expression: logistic (AUC: 0.599)
  methylation: logistic (AUC: 0.569)
  protein: xgboost (AUC: 0.700)
  mutation: xgboost (AUC: 0.707)
  Fusion (trimmed_mean): 0.726

mixed_2 Mean AUC: 0.742 ± 0.056

======================================================================
TRAIN/VAL/TEST WITH ADVANCED FUSION
======================================================================
Train: 118, Val: 40, Test: 69

Optimizing expression...
  300 features: 0.628
  500 features: 0.688
  1000 features: 0.676
  Best: 500 features

Optimizing methylation...
  400 features: 0.667
  800 features: 0.673
  1500 features: 0.708
  Best: 1500 features

Optimizing protein...
  75 features: 0.577
  100 features: 0.568
  110 features: 0.592
  Best: 110 features

Optimizing mutation...
  200 features: 0.664
  400 features: 0.759
  600 features: 0.753
  Best: 400 features

Final test evaluation...
  expression: rf (AUC: 0.674)
  methylation: logistic (AUC: 0.645)
  protein: xgboost (AUC: 0.659)
  mutation: rf (AUC: 0.546)

TEST FUSION AUC (rank): 0.697

================================================================================
FINAL SUMMARY - VALID ADVANCED RESULTS
================================================================================

Nested CV Results:
  minimal: 0.749 ± 0.062
  diverse: 0.749 ± 0.047
  mixed_1: 0.749 ± 0.060
  mixed_2: 0.742 ± 0.056

Best Strategy: mixed_1
Best Nested CV: 0.749 ± 0.060

Train/Val/Test: 0.697

Runtime: 262.5 minutes
Results saved to: /Users/tobyliu/bladder/VALID_advanced_fusion_results

📊 Valid AUC: 0.749

✅ These results are 100% VALID for your poster!










2025-07-24 04:51:48,723 - INFO - 
BEST MODEL for methylation: logistic with AUC = 0.7511
2025-07-24 04:51:48,730 - INFO - 
============================================================
2025-07-24 04:51:48,730 - INFO - FEATURE SELECTION: PROTEIN
2025-07-24 04:51:48,730 - INFO - ============================================================
2025-07-24 04:51:48,730 - INFO - Selecting 110 features from 185 total features
2025-07-24 04:51:48,730 - INFO - Class distribution: 80 responders, 36 non-responders
2025-07-24 04:51:48,730 - INFO - Using F-test (ANOVA) for protein feature selection
2025-07-24 04:51:48,732 - INFO - Top 5 features by F-score:
2025-07-24 04:51:48,732 - INFO -   TSC1-R-C: F=12.4609, p=6.0050e-04
2025-07-24 04:51:48,732 - INFO -   VEGFR2-R-V: F=10.7374, p=1.3916e-03
2025-07-24 04:51:48,732 - INFO -   PKC-pan_BetaII_pS660-R-V: F=9.6340, p=2.4092e-03
2025-07-24 04:51:48,732 - INFO -   Tuberin-R-E: F=9.0842, p=3.1779e-03
2025-07-24 04:51:48,732 - INFO -   JNK2-R-C: F=7.8616, p=5.9370e-03
2025-07-24 04:51:48,732 - INFO - Selected 110 protein features
2025-07-24 04:51:48,732 - INFO - 
============================================================
2025-07-24 04:51:48,732 - INFO - TRAINING MODELS: PROTEIN
2025-07-24 04:51:48,732 - INFO - ============================================================
2025-07-24 04:51:48,732 - INFO - Training samples: 116, Validation samples: 30
2025-07-24 04:51:48,732 - INFO - Features: 110
2025-07-24 04:51:48,732 - INFO - Class weights: {0: 1.6691176470588236, 1: 0.7138364779874213}
2025-07-24 04:51:48,734 - INFO - Training XGBoost...
2025-07-24 04:51:49,022 - INFO -   XGBoost AUC: 0.6844
2025-07-24 04:51:49,022 - INFO - Training Random Forest...
2025-07-24 04:51:49,140 - INFO -   Random Forest AUC: 0.6356
2025-07-24 04:51:49,140 - INFO - Training Extra Trees...
2025-07-24 04:51:49,235 - INFO -   Extra Trees AUC: 0.7156
2025-07-24 04:51:49,236 - INFO - Training Logistic Regression...
2025-07-24 04:51:49,342 - INFO -   Logistic Regression AUC: 0.5778
2025-07-24 04:51:49,343 - INFO - 
Model comparison:
2025-07-24 04:51:49,343 - INFO -   xgboost: AUC = 0.6844
2025-07-24 04:51:49,344 - INFO -   rf: AUC = 0.6356
2025-07-24 04:51:49,345 - INFO -   extra_trees: AUC = 0.7156
2025-07-24 04:51:49,349 - INFO -   logistic: AUC = 0.5778
2025-07-24 04:51:49,349 - INFO - 
BEST MODEL for protein: extra_trees with AUC = 0.7156
2025-07-24 04:51:49,351 - INFO - 
============================================================
2025-07-24 04:51:49,351 - INFO - FEATURE SELECTION: MUTATION
2025-07-24 04:51:49,351 - INFO - ============================================================
2025-07-24 04:51:49,351 - INFO - Selecting 1000 features from 1725 total features
2025-07-24 04:51:49,351 - INFO - Class distribution: 80 responders, 36 non-responders
2025-07-24 04:51:49,351 - INFO - Using Fisher's exact test for mutation feature selection
2025-07-24 04:51:49,352 - INFO - Found 7 special features (burden/pathway)
2025-07-24 04:51:50,097 - INFO - Top 5 mutations by Fisher's p-value:
2025-07-24 04:51:50,097 - INFO -   LRRC7: p=1.2677e-03
2025-07-24 04:51:50,097 - INFO -   KDM6A: p=6.0982e-03
2025-07-24 04:51:50,097 - INFO -   CDKN2A: p=8.2068e-03
2025-07-24 04:51:50,097 - INFO -   AHNAK: p=1.0787e-02
2025-07-24 04:51:50,097 - INFO -   ZFYVE16: p=1.0821e-02
2025-07-24 04:51:50,097 - INFO - Selected 1000 mutation features (including 7 special features)
2025-07-24 04:51:50,098 - INFO - 
============================================================
2025-07-24 04:51:50,098 - INFO - TRAINING MODELS: MUTATION
2025-07-24 04:51:50,098 - INFO - ============================================================
2025-07-24 04:51:50,098 - INFO - Training samples: 116, Validation samples: 30
2025-07-24 04:51:50,098 - INFO - Features: 1000
2025-07-24 04:51:50,098 - INFO - Class weights: {0: 1.6691176470588236, 1: 0.7138364779874213}
2025-07-24 04:51:50,104 - INFO - Training XGBoost...
2025-07-24 04:51:50,466 - INFO -   XGBoost AUC: 0.5067
2025-07-24 04:51:50,466 - INFO - Training Random Forest...
2025-07-24 04:51:50,595 - INFO -   Random Forest AUC: 0.5867
2025-07-24 04:51:50,595 - INFO - Training Extra Trees...
2025-07-24 04:51:50,689 - INFO -   Extra Trees AUC: 0.6267
2025-07-24 04:51:50,689 - INFO - Training Logistic Regression...
2025-07-24 04:51:50,805 - INFO -   Logistic Regression AUC: 0.5911
2025-07-24 04:51:50,805 - INFO - 
Model comparison:
2025-07-24 04:51:50,806 - INFO -   xgboost: AUC = 0.5067
2025-07-24 04:51:50,807 - INFO -   rf: AUC = 0.5867
2025-07-24 04:51:50,808 - INFO -   extra_trees: AUC = 0.6267
2025-07-24 04:51:50,809 - INFO -   logistic: AUC = 0.5911
2025-07-24 04:51:50,809 - INFO - 
BEST MODEL for mutation: extra_trees with AUC = 0.6267
2025-07-24 04:51:50,811 - INFO - 
============================================================
2025-07-24 04:51:50,811 - INFO - ENSEMBLE FUSION
2025-07-24 04:51:50,811 - INFO - ============================================================
2025-07-24 04:51:50,811 - INFO - Method requested: all
2025-07-24 04:51:50,814 - INFO - 
Modality weights based on individual AUCs:
2025-07-24 04:51:50,814 - INFO -   expression: 0.7956
2025-07-24 04:51:50,814 - INFO -   methylation: 0.7511
2025-07-24 04:51:50,814 - INFO -   protein: 0.7156
2025-07-24 04:51:50,815 - INFO -   mutation: 0.6267
2025-07-24 04:51:50,820 - INFO - 
Fusion method performances:
2025-07-24 04:51:50,820 - INFO -   weighted_avg: AUC = 0.7867
2025-07-24 04:51:50,821 - INFO -   rank: AUC = 0.8089
2025-07-24 04:51:50,821 - INFO -   geometric: AUC = 0.7911
2025-07-24 04:51:50,822 - INFO -   trimmed_mean: AUC = 0.7822
2025-07-24 04:51:50,823 - INFO - 
SELECTED BEST FUSION METHOD: rank with AUC = 0.8089
2025-07-24 04:51:50,828 - INFO - 
============================================================
2025-07-24 04:51:50,828 - INFO - FEATURE SELECTION: EXPRESSION
2025-07-24 04:51:50,828 - INFO - ============================================================
2025-07-24 04:51:50,829 - INFO - Selecting 6000 features from 17689 total features
2025-07-24 04:51:50,829 - INFO - Class distribution: 75 responders, 38 non-responders
2025-07-24 04:51:50,829 - INFO - Using mutual information for expression feature selection
2025-07-24 04:52:00,318 - INFO - Top 5 features by MI score:
2025-07-24 04:52:00,318 - INFO -   MYL5: MI=0.2247
2025-07-24 04:52:00,318 - INFO -   DYRK2: MI=0.1791
2025-07-24 04:52:00,318 - INFO -   TACR3: MI=0.1775
2025-07-24 04:52:00,318 - INFO -   PRSS41: MI=0.1756
2025-07-24 04:52:00,318 - INFO -   FLT3LG: MI=0.1751
2025-07-24 04:52:00,318 - INFO - Selected 6000 features with MI scores ranging from 0.0203 to 0.2247
2025-07-24 04:52:00,321 - INFO - 
============================================================
2025-07-24 04:52:00,321 - INFO - TRAINING MODELS: EXPRESSION
2025-07-24 04:52:00,321 - INFO - ============================================================
2025-07-24 04:52:00,321 - INFO - Training samples: 113, Validation samples: 29
2025-07-24 04:52:00,322 - INFO - Features: 6000
2025-07-24 04:52:00,322 - INFO - Class weights: {0: 1.6691176470588236, 1: 0.7138364779874213}
2025-07-24 04:52:00,355 - INFO - Training XGBoost...
2025-07-24 04:52:02,027 - INFO -   XGBoost AUC: 0.5774
2025-07-24 04:52:02,027 - INFO - Training Random Forest...
2025-07-24 04:52:02,189 - INFO -   Random Forest AUC: 0.7143
2025-07-24 04:52:02,189 - INFO - Training Extra Trees...
2025-07-24 04:52:02,309 - INFO -   Extra Trees AUC: 0.6786
2025-07-24 04:52:02,309 - INFO - Training Logistic Regression...
2025-07-24 04:52:02,555 - INFO -   Logistic Regression AUC: 0.5119
2025-07-24 04:52:02,555 - INFO - 
Model comparison:
2025-07-24 04:52:02,556 - INFO -   xgboost: AUC = 0.5774
2025-07-24 04:52:02,557 - INFO -   rf: AUC = 0.7143
2025-07-24 04:52:02,558 - INFO -   extra_trees: AUC = 0.6786
2025-07-24 04:52:02,559 - INFO -   logistic: AUC = 0.5119
2025-07-24 04:52:02,559 - INFO - 
BEST MODEL for expression: rf with AUC = 0.7143
2025-07-24 04:52:02,592 - INFO - 
============================================================
2025-07-24 04:52:02,592 - INFO - FEATURE SELECTION: METHYLATION
2025-07-24 04:52:02,592 - INFO - ============================================================
2025-07-24 04:52:02,592 - INFO - Selecting 1000 features from 39575 total features
2025-07-24 04:52:02,592 - INFO - Class distribution: 75 responders, 38 non-responders
2025-07-24 04:52:02,592 - INFO - Using mutual information for methylation feature selection
2025-07-24 04:52:24,140 - INFO - Top 5 features by MI score:
2025-07-24 04:52:24,141 - INFO -   cg08673814: MI=0.2029
2025-07-24 04:52:24,141 - INFO -   cg14210311: MI=0.1953
2025-07-24 04:52:24,141 - INFO -   cg01888389: MI=0.1920
2025-07-24 04:52:24,141 - INFO -   cg23925994: MI=0.1899
2025-07-24 04:52:24,141 - INFO -   cg18650307: MI=0.1803
2025-07-24 04:52:24,141 - INFO - Selected 1000 features with MI scores ranging from 0.0933 to 0.2029
2025-07-24 04:52:24,143 - INFO - 
============================================================
2025-07-24 04:52:24,143 - INFO - TRAINING MODELS: METHYLATION
2025-07-24 04:52:24,143 - INFO - ============================================================
2025-07-24 04:52:24,143 - INFO - Training samples: 113, Validation samples: 29
2025-07-24 04:52:24,143 - INFO - Features: 1000
2025-07-24 04:52:24,143 - INFO - Class weights: {0: 1.6691176470588236, 1: 0.7138364779874213}
2025-07-24 04:52:24,149 - INFO - Training XGBoost...
2025-07-24 04:52:24,478 - INFO -   XGBoost AUC: 0.7619
2025-07-24 04:52:24,478 - INFO - Training Random Forest...
2025-07-24 04:52:24,600 - INFO -   Random Forest AUC: 0.7798
2025-07-24 04:52:24,600 - INFO - Training Extra Trees...
2025-07-24 04:52:24,691 - INFO -   Extra Trees AUC: 0.7500
2025-07-24 04:52:24,692 - INFO - Training Logistic Regression...
2025-07-24 04:52:24,706 - INFO -   Logistic Regression AUC: 0.6964
2025-07-24 04:52:24,707 - INFO - 
Model comparison:
2025-07-24 04:52:24,707 - INFO -   xgboost: AUC = 0.7619
2025-07-24 04:52:24,708 - INFO -   rf: AUC = 0.7798
2025-07-24 04:52:24,708 - INFO -   extra_trees: AUC = 0.7500
2025-07-24 04:52:24,709 - INFO -   logistic: AUC = 0.6964
2025-07-24 04:52:24,709 - INFO - 
BEST MODEL for methylation: rf with AUC = 0.7798
2025-07-24 04:52:24,711 - INFO - 
============================================================
2025-07-24 04:52:24,711 - INFO - FEATURE SELECTION: PROTEIN
2025-07-24 04:52:24,711 - INFO - ============================================================
2025-07-24 04:52:24,711 - INFO - Selecting 110 features from 185 total features
2025-07-24 04:52:24,711 - INFO - Class distribution: 75 responders, 38 non-responders
2025-07-24 04:52:24,711 - INFO - Using F-test (ANOVA) for protein feature selection
2025-07-24 04:52:24,712 - INFO - Top 5 features by F-score:
2025-07-24 04:52:24,712 - INFO -   VEGFR2-R-V: F=7.2226, p=8.3069e-03
2025-07-24 04:52:24,712 - INFO -   JNK2-R-C: F=5.8199, p=1.7485e-02
2025-07-24 04:52:24,712 - INFO -   PKC-pan_BetaII_pS660-R-V: F=4.6213, p=3.3747e-02
2025-07-24 04:52:24,713 - INFO -   p90RSK_pT359_S363-R-C: F=4.0720, p=4.6012e-02
2025-07-24 04:52:24,713 - INFO -   ADAR1-M-V: F=3.9866, p=4.8311e-02
2025-07-24 04:52:24,713 - INFO - Selected 110 protein features
2025-07-24 04:52:24,713 - INFO - 
============================================================
2025-07-24 04:52:24,714 - INFO - TRAINING MODELS: PROTEIN
2025-07-24 04:52:24,714 - INFO - ============================================================
2025-07-24 04:52:24,714 - INFO - Training samples: 113, Validation samples: 29
2025-07-24 04:52:24,714 - INFO - Features: 110
2025-07-24 04:52:24,714 - INFO - Class weights: {0: 1.6691176470588236, 1: 0.7138364779874213}
2025-07-24 04:52:24,717 - INFO - Training XGBoost...
2025-07-24 04:52:25,011 - INFO -   XGBoost AUC: 0.6845
2025-07-24 04:52:25,012 - INFO - Training Random Forest...
2025-07-24 04:52:25,139 - INFO -   Random Forest AUC: 0.7262
2025-07-24 04:52:25,139 - INFO - Training Extra Trees...
2025-07-24 04:52:25,234 - INFO -   Extra Trees AUC: 0.7262
2025-07-24 04:52:25,234 - INFO - Training Logistic Regression...
2025-07-24 04:52:25,245 - INFO -   Logistic Regression AUC: 0.6726
2025-07-24 04:52:25,245 - INFO - 
Model comparison:
2025-07-24 04:52:25,246 - INFO -   xgboost: AUC = 0.6845
2025-07-24 04:52:25,247 - INFO -   rf: AUC = 0.7262
2025-07-24 04:52:25,248 - INFO -   extra_trees: AUC = 0.7262
2025-07-24 04:52:25,249 - INFO -   logistic: AUC = 0.6726
2025-07-24 04:52:25,249 - INFO - 
BEST MODEL for protein: rf with AUC = 0.7262
2025-07-24 04:52:25,254 - INFO - 
============================================================
2025-07-24 04:52:25,254 - INFO - FEATURE SELECTION: MUTATION
2025-07-24 04:52:25,254 - INFO - ============================================================
2025-07-24 04:52:25,254 - INFO - Selecting 1000 features from 1725 total features
2025-07-24 04:52:25,254 - INFO - Class distribution: 75 responders, 38 non-responders
2025-07-24 04:52:25,255 - INFO - Using Fisher's exact test for mutation feature selection
2025-07-24 04:52:25,255 - INFO - Found 7 special features (burden/pathway)
2025-07-24 04:52:25,998 - INFO - Top 5 mutations by Fisher's p-value:
2025-07-24 04:52:25,999 - INFO -   CUL1: p=1.1464e-02
2025-07-24 04:52:25,999 - INFO -   EPG5: p=1.5741e-02
2025-07-24 04:52:25,999 - INFO -   GON4L: p=1.5741e-02
2025-07-24 04:52:25,999 - INFO -   DMD: p=2.7545e-02
2025-07-24 04:52:25,999 - INFO -   FLNC: p=2.7545e-02
2025-07-24 04:52:25,999 - INFO - Selected 1000 mutation features (including 7 special features)
2025-07-24 04:52:26,000 - INFO - 
============================================================
2025-07-24 04:52:26,000 - INFO - TRAINING MODELS: MUTATION
2025-07-24 04:52:26,000 - INFO - ============================================================
2025-07-24 04:52:26,000 - INFO - Training samples: 113, Validation samples: 29
2025-07-24 04:52:26,000 - INFO - Features: 1000
2025-07-24 04:52:26,000 - INFO - Class weights: {0: 1.6691176470588236, 1: 0.7138364779874213}
2025-07-24 04:52:26,006 - INFO - Training XGBoost...
2025-07-24 04:52:26,362 - INFO -   XGBoost AUC: 0.6190
2025-07-24 04:52:26,362 - INFO - Training Random Forest...
2025-07-24 04:52:26,481 - INFO -   Random Forest AUC: 0.6310
2025-07-24 04:52:26,481 - INFO - Training Extra Trees...
2025-07-24 04:52:26,592 - INFO -   Extra Trees AUC: 0.5179
2025-07-24 04:52:26,592 - INFO - Training Logistic Regression...
2025-07-24 04:52:26,600 - INFO -   Logistic Regression AUC: 0.5774
2025-07-24 04:52:26,600 - INFO - 
Model comparison:
2025-07-24 04:52:26,601 - INFO -   xgboost: AUC = 0.6190
2025-07-24 04:52:26,602 - INFO -   rf: AUC = 0.6310
2025-07-24 04:52:26,603 - INFO -   extra_trees: AUC = 0.5179
2025-07-24 04:52:26,604 - INFO -   logistic: AUC = 0.5774
2025-07-24 04:52:26,604 - INFO - 
BEST MODEL for mutation: rf with AUC = 0.6310
2025-07-24 04:52:26,607 - INFO - 
============================================================
2025-07-24 04:52:26,607 - INFO - ENSEMBLE FUSION
2025-07-24 04:52:26,607 - INFO - ============================================================
2025-07-24 04:52:26,607 - INFO - Method requested: all
2025-07-24 04:52:26,609 - INFO - 
Modality weights based on individual AUCs:
2025-07-24 04:52:26,609 - INFO -   expression: 0.7143
2025-07-24 04:52:26,609 - INFO -   methylation: 0.7798
2025-07-24 04:52:26,609 - INFO -   protein: 0.7262
2025-07-24 04:52:26,609 - INFO -   mutation: 0.6310
2025-07-24 04:52:26,611 - INFO - 
Fusion method performances:
2025-07-24 04:52:26,612 - INFO -   weighted_avg: AUC = 0.8452
2025-07-24 04:52:26,612 - INFO -   rank: AUC = 0.8393
2025-07-24 04:52:26,612 - INFO -   geometric: AUC = 0.8512
2025-07-24 04:52:26,613 - INFO -   trimmed_mean: AUC = 0.8155
2025-07-24 04:52:26,615 - INFO - 
SELECTED BEST FUSION METHOD: geometric with AUC = 0.8512
2025-07-24 04:52:26,620 - INFO - 
============================================================
2025-07-24 04:52:26,620 - INFO - FEATURE SELECTION: EXPRESSION
2025-07-24 04:52:26,620 - INFO - ============================================================
2025-07-24 04:52:26,620 - INFO - Selecting 6000 features from 17689 total features
2025-07-24 04:52:26,620 - INFO - Class distribution: 76 responders, 33 non-responders
2025-07-24 04:52:26,620 - INFO - Using mutual information for expression feature selection
2025-07-24 04:52:35,912 - INFO - Top 5 features by MI score:
2025-07-24 04:52:35,912 - INFO -   PALMD: MI=0.2110
2025-07-24 04:52:35,912 - INFO -   MYCBP: MI=0.2083
2025-07-24 04:52:35,912 - INFO -   KRT71: MI=0.1894
2025-07-24 04:52:35,912 - INFO -   LOC441455: MI=0.1886
2025-07-24 04:52:35,912 - INFO -   DYRK2: MI=0.1742
2025-07-24 04:52:35,912 - INFO - Selected 6000 features with MI scores ranging from 0.0168 to 0.2110
2025-07-24 04:52:35,916 - INFO - 
============================================================
2025-07-24 04:52:35,916 - INFO - TRAINING MODELS: EXPRESSION
2025-07-24 04:52:35,916 - INFO - ============================================================
2025-07-24 04:52:35,916 - INFO - Training samples: 109, Validation samples: 28
2025-07-24 04:52:35,916 - INFO - Features: 6000
2025-07-24 04:52:35,916 - INFO - Class weights: {0: 1.6691176470588236, 1: 0.7138364779874213}
2025-07-24 04:52:35,948 - INFO - Training XGBoost...
2025-07-24 04:52:37,472 - INFO -   XGBoost AUC: 0.7007
2025-07-24 04:52:37,472 - INFO - Training Random Forest...
2025-07-24 04:52:37,624 - INFO -   Random Forest AUC: 0.6054
2025-07-24 04:52:37,624 - INFO - Training Extra Trees...
2025-07-24 04:52:37,739 - INFO -   Extra Trees AUC: 0.5442
2025-07-24 04:52:37,739 - INFO - Training Logistic Regression...
2025-07-24 04:52:37,800 - INFO -   Logistic Regression AUC: 0.5782
2025-07-24 04:52:37,800 - INFO - 
Model comparison:
2025-07-24 04:52:37,802 - INFO -   xgboost: AUC = 0.7007
2025-07-24 04:52:37,803 - INFO -   rf: AUC = 0.6054
2025-07-24 04:52:37,804 - INFO -   extra_trees: AUC = 0.5442
2025-07-24 04:52:37,804 - INFO -   logistic: AUC = 0.5782
2025-07-24 04:52:37,805 - INFO - 
BEST MODEL for expression: xgboost with AUC = 0.7007
2025-07-24 04:52:37,832 - INFO - 
============================================================
2025-07-24 04:52:37,832 - INFO - FEATURE SELECTION: METHYLATION
2025-07-24 04:52:37,832 - INFO - ============================================================
2025-07-24 04:52:37,832 - INFO - Selecting 1000 features from 39575 total features
2025-07-24 04:52:37,832 - INFO - Class distribution: 76 responders, 33 non-responders
2025-07-24 04:52:37,832 - INFO - Using mutual information for methylation feature selection
2025-07-24 04:52:58,567 - INFO - Top 5 features by MI score:
2025-07-24 04:52:58,567 - INFO -   cg21741562: MI=0.2008
2025-07-24 04:52:58,567 - INFO -   cg01029032: MI=0.1981
2025-07-24 04:52:58,567 - INFO -   cg09411231: MI=0.1979
2025-07-24 04:52:58,567 - INFO -   cg22381282: MI=0.1929
2025-07-24 04:52:58,567 - INFO -   cg14706297: MI=0.1907
2025-07-24 04:52:58,567 - INFO - Selected 1000 features with MI scores ranging from 0.0913 to 0.2008
2025-07-24 04:52:58,568 - INFO - 
============================================================
2025-07-24 04:52:58,568 - INFO - TRAINING MODELS: METHYLATION
2025-07-24 04:52:58,568 - INFO - ============================================================
2025-07-24 04:52:58,568 - INFO - Training samples: 109, Validation samples: 28
2025-07-24 04:52:58,568 - INFO - Features: 1000
2025-07-24 04:52:58,568 - INFO - Class weights: {0: 1.6691176470588236, 1: 0.7138364779874213}
2025-07-24 04:52:58,575 - INFO - Training XGBoost...
2025-07-24 04:52:58,891 - INFO -   XGBoost AUC: 0.3810
2025-07-24 04:52:58,891 - INFO - Training Random Forest...
2025-07-24 04:52:59,010 - INFO -   Random Forest AUC: 0.5170
2025-07-24 04:52:59,010 - INFO - Training Extra Trees...
2025-07-24 04:52:59,107 - INFO -   Extra Trees AUC: 0.5000
2025-07-24 04:52:59,107 - INFO - Training Logistic Regression...
2025-07-24 04:52:59,122 - INFO -   Logistic Regression AUC: 0.4966
2025-07-24 04:52:59,123 - INFO - 
Model comparison:
2025-07-24 04:52:59,123 - INFO -   xgboost: AUC = 0.3810
2025-07-24 04:52:59,124 - INFO -   rf: AUC = 0.5170
2025-07-24 04:52:59,125 - INFO -   extra_trees: AUC = 0.5000
2025-07-24 04:52:59,126 - INFO -   logistic: AUC = 0.4966
2025-07-24 04:52:59,126 - INFO - 
BEST MODEL for methylation: rf with AUC = 0.5170
2025-07-24 04:52:59,132 - INFO - 
============================================================
2025-07-24 04:52:59,132 - INFO - FEATURE SELECTION: PROTEIN
2025-07-24 04:52:59,132 - INFO - ============================================================
2025-07-24 04:52:59,132 - INFO - Selecting 110 features from 185 total features
2025-07-24 04:52:59,132 - INFO - Class distribution: 76 responders, 33 non-responders
2025-07-24 04:52:59,132 - INFO - Using F-test (ANOVA) for protein feature selection
2025-07-24 04:52:59,133 - INFO - Top 5 features by F-score:
2025-07-24 04:52:59,133 - INFO -   VEGFR2-R-V: F=9.5016, p=2.6112e-03
2025-07-24 04:52:59,133 - INFO -   G6PD-M-V: F=8.7172, p=3.8751e-03
2025-07-24 04:52:59,133 - INFO -   Src-M-V: F=5.3907, p=2.2138e-02
2025-07-24 04:52:59,134 - INFO -   SF2-M-V: F=5.3185, p=2.3027e-02
2025-07-24 04:52:59,134 - INFO -   YAP-R-E: F=4.8548, p=2.9713e-02
2025-07-24 04:52:59,134 - INFO - Selected 110 protein features
2025-07-24 04:52:59,134 - INFO - 
============================================================
2025-07-24 04:52:59,134 - INFO - TRAINING MODELS: PROTEIN
2025-07-24 04:52:59,134 - INFO - ============================================================
2025-07-24 04:52:59,134 - INFO - Training samples: 109, Validation samples: 28
2025-07-24 04:52:59,134 - INFO - Features: 110
2025-07-24 04:52:59,134 - INFO - Class weights: {0: 1.6691176470588236, 1: 0.7138364779874213}
2025-07-24 04:52:59,136 - INFO - Training XGBoost...
2025-07-24 04:52:59,405 - INFO -   XGBoost AUC: 0.7619
2025-07-24 04:52:59,405 - INFO - Training Random Forest...
2025-07-24 04:52:59,525 - INFO -   Random Forest AUC: 0.6190
2025-07-24 04:52:59,525 - INFO - Training Extra Trees...
2025-07-24 04:52:59,618 - INFO -   Extra Trees AUC: 0.5578
2025-07-24 04:52:59,618 - INFO - Training Logistic Regression...
2025-07-24 04:52:59,631 - INFO -   Logistic Regression AUC: 0.7007
2025-07-24 04:52:59,631 - INFO - 
Model comparison:
2025-07-24 04:52:59,632 - INFO -   xgboost: AUC = 0.7619
2025-07-24 04:52:59,633 - INFO -   rf: AUC = 0.6190
2025-07-24 04:52:59,633 - INFO -   extra_trees: AUC = 0.5578
2025-07-24 04:52:59,634 - INFO -   logistic: AUC = 0.7007
2025-07-24 04:52:59,634 - INFO - 
BEST MODEL for protein: xgboost with AUC = 0.7619
2025-07-24 04:52:59,636 - INFO - 
============================================================
2025-07-24 04:52:59,636 - INFO - FEATURE SELECTION: MUTATION
2025-07-24 04:52:59,637 - INFO - ============================================================
2025-07-24 04:52:59,637 - INFO - Selecting 1000 features from 1725 total features
2025-07-24 04:52:59,637 - INFO - Class distribution: 76 responders, 33 non-responders
2025-07-24 04:52:59,637 - INFO - Using Fisher's exact test for mutation feature selection
2025-07-24 04:52:59,637 - INFO - Found 7 special features (burden/pathway)
2025-07-24 04:53:00,363 - INFO - Top 5 mutations by Fisher's p-value:
2025-07-24 04:53:00,363 - INFO -   NFE2L2: p=1.7133e-02
2025-07-24 04:53:00,363 - INFO -   SPTBN5: p=2.5669e-02
2025-07-24 04:53:00,363 - INFO -   PTPRB: p=2.5669e-02
2025-07-24 04:53:00,363 - INFO -   ZFYVE16: p=2.5989e-02
2025-07-24 04:53:00,363 - INFO -   RAI14: p=2.5989e-02
2025-07-24 04:53:00,363 - INFO - Selected 1000 mutation features (including 7 special features)
2025-07-24 04:53:00,363 - INFO - 
============================================================
2025-07-24 04:53:00,363 - INFO - TRAINING MODELS: MUTATION
2025-07-24 04:53:00,363 - INFO - ============================================================
2025-07-24 04:53:00,363 - INFO - Training samples: 109, Validation samples: 28
2025-07-24 04:53:00,364 - INFO - Features: 1000
2025-07-24 04:53:00,364 - INFO - Class weights: {0: 1.6691176470588236, 1: 0.7138364779874213}
2025-07-24 04:53:00,370 - INFO - Training XGBoost...
2025-07-24 04:53:00,706 - INFO -   XGBoost AUC: 0.4762
2025-07-24 04:53:00,706 - INFO - Training Random Forest...
2025-07-24 04:53:00,833 - INFO -   Random Forest AUC: 0.6939
2025-07-24 04:53:00,833 - INFO - Training Extra Trees...
2025-07-24 04:53:00,933 - INFO -   Extra Trees AUC: 0.5986
2025-07-24 04:53:00,933 - INFO - Training Logistic Regression...
2025-07-24 04:53:00,944 - INFO -   Logistic Regression AUC: 0.6395
2025-07-24 04:53:00,944 - INFO - 
Model comparison:
2025-07-24 04:53:00,945 - INFO -   xgboost: AUC = 0.4762
2025-07-24 04:53:00,946 - INFO -   rf: AUC = 0.6939
2025-07-24 04:53:00,947 - INFO -   extra_trees: AUC = 0.5986
2025-07-24 04:53:00,948 - INFO -   logistic: AUC = 0.6395
2025-07-24 04:53:00,948 - INFO - 
BEST MODEL for mutation: rf with AUC = 0.6939
2025-07-24 04:53:00,954 - INFO - 
============================================================
2025-07-24 04:53:00,954 - INFO - ENSEMBLE FUSION
2025-07-24 04:53:00,954 - INFO - ============================================================
2025-07-24 04:53:00,954 - INFO - Method requested: all
2025-07-24 04:53:00,956 - INFO - 
Modality weights based on individual AUCs:
2025-07-24 04:53:00,956 - INFO -   expression: 0.7007
2025-07-24 04:53:00,956 - INFO -   methylation: 0.5170
2025-07-24 04:53:00,956 - INFO -   protein: 0.7619
2025-07-24 04:53:00,956 - INFO -   mutation: 0.6939
2025-07-24 04:53:00,958 - INFO - 
Fusion method performances:
2025-07-24 04:53:00,958 - INFO -   weighted_avg: AUC = 0.7755
2025-07-24 04:53:00,959 - INFO -   rank: AUC = 0.7415
2025-07-24 04:53:00,959 - INFO -   geometric: AUC = 0.7687
2025-07-24 04:53:00,960 - INFO -   trimmed_mean: AUC = 0.7211
2025-07-24 04:53:00,961 - INFO - 
SELECTED BEST FUSION METHOD: weighted_avg with AUC = 0.7755

Bootstrap AUC: 0.729 [95% CI: 0.490-0.909]
Based on 200 successful iterations
2025-07-24 04:53:00,964 - INFO - 
================================================================================
2025-07-24 04:53:00,964 - INFO - REPRODUCTION INSTRUCTIONS FOR TOP 2 MODELS
2025-07-24 04:53:00,964 - INFO - ================================================================================
2025-07-24 04:53:00,964 - INFO - 
======================================================================
2025-07-24 04:53:00,964 - INFO - MODEL #1 REPRODUCTION STEPS
2025-07-24 04:53:00,964 - INFO - ======================================================================
2025-07-24 04:53:00,964 - INFO - 
1. Strategy: diverse
2025-07-24 04:53:00,964 - INFO - 
2. Feature Selection:
2025-07-24 04:53:00,964 - INFO -    - expression: Select top 6000 features using mutual_info_classif
2025-07-24 04:53:00,964 - INFO -    - methylation: Select top 1000 features using mutual_info_classif
2025-07-24 04:53:00,964 - INFO -    - protein: Select top 110 features using f_classif
2025-07-24 04:53:00,964 - INFO -    - mutation: Select top 1000 features using fisher_exact
2025-07-24 04:53:00,964 - INFO - 
3. Model Training:
2025-07-24 04:53:00,964 - INFO -    - Train XGBoost, RandomForest, ExtraTrees, and LogisticRegression for each modality
2025-07-24 04:53:00,964 - INFO -    - Select best model per modality based on validation AUC
2025-07-24 04:53:00,964 - INFO - 
4. Fusion Method: ensemble
2025-07-24 04:53:00,965 - INFO -    - Compute weighted average, rank fusion, geometric mean, and trimmed mean
2025-07-24 04:53:00,965 - INFO -    - Select best fusion method based on validation AUC
2025-07-24 04:53:00,965 - INFO - 
5. Expected Performance:
2025-07-24 04:53:00,965 - INFO -    - Mean AUC: 0.7712 ± 0.0518
2025-07-24 04:53:00,965 - INFO -    - Fold AUCs: [0.8638392857142857, 0.7633928571428572, 0.7451923076923077, 0.7764423076923077, 0.7073732718894009]
2025-07-24 04:53:00,965 - INFO - 
======================================================================
2025-07-24 04:53:00,965 - INFO - MODEL #2 REPRODUCTION STEPS
2025-07-24 04:53:00,965 - INFO - ======================================================================
2025-07-24 04:53:00,965 - INFO - 
1. Strategy: minimal
2025-07-24 04:53:00,965 - INFO - 
2. Feature Selection:
2025-07-24 04:53:00,965 - INFO -    - expression: Select top 300 features using mutual_info_classif
2025-07-24 04:53:00,965 - INFO -    - methylation: Select top 400 features using mutual_info_classif
2025-07-24 04:53:00,965 - INFO -    - protein: Select top 110 features using f_classif
2025-07-24 04:53:00,965 - INFO -    - mutation: Select top 400 features using fisher_exact
2025-07-24 04:53:00,965 - INFO - 
3. Model Training:
2025-07-24 04:53:00,965 - INFO -    - Train XGBoost, RandomForest, ExtraTrees, and LogisticRegression for each modality
2025-07-24 04:53:00,965 - INFO -    - Select best model per modality based on validation AUC
2025-07-24 04:53:00,966 - INFO - 
4. Fusion Method: ensemble
2025-07-24 04:53:00,966 - INFO -    - Compute weighted average, rank fusion, geometric mean, and trimmed mean
2025-07-24 04:53:00,966 - INFO -    - Select best fusion method based on validation AUC
2025-07-24 04:53:00,966 - INFO - 
5. Expected Performance:
2025-07-24 04:53:00,966 - INFO -    - Mean AUC: 0.7659 ± 0.0540
2025-07-24 04:53:00,966 - INFO -    - Fold AUCs: [0.8705357142857142, 0.7433035714285714, 0.7211538461538463, 0.7620192307692307, 0.7327188940092165]
2025-07-24 04:53:00,966 - INFO - 
Complete tracking data saved to: /Users/tobyliu/bladder/advanced_fusion_detailed_logs/top_models_complete_tracking.json

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

Bootstrap 95% CI: [0.490, 0.909]

Total runtime: 278.4 minutes
Results saved to: /Users/tobyliu/bladder/advanced_fusion_results

🎯 TARGET ACHIEVED! Fusion AUC ≥ 0.75

Completed at: 2025-07-24 04:53:00

Detailed logs saved to:
  /Users/tobyliu/bladder/advanced_fusion_detailed_logs/
    - fusion_advanced_detailed_log.txt (all detailed execution logs)
    - top_models_detailed_log.txt (top 2 models complete config)
    - top_models_complete_tracking.json (reproducible tracking data)
2025-07-24 04:53:00,967 - INFO - 
================================================================================
2025-07-24 04:53:00,967 - INFO - SCRIPT COMPLETED SUCCESSFULLY
2025-07-24 04:53:00,967 - INFO - ================================================================================
2025-07-24 04:53:00,967 - INFO - Total runtime: 278.4 minutes
2025-07-24 04:53:00,967 - INFO - Best AUC achieved: 0.7712
2025-07-24 04:53:00,967 - INFO - Best strategy: diverse
2025-07-24 04:53:00,967 - INFO - Best fusion method: ensemble
2025-07-24 04:53:00,967 - INFO - 
All detailed logs have been saved for reproduction.










(ml-env) tobyliu@OSXLAP11545 bladder % caffeinate -dims python /Users/tobyliu/bladder/03_fusion_approaches/step7_fusion_advanced_ORIGINAL_NO_LOGS.py
================================================================================
ADVANCED MULTI-MODAL FUSION - TARGETING 0.75+ AUC
================================================================================

Started at: 2025-07-24 00:14:39
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
BOOTSTRAP CONFIDENCE INTERVALS
======================================================================
  Bootstrap iteration 0/200...
  Bootstrap iteration 40/200...
  Bootstrap iteration 80/200...
  Bootstrap iteration 120/200...
  Bootstrap iteration 160/200...

Bootstrap AUC: 0.740 [95% CI: 0.545-0.917]
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

Bootstrap 95% CI: [0.545, 0.917]

Total runtime: 277.4 minutes
Results saved to: /Users/tobyliu/bladder/advanced_fusion_results

🎯 TARGET ACHIEVED! Fusion AUC ≥ 0.75

Completed at: 2025-07-24 04:52:09