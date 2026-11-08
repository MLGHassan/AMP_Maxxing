# DeepAMP results
All numbers are computational predictions. Nothing here was tested in a laboratory.
Protocol: nested 5-fold cross-validation over 40%-identity clusters of the scorer pool.
`test` columns are the fixed held-out cluster group, scored once per fold (mean +/- sd over
the 5 folds, so they share one test set and understate total uncertainty). `CV` columns are
the pooled out-of-fold predictions, where the decision threshold never saw the fold it is
applied to.

## random fragment negatives, held-out cluster test group
| model | ROC-AUC | PR-AUC | MCC | sensitivity | specificity | CV MCC | CV ROC-AUC |
|---|---|---|---|---|---|---|---|
| charge only (floor) | 0.827 +/- 0.000 | 0.834 +/- 0.000 | 0.532 +/- 0.015 | 0.687 +/- 0.090 | 0.832 +/- 0.075 | 0.444 | 0.765 |
| logistic regression | 0.850 +/- 0.005 | 0.874 +/- 0.004 | 0.588 +/- 0.023 | 0.672 +/- 0.045 | 0.897 +/- 0.055 | 0.513 | 0.803 |
| random forest | 0.882 +/- 0.002 | 0.907 +/- 0.001 | 0.659 +/- 0.009 | 0.747 +/- 0.037 | 0.903 +/- 0.030 | 0.608 | 0.869 |
| CNN-BiLSTM (Veltri) | 0.915 +/- 0.004 | 0.935 +/- 0.004 | 0.718 +/- 0.013 | 0.812 +/- 0.041 | 0.902 +/- 0.031 | 0.670 | 0.892 |
| ESM-2 + mean pool | 0.932 +/- 0.002 | 0.948 +/- 0.002 | 0.763 +/- 0.022 | 0.850 +/- 0.017 | 0.910 +/- 0.038 | 0.755 | 0.936 |
| ESM-2 + multi-head attention | 0.926 +/- 0.004 | 0.944 +/- 0.003 | 0.740 +/- 0.027 | 0.852 +/- 0.027 | 0.886 +/- 0.049 | 0.731 | 0.926 |

## charge-matched fragment negatives, held-out cluster test group
| model | ROC-AUC | PR-AUC | MCC | sensitivity | specificity | CV MCC | CV ROC-AUC |
|---|---|---|---|---|---|---|---|
| charge only (floor) | 0.622 +/- 0.000 | 0.597 +/- 0.000 | 0.112 +/- 0.075 | 0.851 +/- 0.172 | 0.240 +/- 0.232 | 0.033 | 0.558 |
| logistic regression | 0.772 +/- 0.004 | 0.807 +/- 0.003 | 0.436 +/- 0.041 | 0.604 +/- 0.095 | 0.810 +/- 0.125 | 0.415 | 0.770 |
| random forest | 0.853 +/- 0.004 | 0.875 +/- 0.004 | 0.559 +/- 0.028 | 0.668 +/- 0.092 | 0.870 +/- 0.076 | 0.542 | 0.846 |
| CNN-BiLSTM (Veltri) | 0.915 +/- 0.006 | 0.929 +/- 0.005 | 0.705 +/- 0.015 | 0.782 +/- 0.029 | 0.915 +/- 0.016 | 0.707 | 0.912 |
| ESM-2 + mean pool | 0.930 +/- 0.003 | 0.943 +/- 0.003 | 0.726 +/- 0.005 | 0.825 +/- 0.023 | 0.898 +/- 0.020 | 0.746 | 0.938 |
| ESM-2 + 1-head attention | 0.923 +/- 0.005 | 0.937 +/- 0.004 | 0.721 +/- 0.009 | 0.805 +/- 0.016 | 0.911 +/- 0.016 | 0.706 | 0.927 |
| ESM-2 + multi-head attention | 0.927 +/- 0.004 | 0.941 +/- 0.003 | 0.723 +/- 0.007 | 0.820 +/- 0.036 | 0.898 +/- 0.028 | 0.730 | 0.935 |

## real non-AMP peptide negatives, held-out cluster test group
| model | ROC-AUC | PR-AUC | MCC | sensitivity | specificity | CV MCC | CV ROC-AUC |
|---|---|---|---|---|---|---|---|
| charge only (floor) | 0.757 +/- 0.000 | 0.692 +/- 0.000 | 0.441 +/- 0.003 | 0.772 +/- 0.020 | 0.667 +/- 0.024 | 0.345 | 0.678 |
| logistic regression | 0.763 +/- 0.009 | 0.786 +/- 0.008 | 0.362 +/- 0.037 | 0.620 +/- 0.175 | 0.719 +/- 0.173 | 0.376 | 0.760 |
| random forest | 0.840 +/- 0.004 | 0.858 +/- 0.006 | 0.493 +/- 0.013 | 0.719 +/- 0.128 | 0.760 +/- 0.108 | 0.542 | 0.850 |
| CNN-BiLSTM (Veltri) | 0.902 +/- 0.007 | 0.901 +/- 0.008 | 0.659 +/- 0.018 | 0.821 +/- 0.028 | 0.837 +/- 0.028 | 0.635 | 0.889 |
| ESM-2 + mean pool | 0.905 +/- 0.006 | 0.898 +/- 0.013 | 0.666 +/- 0.027 | 0.854 +/- 0.036 | 0.808 +/- 0.066 | 0.680 | 0.913 |
| ESM-2 + multi-head attention | 0.903 +/- 0.007 | 0.897 +/- 0.009 | 0.664 +/- 0.023 | 0.849 +/- 0.032 | 0.813 +/- 0.051 | 0.668 | 0.897 |

## Leakage check: cluster-grouped split vs a naive random split (MCC on the test group)

| model | negatives | cluster split | random split | inflation |
|---|---|---|---|---|
| charge only (floor) | hard | 0.112 +/- 0.075 | 0.212 +/- 0.014 | +0.100 |
| charge only (floor) | peptide | 0.441 +/- 0.003 | 0.445 +/- 0.004 | +0.004 |
| charge only (floor) | random | 0.532 +/- 0.015 | 0.539 +/- 0.010 | +0.007 |
| ESM-2 + multi-head attention | hard | 0.723 +/- 0.007 | 0.722 +/- 0.011 | -0.001 |
| logistic regression | hard | 0.436 +/- 0.041 | 0.463 +/- 0.020 | +0.027 |
| logistic regression | peptide | 0.362 +/- 0.037 | 0.387 +/- 0.011 | +0.025 |
| logistic regression | random | 0.588 +/- 0.023 | 0.588 +/- 0.010 | -0.000 |
| random forest | hard | 0.559 +/- 0.028 | 0.575 +/- 0.014 | +0.016 |
| random forest | peptide | 0.493 +/- 0.013 | 0.536 +/- 0.013 | +0.043 |
| random forest | random | 0.659 +/- 0.009 | 0.660 +/- 0.012 | +0.001 |

## External reference: AMP Scanner v2 (Veltri et al. 2018)

EthnoHack has no numeric leaderboard, so the proposal's external reference is the
published accuracy of AMP Scanner v2. **The datasets are not the same** (different
positives, different negatives, different split), so this is a sanity check on the
order of magnitude, not a head-to-head comparison.

> The reference digits below have NOT yet been checked against the paper by the
> team. Confirm them and set `verified: true` in `external_reference.json`.

| | sensitivity | specificity | MCC | ROC-AUC |
|---|---|---|---|---|
| AMP Scanner v2 (Veltri et al. 2018) (authors' own benchmark (not ours)) | 0.893 | 0.927 | 0.821 | 0.964 |
| ours: ESM-2 + multi-head attention, charge-matched negatives | 0.820 | 0.898 | 0.723 | 0.927 |

## Search audit (ESM judge): judge scores for 120 candidates

| group | mean judge P |
|---|---|
| plant seeds (unmodified) | 0.638 |
| random BLOSUM mutants (same edit count) | 0.614 |
| annealed candidates | 0.926 |

candidates minus plant seeds: +0.288 (95% bootstrap CI over seeds [+0.192, +0.392])

candidates minus random mutants: +0.312 (95% bootstrap CI over seeds [+0.218, +0.411])

novelty: median edit distance 5, median local-alignment identity to the nearest known AMP 79.2%

## Search audit_cnn: judge scores for 120 candidates

| group | mean judge P |
|---|---|
| plant seeds (unmodified) | 0.638 |
| random BLOSUM mutants (same edit count) | 0.611 |
| annealed candidates | 0.879 |

candidates minus plant seeds: +0.242 (95% bootstrap CI over seeds [+0.148, +0.343])

candidates minus random mutants: +0.268 (95% bootstrap CI over seeds [+0.180, +0.363])

novelty: median edit distance 5, median local-alignment identity to the nearest known AMP 79.2%
