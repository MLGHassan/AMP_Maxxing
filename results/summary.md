
### random-negatives, cluster-held-out test set (mean ± sd over seeds)

| model | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|
| charge_only | 0.791 ± 0.000 | 0.778 ± 0.000 | 0.485 ± 0.038 |
| logreg | 0.886 ± 0.002 | 0.910 ± 0.002 | 0.607 ± 0.009 |
| random_forest | 0.893 ± 0.006 | 0.914 ± 0.006 | 0.644 ± 0.019 |
| cnn_lstm | 0.932 ± 0.010 | 0.949 ± 0.011 | 0.712 ± 0.046 |
| esm_mean | 0.952 ± 0.003 | 0.968 ± 0.002 | 0.779 ± 0.011 |
| esm_attn | 0.948 ± 0.003 | 0.964 ± 0.002 | 0.768 ± 0.015 |

### hard-negatives, cluster-held-out test set (mean ± sd over seeds)

| model | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|
| charge_only | 0.568 ± 0.000 | 0.543 ± 0.000 | 0.066 ± 0.042 |
| logreg | 0.818 ± 0.002 | 0.814 ± 0.012 | 0.480 ± 0.019 |
| random_forest | 0.870 ± 0.006 | 0.880 ± 0.008 | 0.544 ± 0.035 |
| cnn_lstm | 0.914 ± 0.009 | 0.922 ± 0.011 | 0.696 ± 0.053 |
| esm_mean | 0.941 ± 0.014 | 0.950 ± 0.010 | 0.781 ± 0.039 |
| esm_attn | 0.924 ± 0.022 | 0.936 ± 0.017 | 0.748 ± 0.035 |

### Leakage check: same model, cluster split vs naive random split (MCC)

| model | negatives | cluster split | random split |
|---|---|---|---|
| charge_only | hard | 0.066 ± 0.042 | -0.010 ± 0.073 |
| charge_only | random | 0.485 ± 0.038 | 0.449 ± 0.041 |
| esm_attn | hard | 0.748 ± 0.035 | 0.769 ± 0.025 |
| logreg | hard | 0.480 ± 0.019 | 0.499 ± 0.054 |
| logreg | random | 0.607 ± 0.009 | 0.597 ± 0.047 |
| random_forest | hard | 0.544 ± 0.035 | 0.602 ± 0.050 |
| random_forest | random | 0.644 ± 0.019 | 0.615 ± 0.042 |
