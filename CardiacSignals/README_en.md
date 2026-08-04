# CardiacSignals

[中文](README.md)

CardiacSignals contains ECG and heart-disease data-mining experiments: unsupervised ECG beat clustering, PCA-based feature reduction, ARIMA signal forecasting, and association-rule mining for heart-disease indicators.

## Contents

| File | Purpose |
| --- | --- |
| `q1.py` | KMeans clustering with all features and PCA-reduced features |
| `q2.py` | Gaussian mixture clustering |
| `q3.py` | Agglomerative clustering with average/complete linkage |
| `q4.py` | ECG single-signal stationarity analysis and ARIMA forecasting |
| `q5.py` | Heart-disease association-rule mining |
| `report.pdf` | Preserved written report |

## One-Command Setup

```bash
bash scripts/setup_env.sh
```

## Quick Run

```bash
bash scripts/run_question.sh q1
bash scripts/run_question.sh q4
```

The source datasets are not bundled. Place the expected files in this folder before running a question.

| Question | Expected data |
| --- | --- |
| Q1-Q3 | `ecg_signals_preprocessed.csv` |
| Q4 | `single_ecg_signal.csv` |
| Q5 | `heart-statlog.csv` |

## Result Snapshot

| Experiment | Output |
| --- | --- |
| KMeans/GMM/agglomerative clustering | Confusion matrices, macro precision/recall/F1, PCA variance plots |
| ARIMA ECG modelling | ADF test statistics, ACF/PACF plots, and forecast diagnostics |
| Association rules | Frequent itemsets, lift rules, conviction rules, and disease-related rules |
