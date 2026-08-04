# Clinical ML Benchmarks

This project collects two compact medical machine-learning benchmarks: surgical-skill classification from motion features and COVID/non-COVID CT image classification from extracted image descriptors.

## Contents

| File | Purpose |
| --- | --- |
| `analysis.ipynb` | Combined benchmark notebook |
| `report.pdf` | Preserved written report |
| `scripts/run_summary.sh` | Prints key metrics from the notebook summary |
| `requirements.txt` | Python dependencies for rerunning the notebook |

## One-Command Setup

```bash
bash scripts/setup_env.sh
```

## Quick Run

```bash
bash scripts/run_summary.sh
```

## Result Snapshot

| Task | Best model or method | Metric snapshot |
| --- | --- | --- |
| Surgical skill classification | Random Forest | Accuracy 0.8500, F1 0.8571 |
| COVID CT discriminant analysis | QDA | Accuracy 0.7252, F1 0.7050 |
| COVID CT HoG + SVM | RBF SVM | Accuracy 0.8097, F1 0.7942 |
| Imbalanced COVID CT setting | Balanced-class SVM variant | Accuracy 0.7912, F1 0.8394 |
