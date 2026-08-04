# SurgeryStats

SurgeryStats analyses surgical skill differences between expert and novice groups using completion-time statistics, Mann-Whitney U tests, procedural error scoring, and fixation-map sparsity.

## Contents

| File | Purpose |
| --- | --- |
| `analysis.ipynb` | Notebook containing the statistical analysis workflow |
| `report.pdf` | Preserved written report |
| `scripts/run_summary.sh` | Lightweight command-line project summary |
| `requirements.txt` | Python dependencies for rerunning the notebook |

## One-Command Setup

```bash
bash scripts/setup_env.sh
```

## Quick Run

```bash
bash scripts/run_summary.sh
```

The summary command does not require the private/raw datasets. To rerun the full notebook, place the expected input files in this folder:

| Expected input | Used for |
| --- | --- |
| `time_experts.csv` | Expert completion-time statistics |
| `time_novices.csv` | Novice completion-time statistics |
| `error_data.xlsx` | Procedural error scoring |
| `fixation_maps/experts/` | Expert gaze/fixation heatmaps |
| `fixation_maps/novice/` | Novice gaze/fixation heatmaps |

## Result Snapshot

| Analysis | Output |
| --- | --- |
| Descriptive statistics | Mean, median, variance, standard deviation, skewness, and kurtosis |
| Group comparison | Mann-Whitney U statistic, p-value, significance flag, and rank-biserial effect size |
| Error analysis | Expert/novice procedural error summaries |
| Fixation analysis | Non-white-pixel fixation sparsity and group comparison |
