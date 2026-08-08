# ClinicalAnalytics

[中文](README.md)

ClinicalAnalytics is a medical data-analysis collection covering statistical testing, image-quality analysis, clinical machine learning, and ECG time-series analysis.

![ClinicalAnalytics project matrix preview](docs/images/clinical-analytics-preview.svg)

## Features

- `SurgeryStats`: surgical time, error-score, and gaze/fixation statistics.
- `ImageQuality`: TOE image-quality scoring, similarity metrics, and registration analysis.
- `ClinicalBenchmarks`: surgical-skill classification and COVID CT feature classification.
- `CardiacSignals`: ECG clustering, ARIMA forecasting, and association-rule analysis.

## Results

| Subproject | Entry point |
| --- | --- |
| `SurgeryStats` | `bash scripts/run_summary.sh` |
| `ImageQuality` | `bash scripts/run_all.sh` |
| `ClinicalBenchmarks` | `bash scripts/run_summary.sh` |
| `CardiacSignals` | `bash scripts/run_question.sh q1` |

## Quick Start

```bash
cd ImageQuality
bash scripts/setup_env.sh
bash scripts/run_all.sh
```

No-data summary example:

```bash
cd SurgeryStats
conda run -n codex_python bash scripts/run_summary.sh
```

## Requirements

- Python 3.10+
- Subproject dependencies are listed in each directory's `requirements.txt`

## Data Notes

- `ImageQuality` includes runnable `.mat` data.
- Other subprojects keep scripts, notebooks, and summary outputs; some original datasets are not distributed.

## Project Layout

```text
SurgeryStats/           Surgical statistics
ImageQuality/           Image-quality analysis
ClinicalBenchmarks/     Clinical machine-learning benchmarks
CardiacSignals/         ECG time-series analysis
tests/                  Structure tests
docs/images/            README result image
archive/                Original material archive
```

## Tests

```bash
pytest tests/ -q
```
