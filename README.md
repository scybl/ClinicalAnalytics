# Medical Data Science Showcase

This folder has been split into independent, function-named projects. Original submission archives are retained under `archive/` for traceability.

| Project | Focus | Quick command |
| --- | --- | --- |
| `surgical_skill_statistics` | Surgical timing, error, and gaze/fixation statistics | `bash scripts/run_summary.sh` |
| `toe_image_quality_assessment` | TOE image-quality scoring, similarity metrics, and alignment analysis | `bash scripts/run_all.sh` |
| `clinical_ml_benchmarks` | Surgical-motion skill classification and COVID CT feature classifiers | `bash scripts/run_summary.sh` |
| `ecg_signal_mining` | ECG clustering, ARIMA forecasting, and heart-disease association rules | `bash scripts/run_question.sh q1` |

## Quick Start Index

| Need | Start here |
| --- | --- |
| Fastest no-data summary | `cd surgical_skill_statistics && conda run -n codex_python bash scripts/run_summary.sh` |
| Included-dataset full run | `cd toe_image_quality_assessment && bash scripts/setup_env.sh && bash scripts/run_all.sh` |
| Clinical benchmark metrics | `cd clinical_ml_benchmarks && conda run -n codex_python bash scripts/run_summary.sh` |
| ECG question scripts | `cd ecg_signal_mining && bash scripts/run_question.sh q1` |
| Structural tests | `conda run -n codex_python pytest tests/ -q` |

## Shared Python Environment

Each subproject setup script can install into an active conda environment, while retaining a `.venv` fallback for fresh clones. For example:

```bash
cd surgical_skill_statistics
conda run -n codex_python bash scripts/run_summary.sh
```

## Notes

- `toe_image_quality_assessment` includes its `.mat` dataset and can run after dependency installation.
- `surgical_skill_statistics`, `clinical_ml_benchmarks`, and `ecg_signal_mining` retain notebooks/reports and scripts, but some source datasets are not bundled.
- Each subfolder has its own README, dependency file, and script entry point.
