# Medical Data Science Showcase

This folder has been split into independent, function-named projects. Original submission archives are retained under `archive/` for traceability.

| Project | Focus | Quick command |
| --- | --- | --- |
| `surgical_skill_statistics` | Surgical timing, error, and gaze/fixation statistics | `bash scripts/run_summary.sh` |
| `toe_image_quality_assessment` | TOE image-quality scoring, similarity metrics, and alignment analysis | `bash scripts/run_all.sh` |
| `clinical_ml_benchmarks` | Surgical-motion skill classification and COVID CT feature classifiers | `bash scripts/run_summary.sh` |
| `ecg_signal_mining` | ECG clustering, ARIMA forecasting, and heart-disease association rules | `bash scripts/run_question.sh q1` |

## Notes

- `toe_image_quality_assessment` includes its `.mat` dataset and can run after dependency installation.
- `surgical_skill_statistics`, `clinical_ml_benchmarks`, and `ecg_signal_mining` retain notebooks/reports and scripts, but some source datasets are not bundled.
- Each subfolder has its own README, dependency file, and script entry point.
