# ImageQuality

[中文](README.md)

ImageQuality analyses transoesophageal echocardiography image quality. It compares general-impression scores and criterion percentages, ranks image views, computes image similarity against gold references, and studies whether alignment-derived rotation/translation features explain quality scores.

## Project Layout

| File | Purpose |
| --- | --- |
| `toe_image_quality.mat` | Included MATLAB dataset |
| `q1.py` | Pearson correlation and linear regression by view |
| `q2.py` | SSIM, mutual information, cosine similarity, and expert/novice tests |
| `q3.py` | Gaussian-basis ridge regression for score prediction |
| `q4.py` | ECC alignment features and score regression |
| `scripts/run_all.sh` | Runs the four analysis scripts in sequence |

## Quick Start

```bash
bash scripts/setup_env.sh
```

## Run

```bash
bash scripts/run_all.sh
```

The script sets `MPLBACKEND=Agg` by default, so plots can be generated in headless terminal sessions.

## Results

| Question | Output |
| --- | --- |
| Q1 | Pearson correlation, RMSE, R2, and top-view scatter plots |
| Q2 | Top-3 similar participants per view and expert/novice t-tests |
| Q3 | Best Gaussian-basis regression views, RMSE, R2, basis count, and ridge alpha |
| Q4 | Alignment rotation/translation features and score-regression plots |
