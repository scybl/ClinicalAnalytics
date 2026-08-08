# ClinicalAnalytics

[English](README_en.md)

ClinicalAnalytics 是一个医学数据分析项目集合，包含统计检验、图像质量分析、临床机器学习和 ECG 时间序列分析。

![ClinicalAnalytics 项目矩阵预览](docs/images/clinical-analytics-preview.svg)

## 功能说明

- `SurgeryStats`：手术时间、错误评分和眼动/注视统计。
- `ImageQuality`：TOE 图像质量评分、相似度指标和配准分析。
- `ClinicalBenchmarks`：手术技能分类和 COVID CT 特征分类。
- `CardiacSignals`：ECG 聚类、ARIMA 预测和关联规则分析。

## 结果展示

| 子项目 | 运行入口 |
| --- | --- |
| `SurgeryStats` | `bash scripts/run_summary.sh` |
| `ImageQuality` | `bash scripts/run_all.sh` |
| `ClinicalBenchmarks` | `bash scripts/run_summary.sh` |
| `CardiacSignals` | `bash scripts/run_question.sh q1` |

## 快速上手

```bash
cd ImageQuality
bash scripts/setup_env.sh
bash scripts/run_all.sh
```

无数据摘要示例：

```bash
cd SurgeryStats
conda run -n codex_python bash scripts/run_summary.sh
```

## 环境要求

- Python 3.10+
- 各子项目依赖见对应目录的 `requirements.txt`

## 数据说明

- `ImageQuality` 包含可运行的 `.mat` 数据。
- 其他子项目保留脚本、notebook 和结果摘要；部分原始数据未随仓库分发。

## 目录结构

```text
SurgeryStats/           手术统计分析
ImageQuality/           图像质量分析
ClinicalBenchmarks/     临床机器学习基准
CardiacSignals/         ECG 时间序列分析
tests/                  结构测试
docs/images/            README 结果图
archive/                原始材料归档
```

## 测试

```bash
pytest tests/ -q
```
