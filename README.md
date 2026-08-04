# ClinicalAnalytics

[English](README_en.md)

ClinicalAnalytics 是一个医学数据科学展示仓库，已经拆分为多个相互独立、按功能命名的子项目。原始提交归档保留在 `archive/`，当前 README 和目录名优先服务于作品展示和快速复现。

| 项目 | 重点能力 | 快速命令 |
| --- | --- | --- |
| `SurgeryStats` | 手术时间、错误评分和眼动/注视统计 | `bash scripts/run_summary.sh` |
| `ImageQuality` | TOE 图像质量评分、相似度指标和配准分析 | `bash scripts/run_all.sh` |
| `ClinicalBenchmarks` | 手术动作技能分类与 COVID CT 特征分类 | `bash scripts/run_summary.sh` |
| `CardiacSignals` | ECG 聚类、ARIMA 预测和心脏病关联规则 | `bash scripts/run_question.sh q1` |

## 快速上手索引

| 目标 | 入口 |
| --- | --- |
| 最快无数据摘要 | `cd SurgeryStats && conda run -n codex_python bash scripts/run_summary.sh` |
| 含数据完整运行 | `cd ImageQuality && bash scripts/setup_env.sh && bash scripts/run_all.sh` |
| 临床基准指标 | `cd ClinicalBenchmarks && conda run -n codex_python bash scripts/run_summary.sh` |
| ECG 问题脚本 | `cd CardiacSignals && bash scripts/run_question.sh q1` |
| 结构测试 | `conda run -n codex_python pytest tests/ -q` |

## 共享 Python 环境

每个子项目的 setup 脚本都可以安装到当前 conda 环境，同时保留 `.venv` 回退方式。当前机器推荐直接复用 `codex_python`：

```bash
cd SurgeryStats
conda run -n codex_python bash scripts/run_summary.sh
```

## 项目说明

- `ImageQuality` 包含 `.mat` 数据集，安装依赖后可运行完整分析。
- `SurgeryStats`、`ClinicalBenchmarks`、`CardiacSignals` 保留 notebook、报告和脚本，但部分原始数据未随仓库分发。
- 每个子目录都有自己的中文 README、英文 `README_en.md`、依赖文件和脚本入口。

## 结果快照

| 项目 | 输出 |
| --- | --- |
| `SurgeryStats` | 描述统计、Mann-Whitney U 检验、错误评分和注视图稀疏度 |
| `ImageQuality` | 相关性、回归、SSIM/互信息/余弦相似度和配准特征分析 |
| `ClinicalBenchmarks` | 传统机器学习分类器的 accuracy、F1 等指标摘要 |
| `CardiacSignals` | 聚类指标、ARIMA 诊断和关联规则结果 |
