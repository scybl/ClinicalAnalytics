# SurgeryStats

[English](README_en.md)

SurgeryStats 用统计方法比较专家组和新手组的手术技能差异，覆盖完成时间、Mann-Whitney U 检验、操作错误评分和注视图稀疏度分析。

## 目录结构

| 文件 | 用途 |
| --- | --- |
| `analysis.ipynb` | 统计分析 notebook |
| `report.pdf` | 保留的项目报告 |
| `scripts/run_summary.sh` | 轻量命令行摘要 |
| `requirements.txt` | 复现 notebook 所需依赖 |

## 快速上手

```bash
bash scripts/setup_env.sh
```

## 运行

```bash
bash scripts/run_summary.sh
```

摘要命令不需要私有/原始数据。若要完整重跑 notebook，请把预期输入文件放到本目录：

| 预期输入 | 用途 |
| --- | --- |
| `time_experts.csv` | 专家组完成时间统计 |
| `time_novices.csv` | 新手组完成时间统计 |
| `error_data.xlsx` | 操作错误评分 |
| `fixation_maps/experts/` | 专家组注视热图 |
| `fixation_maps/novice/` | 新手组注视热图 |

## 结果展示

| 分析 | 输出 |
| --- | --- |
| 描述统计 | 均值、中位数、方差、标准差、偏度和峰度 |
| 组间比较 | Mann-Whitney U 统计量、p 值、显著性和 rank-biserial 效应量 |
| 错误分析 | 专家/新手操作错误摘要 |
| 注视分析 | 非白像素注视稀疏度和组间比较 |
