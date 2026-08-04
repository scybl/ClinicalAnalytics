# ClinicalBenchmarks

[English](README_en.md)

ClinicalBenchmarks 汇总两个紧凑的医学机器学习基准：基于运动特征的手术技能分类，以及基于 CT 图像描述子的 COVID/non-COVID 分类。

## 内容

| 文件 | 用途 |
| --- | --- |
| `analysis.ipynb` | 综合基准 notebook |
| `report.pdf` | 保留的项目报告 |
| `scripts/run_summary.sh` | 打印 notebook 摘要中的关键指标 |
| `requirements.txt` | 复现 notebook 所需依赖 |

## 一键配置

```bash
bash scripts/setup_env.sh
```

## 快速运行

```bash
bash scripts/run_summary.sh
```

## 结果快照

| 任务 | 最佳模型或方法 | 指标快照 |
| --- | --- | --- |
| 手术技能分类 | Random Forest | Accuracy 0.8500, F1 0.8571 |
| COVID CT 判别分析 | QDA | Accuracy 0.7252, F1 0.7050 |
| COVID CT HoG + SVM | RBF SVM | Accuracy 0.8097, F1 0.7942 |
| 不平衡 COVID CT 设置 | Balanced-class SVM variant | Accuracy 0.7912, F1 0.8394 |
