# CardiacSignals

[English](README_en.md)

CardiacSignals 包含 ECG 和心脏病数据挖掘实验：无监督 ECG beat 聚类、基于 PCA 的特征降维、ARIMA 信号预测，以及心脏病指标关联规则挖掘。

## 目录结构

| 文件 | 用途 |
| --- | --- |
| `q1.py` | 使用全量特征和 PCA 降维特征进行 KMeans 聚类 |
| `q2.py` | Gaussian mixture 聚类 |
| `q3.py` | average/complete linkage 层次聚类 |
| `q4.py` | 单条 ECG 信号平稳性分析和 ARIMA 预测 |
| `q5.py` | 心脏病关联规则挖掘 |
| `report.pdf` | 保留的项目报告 |

## 快速上手

```bash
bash scripts/setup_env.sh
```

## 运行

```bash
bash scripts/run_question.sh q1
bash scripts/run_question.sh q4
```

源数据未随仓库分发。运行具体问题前，请把对应数据放到本目录：

| 问题 | 预期数据 |
| --- | --- |
| Q1-Q3 | `ecg_signals_preprocessed.csv` |
| Q4 | `single_ecg_signal.csv` |
| Q5 | `heart-statlog.csv` |

## 结果展示

| 实验 | 输出 |
| --- | --- |
| KMeans/GMM/层次聚类 | 混淆矩阵、macro precision/recall/F1 和 PCA 方差图 |
| ARIMA ECG 建模 | ADF 检验统计量、ACF/PACF 图和预测诊断 |
| 关联规则 | 频繁项集、lift 规则、conviction 规则和疾病相关规则 |
