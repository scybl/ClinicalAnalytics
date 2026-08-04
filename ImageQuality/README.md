# ImageQuality

[English](README_en.md)

ImageQuality 分析经食道超声心动图 TOE 图像质量。项目比较总体印象评分和标准百分比，对视图质量排序，计算图像与 gold reference 的相似度，并研究配准得到的旋转/平移特征是否能解释质量评分。

## 内容

| 文件 | 用途 |
| --- | --- |
| `toe_image_quality.mat` | 随仓库提供的 MATLAB 数据集 |
| `q1.py` | 按视图计算 Pearson 相关和线性回归 |
| `q2.py` | SSIM、互信息、余弦相似度和专家/新手检验 |
| `q3.py` | 用 Gaussian basis ridge regression 预测评分 |
| `q4.py` | ECC 配准特征和评分回归 |
| `scripts/run_all.sh` | 顺序运行四个分析脚本 |

## 一键配置

```bash
bash scripts/setup_env.sh
```

## 快速运行

```bash
bash scripts/run_all.sh
```

脚本默认设置 `MPLBACKEND=Agg`，适合无图形界面的终端环境生成图表。

## 结果快照

| 问题 | 输出 |
| --- | --- |
| Q1 | Pearson 相关、RMSE、R2 和 top-view 散点图 |
| Q2 | 每个视图 top-3 相似参与者和专家/新手 t 检验 |
| Q3 | 最佳 Gaussian basis 回归视图、RMSE、R2、basis 数量和 ridge alpha |
| Q4 | 配准旋转/平移特征和评分回归图 |
