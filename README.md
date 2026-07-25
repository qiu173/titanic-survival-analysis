# Titanic 生存预测分析

## 项目背景
基于 Kaggle 经典数据集 Titanic - Machine Learning from Disaster，对乘客特征与生存率的关系进行探索性分析，并对比多种分类模型的预测效果。

## 数据来源
[Kaggle: Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)（train.csv）

## 项目流程
1. **数据清洗**：处理 Age / Embarked / Fare 缺失值，将高缺失率的 Cabin 字段转为二值特征 HasCabin
2. **特征工程**：从姓名中提取称谓（Title）、构造家庭规模（FamilySize）、是否独自出行（IsAlone）
3. **探索性分析与可视化**：分析性别、舱位等级、年龄、家庭规模与生存率的关系，绘制相关性热力图
4. **建模**：对比逻辑回归、决策树、随机森林三种模型的准确率，并输出随机森林的特征重要性

## 关键发现
- 性别对生存率影响极大：女性生存率 74.2%，男性仅 18.9%（"女士优先"原则的真实体现）
- 舱位等级与生存率显著相关：一等舱生存率 63.0%，三等舱仅 24.2%
- 随机森林模型准确率达 81.56%，性别（Sex）是最重要的预测特征，其次是票价（Fare）和称谓（Title）

## 技术栈
Python · pandas · numpy · matplotlib · seaborn · scikit-learn

## 如何运行
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 从 Kaggle 下载 train.csv，放到 data/ 目录下
#    data/train.csv

# 3. 运行分析脚本
python analysis.py
```

运行后会在 `images/` 目录生成以下图表：
- `01_survival_by_sex.png` 性别与生存率
- `02_survival_by_pclass.png` 舱位等级与生存率
- `03_age_distribution.png` 年龄分布（生还 vs 遇难）
- `04_survival_by_familysize.png` 家庭规模与生存率
- `05_correlation_heatmap.png` 特征相关性热力图
- `06_feature_importance.png` 随机森林特征重要性

## 目录结构
```
titanic-survival-analysis/
├── data/               # 数据集（.gitignore 中已排除，不上传原始数据）
├── images/             # 可视化输出图表
├── notebooks/          # 可选：Jupyter Notebook 版本
├── analysis.py         # 主分析脚本
├── requirements.txt
└── README.md
```
