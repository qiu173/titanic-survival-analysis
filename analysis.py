# -*- coding: utf-8 -*-
"""
Titanic 生存预测分析
数据清洗 -> 探索性分析与可视化 -> 特征工程 -> 建模

使用方法:
1. 去 Kaggle 下载数据集: https://www.kaggle.com/competitions/titanic/data
2. 把 train.csv 放到 data/ 目录下
3. 运行: python analysis.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 中文显示（如果环境没有中文字体，图表标题会显示方框，不影响功能）
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
IMG_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(IMG_DIR, exist_ok=True)



# 1. 数据加载
def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f'原始数据形状: {df.shape}')
    print('\n缺失值统计:')
    print(df.isnull().sum()[df.isnull().sum() > 0])
    return df



# 2. 数据清洗与特征工程
def clean_and_engineer(df):
    df = df.copy()

    # 2.1 缺失值处理
    df['Age'] = df['Age'].fillna(df['Age'].median())
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    else:
        print('提示: 数据中没有 Embarked 列，已跳过该字段相关处理')
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Cabin 缺失率很高，转成"是否有房间记录"的二值特征，而不是直接删列
    if 'Cabin' in df.columns:
        df['HasCabin'] = df['Cabin'].notnull().astype(int)
    else:
        print('提示: 数据中没有 Cabin 列，已跳过 HasCabin 特征构造')

    # 2.2 特征工程
    # 从姓名里提取称谓（Mr / Mrs / Miss / Master / 其他）
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]*)\.')
    rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')

    # 家庭规模 = 兄弟姐妹/配偶 + 父母/子女 + 自己
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    print('\n清洗与特征工程完成，新增字段: HasCabin, Title, FamilySize, IsAlone')
    return df


# 3. 探索性分析与可视化
def explore_and_visualize(df):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')

    # 3.1 性别与生存率
    plt.figure(figsize=(6, 4))
    df.groupby('Sex')['Survived'].mean().plot(kind='bar', color=['#4C72B0', '#DD8452'])
    plt.title('Survival Rate by Sex')
    plt.ylabel('Survival Rate')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, '01_survival_by_sex.png'), dpi=120)
    plt.close()

    # 3.2 舱位等级与生存率
    plt.figure(figsize=(6, 4))
    df.groupby('Pclass')['Survived'].mean().plot(kind='bar', color='#55A868')
    plt.title('Survival Rate by Pclass')
    plt.ylabel('Survival Rate')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, '02_survival_by_pclass.png'), dpi=120)
    plt.close()

    # 3.3 年龄分布（生还 vs 遇难）
    plt.figure(figsize=(6, 4))
    plt.hist(df.loc[df['Survived'] == 0, 'Age'], bins=30, alpha=0.6, label='Not Survived')
    plt.hist(df.loc[df['Survived'] == 1, 'Age'], bins=30, alpha=0.6, label='Survived')
    plt.title('Age Distribution by Survival')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, '03_age_distribution.png'), dpi=120)
    plt.close()

    # 3.4 家庭规模与生存率
    plt.figure(figsize=(6, 4))
    df.groupby('FamilySize')['Survived'].mean().plot(kind='bar', color='#C44E52')
    plt.title('Survival Rate by Family Size')
    plt.ylabel('Survival Rate')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, '04_survival_by_familysize.png'), dpi=120)
    plt.close()

    # 3.5 相关性热力图
    plt.figure(figsize=(7, 5))
    candidate_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'HasCabin', 'FamilySize']
    numeric_cols = [c for c in candidate_cols if c in df.columns]
    corr = df[numeric_cols].corr()
    im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha='right')
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            plt.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                     color='black', fontsize=8)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, '05_correlation_heatmap.png'), dpi=120)
    plt.close()

    print(f'\n可视化图表已保存到: {IMG_DIR}')

    # 打印几个关键结论
    print('\n--- 关键发现 ---')
    print(df.groupby('Sex')['Survived'].mean().round(3))
    print(df.groupby('Pclass')['Survived'].mean().round(3))


# 4. 建模
def build_models(df):
    candidate_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                          'Embarked', 'HasCabin', 'Title', 'FamilySize', 'IsAlone']
    features = [f for f in candidate_features if f in df.columns]
    target = 'Survived'

    model_df = df[features + [target]].copy()

    # 类别特征编码（只对实际存在的列做编码）
    for col in ['Sex', 'Embarked', 'Title']:
        if col in model_df.columns:
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))

    X = model_df[features]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    }

    results = {}
    print('\n--- 模型效果对比 ---')
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        results[name] = acc
        print(f'{name}: accuracy = {acc:.4f}')

    # 用效果最好的随机森林输出详细报告 + 特征重要性
    best_model = models['RandomForest']
    pred = best_model.predict(X_test)
    print('\n--- RandomForest 分类报告 ---')
    print(classification_report(y_test, pred))
    print('混淆矩阵:')
    print(confusion_matrix(y_test, pred))

    importance = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
    print('\n特征重要性排序:')
    print(importance.round(4))

    plt.figure(figsize=(7, 5))
    importance.plot(kind='barh')
    plt.title('Feature Importance (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, '06_feature_importance.png'), dpi=120)
    plt.close()

    return results, importance


def main():
    df = load_data()
    df = clean_and_engineer(df)
    explore_and_visualize(df)
    results, importance = build_models(df)

    print('\n========== 分析完成 ==========')
    print('模型对比结果:', results)
    print('最重要的3个特征:', list(importance.head(3).index))


if __name__ == '__main__':
    main()
