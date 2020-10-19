# 第四周周报

> 本次实践完整代码见[../code/competition_1_tianchi/week_3_xgboost](../code/competition_1_tianchi/week_3_xgboost)

- [第四周周报](#第四周周报)
  - [一、特征工程](#一特征工程)
  - [二、模型选择](#二模型选择)
    - [1. Gradient Boosted Decison Trees](#1-gradient-boosted-decison-trees)
      - [a. GBDT与XGBoost介绍](#a-gbdt与xgboost介绍)
      - [b. MacOS中安装XGBoost (Python)](#b-macos中安装xgboost-python)
      - [c. XGBoost参数介绍](#c-xgboost参数介绍)
      - [d. XGBoost的使用](#d-xgboost的使用)
      - [e. 本次成绩](#e-本次成绩)

## 一、特征工程

在第二周的特征工程中，进行了**数据选择**和**缺失值处理**。详见[第二周周报](./week_report_2.md)。

在第三周的特征工程中，进行了**非数值数据转化为数值数据**。详见[第三周周报](./week_report_3.md)。

本周数据沿用前两周特征工程处理出的数据进行训练。


## 二、模型选择

### 1. Gradient Boosted Decison Trees

#### a. GBDT与XGBoost介绍
Gradient Boosted Decison Trees即GBDT，梯度提升树，又可简称为GBT。GBDT每轮迭代的目的是找到决策树，使样本的损失尽量变得更小。

本次代码中使用的是XGBoost中的GBDT模型。XGBoost是一个开源库，实现了梯度提升树模型。

#### b. MacOS中安装XGBoost (Python)

```bash
brew install libomp

pip3 install xgboost
```

在其他环境安装XGBoost的方法可见[官方英文文档-Installation Guide](https://xgboost.readthedocs.io/en/latest/build.html)。

#### c. XGBoost参数介绍

全部参数介绍可见[官方英文文档-XGBoost Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)，在此介绍本次代码中自定义使用的参数（已自行翻译为中文）。

- `eta`：即学习率`learning_rate`，默认为0.3。范围[0, 1]
- `max_depth`：树最大深度，默认为6。此值越大，模型越复杂，越容易过拟合。范围[0, ∞] （只有使用 `lossguided` 生长策略，`tree_method` 设为 `hist` 时才能为0）
- `objective`：确定学习任务及对应的学习目标，默认为 `reg:squarederror`（均方误差回归）。本次代码设为 `binary:logistic` ，即二分类逻辑回归，输出概率。
- `num_round`：提升的轮数。

#### d. XGBoost的使用

本次代码中使用XGBoost：

```python
import xgboost as xgb
# 根据训练数据生成xgboost可用的数据集DMatrix
dtrain = xgb.DMatrix(train_x, label = train_y)
# 确定训练参数
param = {'max_depth': 6, 'eta': 0.3, 'objective':'binary:logistic' }
num_round = 8
# 进行训练
bst = xgb.train(param, dtrain, num_round)
# 保存训练得到的模型
bst.save_model('0001.model')
# 根据测试数据生成xgboost可用的数据集DMatrix
dtest = xgb.DMatrix(test_x)
# 用训练得到的模型进行预测
result = bst.predict(dtest)
```

#### e. 本次成绩

![](./week_report_4_images/xgboost_score.png)

相比第三周使用逻辑回归的成绩0.7033有所提升。











