# 第六周周报

> 本次实践完整代码见[../code/competition_1_tianchi/week_6_adaboost](../code/competition_1_tianchi/week_6_adaboost)

- [第六周周报](#第六周周报)
  - [一、特征工程](#一特征工程)
  - [二、模型选择](#二模型选择)
    - [1. Adaptive Boosting (Adaboost) 自适应增强](#1-adaptive-boosting-adaboost-自适应增强)
      - [a. 自适应增强 Adaboost 介绍](#a-自适应增强-adaboost-介绍)
      - [b. `AdaBoostClassifier` 参数介绍](#b-adaboostclassifier-参数介绍)
      - [c. Adaboost的使用](#c-adaboost的使用)
      - [d. 本次成绩](#d-本次成绩)

## 一、特征工程

在第二周的特征工程中，进行了**数据选择**和**缺失值处理**。详见[第二周周报](./week_report_2.md)。

在第三周的特征工程中，进行了**非数值数据转化为数值数据**。详见[第三周周报](./week_report_3.md)。

本周数据沿用第二周和第三周的特征工程处理出的数据进行训练。


## 二、模型选择

### 1. Adaptive Boosting (Adaboost) 自适应增强



#### a. 自适应增强 Adaboost 介绍

AdaBoost方法的自适应在于：前一个分类器分错的样本会被用来训练下一个分类器。AdaBoost方法对于噪声数据和异常数据很敏感。但在一些问题中，AdaBoost方法相对于大多数其它学习算法而言，不会很容易出现过拟合现象。

加和模型：每个模型都是基于上一次模型的错误率来建立的，过分关注分错的样本，而对正确分类的样本减少关注度，逐次迭代之后，可以得到一个相对较好的模型

有很高精度；可以使用各种方法构建子分类器；当使用简单分类器时，计算出的结果是可以理解的，并且弱分类器的构造极其简单，不用做特征筛选，不容易发生过拟合

本次实现使用scikit-learn库的自适应增强分类器 `sklearn.ensemble.AdaBoostClassifier`。

#### b. `AdaBoostClassifier` 参数介绍

全部参数介绍可见[官方英文文档-sklearn.ensemble.AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)，在此介绍关键参数（已自行翻译为中文）。


- `n_estimators`: `int`, `default=50`：boosting终止时的最大分类器个数。学习过程会早停以防完美拟合训练集。
- `learning_rate`: `float`, `default=1`：学习率通过 `learning_rate` 参数缩小每个分类器的贡献。`learning_rate` 和 `n_estimators` 间存在权衡。
- `random_state`: `int` or `RandomState`, `default=None`：管控每次boosting迭代中给到估计器的随机种子。

#### c. Adaboost的使用

本次代码中使用Adaboost：

```python
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators = 50, random_state = 0)
clf.fit(train_x, train_y)
result = clf.predict_proba(test_x)
```

#### d. 本次成绩

![](./week_report_6_images/adaboost_score.png)

Adaboost是目前应用的准确率最高的方法（不考虑调参的话）。











