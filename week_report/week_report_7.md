# 竞赛1-天池-贷款违约预测 总结

- [竞赛1-天池-贷款违约预测 总结](#竞赛1-天池-贷款违约预测-总结)
  - [一、特征工程](#一特征工程)
    - [1. 已用方法](#1-已用方法)
      - [a. 数据选择](#a-数据选择)
      - [b. 缺失值处理](#b-缺失值处理)
      - [c. 非数值数据数值化](#c-非数值数据数值化)
    - [2. 可改进之处](#2-可改进之处)
  - [二、机器学习模型](#二机器学习模型)
    - [1. 已用方法](#1-已用方法-1)
      - [a. SVM](#a-svm)
      - [b. Logistic Regression](#b-logistic-regression)
      - [c. Gradient Boosted Decison Trees](#c-gradient-boosted-decison-trees)
      - [d. Random Forest 随机森林](#d-random-forest-随机森林)
      - [e. Adaptive Boosting (Adaboost) 自适应增强](#e-adaptive-boosting-adaboost-自适应增强)
    - [2. 可改进之处](#2-可改进之处-1)


## 一、特征工程

### 1. 已用方法

#### a. 数据选择

通过观察发现数据一部分为数值型特征，另一部分为非数值型特征。非数值型特征需要经过处理（如one-hot编码）后才能作为输入数据用以训练。本次实践取所有数值型特征作为输入数据，不使用非数值型特征。

**注意:** 在筛选出所有数据类型为数值的字段后，要去掉标签字段和 id 字段。因为标签字段作为训练集输出要与输入数据区分开；id 字段不能作为量化指标参与训练。

``` python
#所有数值字段
numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
#所有非数值字段
category_fea = list(filter(lambda x: x not in numerical_fea,list(data_train.columns)))
#标签字段为：isDefault
label = 'isDefault'
#去除数值字段中的标签字段
numerical_fea.remove(label)
#去除数值字段中的id字段
numerical_fea.remove("id")
```

#### b. 缺失值处理

```python
#按照中位数填充数值型特征
data_train[numerical_fea] = data_train[numerical_fea].fillna(data_train[numerical_fea].median())
data_test_a[numerical_fea] = data_test_a[numerical_fea].fillna(data_train[numerical_fea].median())
#按照众数填充类别型特征
data_train[category_fea] = data_train[category_fea].fillna(data_train[category_fea].mode())
data_test_a[category_fea] = data_test_a[category_fea].fillna(data_train[category_fea].mode())
```

#### c. 非数值数据数值化

对非数值数据进行数值化处理，以便加入训练。具体的数值化方法均为自己所想，可能有不完善之处。

I. `grade` 数值化

```python
# grade 映射
# 检视grade下有哪些值，放入列表并排序
distinct_grade_value = list(data_train["grade"].unique())
distinct_grade_value.sort()
# 构建grade到数值的映射，grade为A的数值为1，往后依次递增1
grade_number_map = {}
for i in range(0, len(distinct_grade_value)):
    grade_number_map[distinct_grade_value[i]] = i + 1

# 创建使grade映射到数值的函数
def grade_map_to_number(grade_column):
    return grade_number_map[grade_column]

# 将grade映射到数值的函数应用到原数据的grade列上，并将结果储存在新列gradeNumber中
grade_number_column_name = "gradeNumber"
data_train[grade_number_column_name] = data_train["grade"].apply(grade_map_to_number)
data_test_a[grade_number_column_name] = data_test_a["grade"].apply(grade_map_to_number)
# grade_number_column_name是数值型字段，将grade_number_column_name加入numerical_fea
numerical_fea.append(grade_number_column_name)

```

II. `subGrade` 数值化

```python
# subGrade 映射
# 检视subGrade下有哪些值，放入列表并排序
distinct_subGrade_value = list(data_train["subGrade"].unique())
distinct_subGrade_value.sort()

# subGrade到数值的映射：subGrade的数值为 grade字母对应的数值 + 后跟数字的值
# 创建使subGrade映射到数值的函数
def subGrade_map_to_number(subGrade_column):
    return grade_number_map[subGrade_column[0]] + int(subGrade_column[1])

# 将subGrade映射到数值的函数应用到原数据的subGrade列上，并将结果储存在新列subGradeNumber中
subGrade_number_column_name = "subGradeNumber"
data_train[subGrade_number_column_name] = data_train["subGrade"].apply(subGrade_map_to_number)
data_test_a[subGrade_number_column_name] = data_test_a["subGrade"].apply(subGrade_map_to_number)
# subGrade_number_column_name是数值型字段，将subGrade_number_column_name加入numerical_fea
numerical_fea.append(subGrade_number_column_name)

```

III. `employmentLength` 数值化

```python

# employmentLength映射
# 检视employmentLength下有哪些值，放入列表并排序
distinct_employmentLength_value = list(data_train["employmentLength"].unique())
#distinct_employmentLength_value.sort()
#print(distinct_employmentLength_value)

# employmentLength到数值的映射：若employmentLength为n years，其数值为n。10+ years为10，< 1 year为0。
# 创建使employmentLength映射到数值的函数
def employmentLength_map_to_number(employmentLength_column):
    if pd.isnull(employmentLength_column):
        return employmentLength_column
    if employmentLength_column == "10+ years":
        return 10
    if employmentLength_column == "< 1 year":
        return 0
    return int(employmentLength_column[0])

# 将employmentLength映射到数值的函数应用到原数据的employmentLength列上，并将结果储存在新列employmentLengthNumber中
employmentLength_number_column_name = "employmentLengthNumber"
data_train[employmentLength_number_column_name] = data_train["employmentLength"].apply(employmentLength_map_to_number)
data_test_a[employmentLength_number_column_name] = data_test_a["employmentLength"].apply(employmentLength_map_to_number)
# employmentLength_number_column_name是数值型字段，将employmentLength_number_column_name加入numerical_fea
numerical_fea.append(employmentLength_number_column_name)

```

**注意**：数据中有nan，用中位数填补之。

```python
# 数据中有nan，用中位数填补之
data_train[employmentLength_number_column_name] = data_train[employmentLength_number_column_name].fillna(data_train[employmentLength_number_column_name].median())
data_test_a[employmentLength_number_column_name] = data_test_a[employmentLength_number_column_name].fillna(data_train[employmentLength_number_column_name].median())

```

IV. `issueDate` 数值化

```python
# issueDate映射
# 检视issueDate下有哪些值，放入列表并排序
distinct_issueDate_value = list(data_train["issueDate"].unique())
#distinct_issueDate_value.sort()
#print(distinct_issueDate_value)

# issueDate到数值的映射：一个issueDate对应的数值是该issueDate与2020-10-01相差的天数
# 创建使issueDate映射到数值的函数
from datetime import date
def issueDate_map_to_number(issueDate_column):
    year, month, day = [int(i) for i in issueDate_column.split('-')]
    issueDate = date(year, month, day)
    endDate = date(2020, 10, 1)
    return (endDate - issueDate).days
    

# 将issueDate映射到数值的函数应用到原数据的issueDate列上，并将结果储存在新列issueDateNumber中
issueDate_number_column_name = "issueDateNumber"
data_train[issueDate_number_column_name] = data_train["issueDate"].apply(issueDate_map_to_number)
data_test_a[issueDate_number_column_name] = data_test_a["issueDate"].apply(issueDate_map_to_number)
# issueDate_number_column_name是数值型字段，将issueDate_number_column_name加入numerical_fea
numerical_fea.append(issueDate_number_column_name)

```

V. `earliesCreditLine` 数值化

```python

# earliesCreditLine映射
# 检视earliesCreditLine下有哪些值，放入列表并排序
distinct_earliesCreditLine_value = list(data_train["earliesCreditLine"].unique())
#distinct_earliesCreditLine_value.sort()
#print(distinct_earliesCreditLine_value)

# earliesCreditLine到数值的映射：一个earliesCreditLine对应的数值是该earliesCreditLine与2020-10相差的月数
# 创建使earliesCreditLine映射到数值的函数
from datetime import datetime
from dateutil import relativedelta
import calendar
# 构建月份名称到月份数字的映射
month_to_number = {month_name: month_num for month_num,month_name in enumerate(calendar.month_abbr)}
def earliesCreditLine_map_to_number(earliesCreditLine_column):
    year_now = 2020
    month_now = 10
    year_earliesCreditLine = int(earliesCreditLine_column[-4: ])
    month_earliesCreditLine = month_to_number[earliesCreditLine_column[0: 3]]
    return (year_now - year_earliesCreditLine) * 12 + month_now - month_earliesCreditLine
    
# 将earliesCreditLine映射到数值的函数应用到原数据的earliesCreditLine列上，并将结果储存在新列earliesCreditLineNumber中
earliesCreditLine_number_column_name = "earliesCreditLineNumber"
data_train[earliesCreditLine_number_column_name] = data_train["earliesCreditLine"].apply(earliesCreditLine_map_to_number)
data_test_a[earliesCreditLine_number_column_name] = data_test_a["earliesCreditLine"].apply(earliesCreditLine_map_to_number)
# earliesCreditLine_number_column_name是数值型字段，将earliesCreditLine_number_column_name加入numerical_fea
numerical_fea.append(earliesCreditLine_number_column_name)
print(data_train[earliesCreditLine_number_column_name])

```

### 2. 可改进之处

除了以上数据处理步骤，还可以对数据进行离群值处理。剔除含有离群值的数据条目。

具体做法：若某值与该项平均值差的绝对值大于三倍标准差，则认为该值为离群值，从数据中剔除该值所在的数据条目。

并且还可以做特征选择，而不是用所有全部数据进行训练。

## 二、机器学习模型

### 1. 已用方法

#### a. SVM

SVM模型训练时间很长，若使用完整80万条训练数据，数小时都无法完成训练与预测。选取前3万条训练数据训练模型，训练时间数分钟。SVM模型预测效果很不好。

[SVM代码](../code/competition_1_tianchi/week_2_svm_logistic/svm.py)

#### b. Logistic Regression

用逻辑回归模型进行训练，使用完整80万条训练数据，在数分钟内能完成训练与预测。模型预测效果较SVM提升很多。

成绩：

![](./week_report_2_images/logistic_grade.png)

[逻辑回归代码](../code/competition_1_tianchi/week_2_svm_logistic/logistic.py)

#### c. Gradient Boosted Decison Trees

Gradient Boosted Decison Trees即GBDT，梯度提升树，又可简称为GBT。GBDT每轮迭代的目的是找到决策树，使样本的损失尽量变得更小。

代码中使用的是XGBoost中的GBDT模型。XGBoost是一个开源库，实现了梯度提升树模型。

XGBoost全部参数介绍可见[官方英文文档-XGBoost Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)，在此介绍代码中自定义使用的参数（已自行翻译为中文）。

- `eta`：即学习率`learning_rate`，默认为0.3。范围[0, 1]
- `max_depth`：树最大深度，默认为6。此值越大，模型越复杂，越容易过拟合。范围[0, ∞] （只有使用 `lossguided` 生长策略，`tree_method` 设为 `hist` 时才能为0）
- `objective`：确定学习任务及对应的学习目标，默认为 `reg:squarederror`（均方误差回归）。本次代码设为 `binary:logistic` ，即二分类逻辑回归，输出概率。
- `num_round`：提升的轮数。

代码中使用XGBoost：

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

成绩

![](./week_report_4_images/xgboost_score.png)

[XGBoost代码](../code/competition_1_tianchi/week_4_xgboost)

#### d. Random Forest 随机森林

GBDT，梯度提升树——使用Boosting方法的模型。那么很容易想到用运用Bagging方法的模型再做训练。故尝试使用随机森林。

随机森林是由很多决策树构成的，不同决策树之间没有关联。

进行分类任务时，新的输入样本进入，让森林中的每一棵决策树分别进行判断和分类，每个决策树会得到一个自己的分类结果，分类结果中哪一个分类最多，那么这个结果即是最终的结果。

使用scikit-learn库的随机森林分类器 `sklearn.ensemble.RandomForestClassifier`。

`RandomForestClassifier` 全部参数介绍可见[官方英文文档-sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)，在此介绍本次代码中自定义使用的参数（已自行翻译为中文）。

- `max_depth`：树最大深度，默认为None。如果为None，扩展节点到所有叶为纯粹的或者到所有叶包含少于 `min_samples_split` 个样本。
- `min_samples_split`：默认为2。分裂一个内部节点所需最小样本数。
- `random_state`：默认为None。控制建立树时有放回的抽取样本（Bootstrapping）的随机性，以及寻找每个节点最优分裂时特征采样的随机性。

代码中使用随机森林：

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth = 2, random_state = 0)
clf.fit(train_x, train_y)
result = clf.predict_proba(test_x)
```

成绩

![](./week_report_5_images/random_forest_score.png)

[随机森林代码](../code/competition_1_tianchi/week_5_random_forest)

#### e. Adaptive Boosting (Adaboost) 自适应增强

AdaBoost方法的自适应在于：前一个分类器分错的样本会被用来训练下一个分类器。AdaBoost方法对于噪声数据和异常数据很敏感。但在一些问题中，AdaBoost方法相对于大多数其它学习算法而言，不会很容易出现过拟合现象。

加和模型：每个模型都是基于上一次模型的错误率来建立的，过分关注分错的样本，而对正确分类的样本减少关注度，逐次迭代之后，可以得到一个相对较好的模型

有很高精度；可以使用各种方法构建子分类器；当使用简单分类器时，计算出的结果是可以理解的，并且弱分类器的构造极其简单，不用做特征筛选，不容易发生过拟合

本次实现使用scikit-learn库的自适应增强分类器 `sklearn.ensemble.AdaBoostClassifier`。


`AdaBoostClassifier` 全部参数介绍可见[官方英文文档-sklearn.ensemble.AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)，在此介绍关键参数（已自行翻译为中文）。


- `n_estimators`: `int`, `default=50`：boosting终止时的最大分类器个数。学习过程会早停以防完美拟合训练集。
- `learning_rate`: `float`, `default=1`：学习率通过 `learning_rate` 参数缩小每个分类器的贡献。`learning_rate` 和 `n_estimators` 间存在权衡。
- `random_state`: `int` or `RandomState`, `default=None`：管控每次boosting迭代中给到估计器的随机种子。


代码中使用Adaboost：

```python
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators = 50, random_state = 0)
clf.fit(train_x, train_y)
result = clf.predict_proba(test_x)
```

成绩

![](./week_report_6_images/adaboost_score.png)

Adaboost是目前应用的准确率最高的方法（不考虑调参的话）。

[Adaboost代码](../code/competition_1_tianchi/week_6_adaboost)

### 2. 可改进之处

- 对于每种模型，可进行模型调参。试验多种参数组合以获得最好效果。
- 可尝试更多机器学习方法。如神经网络。