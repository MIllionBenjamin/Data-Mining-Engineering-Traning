import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

data_train =pd.read_csv('../train.csv')
data_test_a = pd.read_csv('../testA.csv')

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

#按照中位数填充数值型特征
data_train[numerical_fea] = data_train[numerical_fea].fillna(data_train[numerical_fea].median())
data_test_a[numerical_fea] = data_test_a[numerical_fea].fillna(data_train[numerical_fea].median())
#按照众数填充类别型特征
data_train[category_fea] = data_train[category_fea].fillna(data_train[category_fea].mode())
data_test_a[category_fea] = data_test_a[category_fea].fillna(data_train[category_fea].mode())

print(data_train[numerical_fea])
print(data_train.isnull().sum())

train_x = np.asarray(data_train[numerical_fea])
train_y = np.asarray(data_train[label])

test_x = np.asarray(data_test_a[numerical_fea])
#test_y = np.asarray(data_test_a[label])

clf = LogisticRegression()
print(train_x)
print(train_y)
clf.fit(train_x, train_y)

#result = clf.predict_proba(test_x)
result = clf.predict_proba(test_x)
print(result)

#result_dataframe = pd.DataFrame(data = {"id": data_test_a["id"], "isDefault": [i[1] for i in list(result)]})
result_dataframe = pd.DataFrame(data = {"id": data_test_a["id"], "isDefault": ["{:f}".format(i[1]) for i in list(result)]})
print(result_dataframe)

result_dataframe.to_csv("result_logistic.csv", index=False)

