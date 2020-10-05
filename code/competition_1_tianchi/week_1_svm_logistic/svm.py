import pandas as pd
import numpy as np
from sklearn import svm

data_train =pd.read_csv('../train.csv')
data_test_a = pd.read_csv('../testA.csv')

numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea,list(data_train.columns)))
label = 'isDefault'
numerical_fea.remove(label)
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

clf = svm.SVC(gamma='scale', probability = True)
print(train_x)
print(train_y)
clf.fit(train_x[0: 30000], train_y[0: 30000])

result = clf.predict_proba(test_x)
print(result)

result_dataframe = pd.DataFrame(data = {"id": data_test_a["id"], "isDefault": [i[1] for i in list(result)]})
print(result_dataframe)

result_dataframe.to_csv("result.csv", index=False)

