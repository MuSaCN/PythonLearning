# # 11.2 重复值、缺失值及异常值处理
# 11.2.1 重复值处理
import pandas as pd
data = pd.DataFrame([[1, 2, 3], [1, 2, 3], [4, 5, 6]], columns=['c1', 'c2', 'c3'])

print(data)

print(data[data.duplicated()])

print(data.duplicated().sum())

data = data.drop_duplicates()

print(data)

data = pd.DataFrame([[1, 2, 3], [1, 2, 3], [4, 5, 6]], columns=['c1', 'c2', 'c3'])
data = data.drop_duplicates('c1')
print(data)

# 11.2.2 缺失值处理
import numpy as np
data = pd.DataFrame([[1, np.nan, 3], [np.nan, 2, np.nan], [1, np.nan, 0]], columns=['c1', 'c2', 'c3'])
print(data)

data.isnull()

data['c1'].isnull()

data[data['c1'].isnull()]

a = data.dropna()
print(a)

a = data.dropna(thresh=2)
print(a)

b = data.fillna(data.mean())
print(b)

c = data.fillna(method='pad')
print(c)

d = data.fillna(method='backfill')
e = data.fillna(method='bfill')
print(e)

# 11.2.3 异常值处理
data = pd.DataFrame({'c1': [3, 10, 5, 7, 1, 9, 69], 'c2': [15, 16, 14, 100, 19, 11, 8], 'c3': [20, 15, 18, 21, 120, 27, 29]}, columns=['c1', 'c2', 'c3'])
print(data)

# 1.利用箱型图观察
data.boxplot()  # 画箱型图

# 2.利用标准差检测
a = pd.DataFrame()
for i in data.columns:
    z = (data[i] - data[i].mean()) / data[i].std()
    a[i] = abs(z) > 2
print(a)

# # 11.3 数据标准化
import pandas as pd
X = pd.DataFrame({'酒精含量(%)': [50, 60, 40, 80, 90], '苹果酸含量(%)': [2, 1, 1, 3, 2]})
y = [0, 0, 0, 1, 1]
print(X) # 查看X

# 11.3.1 min-max标准化
from sklearn.preprocessing import MinMaxScaler
X_new = MinMaxScaler().fit_transform(X)
print(X_new)  # 查看X_new

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=123)

# 11.3.2 Z-score标准化
from sklearn.preprocessing import StandardScaler
X_new = StandardScaler().fit_transform(X)
print(X_new)  # 查看X_new

# # 11.4 数据分箱
import pandas as pd
data = pd.DataFrame([[22,1],[25,1],[20,0],[35,0],[32,1],[38,0],[50,0],[46,1]], columns=['年龄', '是否违约'])
print(data)

data_cut = pd.cut(data['年龄'], 3)
print(data_cut)

data['年龄'].groupby(data_cut).count()

print(pd.cut(data['年龄'], 3, labels=[1, 2, 3]))
