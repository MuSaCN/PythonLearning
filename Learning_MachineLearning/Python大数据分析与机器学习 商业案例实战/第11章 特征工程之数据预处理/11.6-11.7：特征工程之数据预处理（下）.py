# # 第十一章 特征工程之数据预处理（11.6至11.7）
# # 11.6 多重共线性的分析与处理
import pandas as pd
df = pd.read_excel('数据.xlsx')
df.head()

X = df.drop(columns='Y')
Y = df['Y']

# 1.相关系数判断
print(X.corr())

# 2.方差膨胀因子法（VIF检验）
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X.values, X.columns.get_loc(i)) for i in X.columns]
print(vif)

vif = []
for i in X.columns:  # i对应的是每一列的列名
    vif.append(variance_inflation_factor(X.values, X.columns.get_loc(i)))
print(vif)

X = df[['X1', 'X3']]
Y = df['Y']

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X.values, X.columns.get_loc(i)) for i in X.columns]
print(vif)

# # 11.7 过采样和欠采样
# 11.7.1 过采样
import pandas as pd
data = pd.read_excel("信用卡数据.xlsx")
data.head()

X = data.drop(columns='分类')
y = data['分类']

from collections import Counter
Counter(y)

# （1）随机过采样
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_oversampled, y_oversampled = ros.fit_resample(X, y)

print(Counter(y_oversampled))

print(X_oversampled.shape)

# （2）SMOTE过采样
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0)
X_smotesampled, y_smotesampled = smote.fit_resample(X, y)

print(Counter(y_smotesampled))

# 11.7.2 欠采样
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_undersampled, y_undersampled = rus.fit_resample(X, y)

print(Counter(y_undersampled))

print(X_undersampled.shape)
