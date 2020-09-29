# # 第十一章 特征工程之数据预处理（11.5）
# # 11.5 特征筛选：WOE值与IV值
# 11.5.3 WOE值与IV值的代码实现
# 1.数据分箱
import pandas as pd
data = pd.DataFrame([[22,1],[25,1],[20,0],[35,0],[32,1],[38,0],[50,0],[46,1]], columns=['年龄', '是否违约'])
print(data)

data_cut = pd.cut(data['年龄'], 3)
print(data_cut)

# 2.统计各个分箱样本总数、坏样本数和好样本数
cut_group_all = data['是否违约'].groupby(data_cut).count()
cut_y = data['是否违约'].groupby(data_cut).sum()
cut_n = cut_group_all - cut_y

print(cut_group_all)

df = pd.DataFrame()  # 创建一个空DataFrame用来汇总数据
df['总数'] = cut_group_all
df['坏样本'] = cut_y
df['好样本'] = cut_n
print(df)

# 3.统计各分箱中坏样本比率和好样本比率
df['坏样本%'] = df['坏样本'] / df['坏样本'].sum()
df['好样本%'] = df['好样本'] / df['好样本'].sum()
print(df)

# 4.计算WOE值
import numpy as np
df['WOE'] = np.log(df['坏样本%'] / df['好样本%'])
print(df)

df = df.replace({'WOE': {np.inf: 0, -np.inf: 0}})

# 5.计算IV值
df['IV'] = df['WOE'] * (df['坏样本%'] - df['好样本%'])
print(df)

iv = df['IV'].sum()
print(iv)

# 1.构造数据
import pandas as pd
data = pd.DataFrame([[22,1],[25,1],[20,0],[35,0],[32,1],[38,0],[50,0],[46,1]], columns=['年龄', '是否违约'])

# 2.数据分箱
data_cut = pd.cut(data['年龄'], 3)

# 3.统计各个分箱样本总数、坏样本数和好样本数并汇总数据
cut_group_all = data['是否违约'].groupby(data_cut).count()
cut_y = data['是否违约'].groupby(data_cut).sum()
cut_n = cut_group_all - cut_y
df = pd.DataFrame()  # 创建一个空DataFrame用来汇总数据
df['总数'] = cut_group_all
df['坏样本'] = cut_y
df['好样本'] = cut_n

# 4.统计坏样本%和好样本%
df['坏样本%'] = df['坏样本'] / df['坏样本'].sum()
df['好样本%'] = df['好样本'] / df['好样本'].sum()

# 5.计算WOE值
import numpy as np
df['WOE'] = np.log(df['坏样本%'] / df['好样本%'])
df = df.replace({'WOE': {np.inf: 0, -np.inf: 0}})  # 替换可能存在的无穷大

# 6.计算各个分箱的IV值
df['IV'] = df['WOE'] * (df['坏样本%'] - df['好样本%'])

# 7.汇总各个分箱的IV值，获得特征变量的IV值
iv = df['IV'].sum()
print(iv)
