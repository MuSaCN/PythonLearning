# 11.5.4 案例实战：客户流失预警模型的IV值计算
import pandas as pd
import numpy as np
def cal_iv(data, cut_num, feature, target):
    # 1.数据分箱
    data_cut = pd.cut(data[feature], cut_num)
    # 2.统计各个分箱样本总数、坏样本数和好样本数
    cut_group_all = data[target].groupby(data_cut).count()  # 总客户数
    cut_y = data[target].groupby(data_cut).sum()  # 坏样本数
    cut_n = cut_group_all - cut_y  # 好样本数
    df = pd.DataFrame()  # 创建一个空DataFrame用来汇总数据
    df['总数'] = cut_group_all
    df['坏样本'] = cut_y
    df['好样本'] = cut_n
    # 3.统计坏样本%和好样本%
    df['坏样本%'] = df['坏样本'] / df['坏样本'].sum()
    df['好样本%'] = df['好样本'] / df['好样本'].sum()
    # 4.计算WOE值
    df['WOE'] = np.log(df['坏样本%'] / df['好样本%'])
    df = df.replace({'WOE': {np.inf: 0, -np.inf: 0}}) 
    # 5.计算各个分箱的IV值
    df['IV'] = df['WOE'] * (df['坏样本%'] - df['好样本%'])
    # 6.汇总各个分箱的IV值，获得特征变量的IV值
    iv = df['IV'].sum()
    print(iv)

data = pd.read_excel('股票客户流失.xlsx')
data.head()

cal_iv(data, 4, '账户资金（元）', '是否流失')

for i in data.columns[:-1]:
    print(i + '的IV值为：')
    cal_iv(data, 4, i, '是否流失')  # 调用函数
