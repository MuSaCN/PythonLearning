# =============================================================================
# 12.3.3 计算平均收益率并进行分析师预测准确度排名 by 王宇韬 & 房宇亮
# =============================================================================

import pandas as pd
import tushare as ts
import datetime
import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息，警告不是报错，不会影响程序执行

# 1. 数据预处理
# 读取数据、删除重复及空值行
df = pd.read_excel('分析师评级报告.xlsx', dtype={'股票代码': str}) # 注意设置dtype参数，让股票代码以字符串格式读进来
df = df.drop_duplicates()  # 删除重复行
df = df.dropna(thresh=5)  # 删除空值行，thresh=5表示非空值少于5个则删除，本案例因为没有空值行，其实可以不写

# 列拼接、选取合适的列
df['研究机构-分析师'] = df['研究机构'] + '-' + df['分析师']
columns = ['股票名称', '股票代码', '研究机构-分析师', '最新评级', '评级调整', '报告日期']
df = df[columns]

# 日期筛选
today = datetime.datetime.now()
t = today - datetime.timedelta(days=30)
t = t.strftime('%Y-%m-%d')
df = df[df["报告日期"] < t]


# 2.通过Tushare库计算股票收益率
df_use = df.iloc[0:100]  # 为了演示选取了前100行，想运行全部可写df_use = df
rate = []  # 创建一个空列表，用来存储每支股票的收益率

for i, row in df_use.iterrows():
    code = row['股票代码']
    analysist_date = row['报告日期']

    # 1.获取开始日期，也即第二天
    begin_date = datetime.datetime.strptime(analysist_date, '%Y-%m-%d')
    begin_date = begin_date + datetime.timedelta(days=1)
    begin_date = begin_date.strftime('%Y-%m-%d')

    # 2.获取结束日期，也即第三十天
    end_date = datetime.datetime.strptime(analysist_date, '%Y-%m-%d')
    end_date = end_date + datetime.timedelta(days=30)
    end_date = end_date.strftime('%Y-%m-%d')

    # 3.通过Tushare库计算股票收益率
    ts_result = ts.get_hist_data(code, begin_date, end_date)
    if ts_result is None or len(ts_result) < 5:  # 防止股票没有数据
        return_rate = 0
    else:
        start_price = ts_result.iloc[-1]['open']
        end_price = ts_result.iloc[0]['close']
        return_rate = (end_price / start_price) - 1.0
    rate.append(return_rate)

df_use['30天收益率'] = rate  # 该添加列的方式参考6.2.1小节


# 3. 计算平均收益率并进行分析师预测准确度排名
# 3.1 计算平均收益率
means = df_use.groupby('研究机构-分析师')[['30天收益率']].mean()

# 3.2 统计预测次数, 记得这里要将列索引重命名
count = df_use.groupby('研究机构-分析师')[['30天收益率']].count()
count = count.rename(columns={'30天收益率': '预测次数'})

# 3.3 合并表格
# 方法1-通过merge()函数合并，注意要设置left_index和right_index按行索引合并
df_final = pd.merge(means, count, left_index=True, right_index=True)
# 方法2-通过concat()函数合并，注意设置axis参数为1使它为横向合并
# df_final = pd.concat([means, count], axis=1)

# 3.4 按照收益率从高往低排序
df_final.sort_values(by='30天收益率', ascending=False, inplace=True)

# 3.5 导出为Excel
df_final.to_excel("分析师预测结果.xlsx")

print(df_final)
print('分析师预测结果排名完成！')


