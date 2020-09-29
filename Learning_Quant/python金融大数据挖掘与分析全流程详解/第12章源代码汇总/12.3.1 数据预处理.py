# =============================================================================
# 12.3.1 数据预处理 by 王宇韬 & 房宇亮
# =============================================================================

import pandas as pd
import datetime

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

print(df)
print('数据预处理完毕！')
