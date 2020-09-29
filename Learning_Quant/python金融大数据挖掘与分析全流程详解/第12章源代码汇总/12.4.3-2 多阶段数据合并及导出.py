# =============================================================================
# 12.4.3-2 多阶段数据合及导出 by 王宇韬 & 房宇亮
# =============================================================================

import pandas as pd

# 1. 读取数据
a = pd.read_excel('10天收益率.xlsx')
b = pd.read_excel('30天收益率.xlsx')
c = pd.read_excel('60天收益率.xlsx')
d = pd.read_excel('90天收益率.xlsx')
e = pd.read_excel('180天收益率.xlsx')

# 2.合并数据
f = pd.merge(a, b)
g = pd.merge(f, c)
h = pd.merge(g, d)
i = pd.merge(h, e)
'''其中merge()函数会默认以共同的列进行拼接，同时选择内（inner）连接，也即取两表共有的内容，这样会导致前几张表格的数据被剔除掉，
因为有的股票还没有180天收益率的数据。如果想取并集的话，也即把所有数据都获取的话，可以在merge()函数中设置how参数为outer。'''

# 3.删除可能出现的重复值
i = i.drop_duplicates()

# 4.导出为Excel
i.to_excel('多阶段收益率.xlsx')

print(i)
print('多阶段收益率完成！')


