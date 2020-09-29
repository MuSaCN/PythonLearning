# =============================================================================
# 6.2.4 数据表拼接 by 王宇韬&肖金鑫
# =============================================================================

# 假设有如下两个DataFrame表格，需要对它们进行合并
import pandas as pd
df1 = pd.DataFrame({'公司': ['万科', '阿里', '百度'], '分数': [90, 95, 85]})
df2 = pd.DataFrame({'公司': ['万科', '阿里', '京东'], '股价': [20, 180, 30]})
print(df1)
print(df2)

# 1 merge函数
# 1.1 最简单的用法，直接选取相同的列名（“公司”这一列）进行合并，而且默认选取的是两种表共有的列内容（万科、阿里）
df3 = pd.merge(df1, df2)
print(df3)

# 1.2 默认的合并其实是取交集（inner连接），也即取两表共有的内容，如果想取并集（outer连接），也即选取两表所有的内容，可以设置how参数
df3 = pd.merge(df1, df2, how='outer')
print(df3)

# 1.3 如果想保留左表全部内容，而对右表不太在意的话，可以将how参数设置为left：
df3 = pd.merge(df1, df2, how='left')
print(df3)

# 1.4 如果想根据行索引进行合并，可以通过设置left_index和right_index参数，代码如下：
df3 = pd.merge(df1, df2, left_index=True, right_index=True)
print(df3)

# 2 concat函数
'''concat方法是一种全连接(UNION ALL)方式，它不需要对齐，而是直接进行合并（即它不需要两表的某些列或者索引相同，只是把数据整合到一起）。
所以concat没有"how"和"on"参数，而是通过“axis”指定连接的轴向。'''
# 2.1 默认情况下，axis=0，按行方向进行连接。
df3 = pd.concat([df1, df2])  # 或者写成df3 = pd.concat([df1, df2], axis=0)
print(df3)

# 2.2 如果想按列方向进行连接，可以设置axis参数为1。
df3 = pd.concat([df1, df2], axis=1)
print(df3)

# 3 append函数
'''append函数可以说concat函数的简化版，效果和pd.concat([df1,df2]) 类似'''
# 3.1 常规用法
df3 = df1.append(df2)
print(df3)

# 3.2 append()函数还有个常用的功能，和列表.append()一样，可用来新增元素，代码如下：
df3 = df1.append({'公司': '腾讯', '分数': '90'}, ignore_index=True)
print(df3)
# 该方法在第十四章14.2小节获得股票基本数据及衍生变量的时候便有用到。
