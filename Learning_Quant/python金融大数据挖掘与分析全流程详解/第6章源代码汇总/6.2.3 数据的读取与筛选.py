# =============================================================================
# 6.2.3 数据读取与筛选 by 王宇韬&肖金鑫
# =============================================================================

# 先创建一个DataFrame为之后做准备
import pandas as pd
data = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['r1', 'r2', 'r3'], columns=['c1', 'c2', 'c3'])
# 另一种创建的方法：
# import numpy as np
# data = pd.DataFrame(np.arange(1,10).reshape(3,3), index=['r1', 'r2', 'r3'], columns=['c1', 'c2', 'c3'])
print(data)

# 1 按照列来选取数据
# 1.1 下面的方法获得是一个一维序列
a = data['c1']
print(a)

# 1.2 如果想在上面的基础上获得一个二维表格，再加一个中括号
b = data[['c1']]
print(b)

# 1.3 选取多列
c = data[['c1', 'c3']]
print(c)

# 2 按照行来选取数据
# 2.1 传统写法
# 选取第2到3行的数据，注意序号从0开始，左闭右开
a = data[1:3]
print(a)

# 2.2 pandas官方推荐的iloc写法，根据行索引的序号来选取
b = data.iloc[1:3]
print(b)
# 如果要选取单行的话，就得用iloc了
c = data.iloc[-1]
print(c)

# 2.3 pandas官方推荐的loc写法，根据行索引的名称来选取
d = data.loc[['r2', 'r3']]
print(d)

# 2.4 选取前几行或者后几行的简便写法
e = data.head()  # 选取前5行的快捷写法，不足5行，则全部选取
print(e)
f = data.head(10)  # 选取前10行的快捷写法，不足10行，则全部选取
print(f)
g = data.tail()  # 选取后5行的快捷写法，不足5行，则全部选取
print(g)
h = data.tail(2)  # 选取后2行的快捷写法
print(h)

# 3 按照区块来选取数据
# 3.1 常规写法
a = data[['c1', 'c3']][0:2]  # 也可写成data[0:2][['c1', 'c3']]
print(a)

# 3.2 pandas官方推荐写法:iloc加列选取结合的方法了
b = data.iloc[0:2][['c1', 'c3']]
print(b)

# 3.3 选取单个值，就更推荐用iloc加列选取结合的方法了
c = data.iloc[0]['c3']  # 选取c3列的第一行的值
print(c)

# 3.4 也可以直接利用loc和iloc同时选取行和列
d = data.loc[['r1', 'r2'], ['c1', 'c3']]
e = data.iloc[0:2, [0, 2]]
print(d)
print(e)
'''注意loc方法使用字符串作为索引选择行和列，iloc方法使用数值作为索引选择行和列。有个记忆的方法，loc是location（定位、位置）的缩写，
所以通过字符索引来定位，而iloc中多了一个字母i，而i又经常代表数值，所以iloc方法是用数值作为索引'''

# 3.5 通过ix方法同时选取行和列，目前已经不被pandas官方推荐了，推荐使用data.iloc[0:2][['c1', 'c3']]的格式
f = data.ix[0:2, ['c1', 'c3']]
print(f)

# 4 数据运算
# 从已有的列中，通过数据运算创造一个新的一列
data['c4'] = data['c3'] - data['c1']
a = data.head()
print(a)

# 5 数据筛选
# 在方括号里通过判断条件来过滤行，比如选取c1列数字大于1的行
a = data[data['c1'] > 1]
print(a)
# 多个筛选条件
b = data[(data['c1'] > 1) & (data['c2'] == 5)]
print(b)

# 6 数据排序
# 通过sort_values()可以根据列对数据进行排序，比如要对c2列进行降序排序
a = data.sort_values(by='c2', ascending=False)  # 参数ascending为上升的意思，默认参数为True，设置为False的话，则表示降序排序
print(a)
# 通过sort_index()则可以根据行索引进行排序，比如按行索引进行升序排列
a = a.sort_index()
print(a)

# 7 数据删除
# 删除c1列的数据
a = data.drop(columns='c1')
print(a)
# 删除多列的数据，比如c1和c3列，可以通过列表的方式声明
b = data.drop(columns=['c1', 'c3'])
print(b)
# 如果要删除行数据，比如删去第一行和第三行的数据
c = data.drop(index=['r1', 'r3'])
print(c)
# 上面删除数据后又赋值给新的变量不会改变原来表格data的结构，如果想改变原来表格的结构，可以令inplace参数为True
# data.drop(index=['r1', 'r3'], inplace=True)
# print(data)



