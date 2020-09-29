# =============================================================================
# 12.2.1 重复值及缺失值处理 by 王宇韬 & 房宇亮
# =============================================================================

# 1.重复值处理
# 1.1 先创建一个DataFrame
import pandas as pd
data = pd.DataFrame([[1, 2, 3], [1, 2, 3], [4, 5, 6]], columns=['c1', 'c2', 'c3'])
print(data)

# 1.2 显示重复的行，其中通过duplicated()函数来查询重复的内容
print(data[data.duplicated()])

# 1.3 通过sum()函数统计重复行的数量
print(data.duplicated().sum())

# 1.4 通过drop_duplicates()函数删除重复行
a = data.drop_duplicates()
print(a)

# 1.5 按列进行去重
b = data.drop_duplicates('c1')
print(b)


# 2.缺失值处理
# 2.1 这里先构造一个含有缺失值的DataFrame，代码如下：
import numpy as np
data = pd.DataFrame([[1, 2, 3], [np.nan, 2, np.nan], [np.nan, np.nan, np.nan]], columns=['c1', 'c2', 'c3'])
print(data)

# 2.2 用isnull()函数或isna()函数（两者作用类似）来查看空值
print(data.isnull())  # 或者写data.isna()

# 2.3 对单列查看缺失值
print(data['c1'].isnull())

# 2.4 查看空值行
print(data[data['c1'].isnull()])

# 2.5 空值处理方式1 - 删除空值
a = data.dropna()  # ，可以设置thresh参数，比如将其设置为n，其含义是如果该行的非空值少于n个，则删除该行，写法为data.dropna(thresh=5)
print(a)

# 2.6 空值处理方式2 - 填充空值
b = data.fillna(data.mean())
print(b)

