# =============================================================================
# 6.2.1 二维数据表格DataFrame的创建 by 王宇韬&肖金鑫
# =============================================================================

# 1.通过Pandas创建二维数组 - 列表法
import pandas as pd
a = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
print(a)

# 1.1 添加列索引和行索引，columns代表列，index代表行
a = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['date', 'score'], index=['A', 'B', 'C'])
print(a)

# 1.2 也是一种列表创建DataFrame的方法，在第十二章12.3小节判别券商分析师分析准确度的时候便用应用
a = pd.DataFrame()  # 创建一个空DataFrame
date = [1, 3, 5]
score = [2, 4, 6]
a['date'] = date
a['score'] = score
print(a)

# 2.通过Pandas创建二维数组 - 字典法
# 2.1 此时字典键为列索引
b = pd.DataFrame({'a': [1, 3, 5], 'b': [2, 4, 6]}, index=['x', 'y', 'z'])
print(b)

# 2.2 将字典键变成行索引
c = pd.DataFrame.from_dict({'a': [1, 3, 5], 'b': [2, 4, 6]}, orient="index")
print(c)

# 3.通过Pandas创建二维数组 - 二维数组法
import numpy as np
d = pd.DataFrame(np.arange(12).reshape(3,4), index=[1, 2, 3], columns=['A', 'B', 'C', 'D'])
print(d)

# 4.修改索引
# 4.1 添加行索引那一列的名称
a = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['date', 'score'], index=['A', 'B', 'C'])
a.index.name = '公司'
print(a)

# 4.2 对索引重命名
a = a.rename(index={'A': '万科', 'B': '阿里', 'C': '百度'}, columns={'date': '日期','score': '分数'})
# 或者直接写a.rename(index={'A':'万科', 'B':'阿里', 'C':'百度'}, columns={'date':'日期','score':'分数'}, inplace=True)
print(a)

# 4.3 重置索引，把索引还变成数字索引格式
a = a.reset_index()
print(a)

# 4.4 把常规列设置为行索引
a = a.set_index('日期')
print(a)



