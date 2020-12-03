# Author:Zhang Yuan整理，版本Pandas0.24.2
# 0. 习惯上，我们会按下面格式引入所需要的包：
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 创建对象 Object Creation---------------------------------------------------------------
# 可以通过 数据结构入门 来查看有关该节内容的详细信息。

# 1.1 可以通过传递一个 list 对象来创建一个 Series ，pandas 会默认创建整型索引：
s = pd.Series([1,3,5,np.nan,6,8])
s

# 1.2 通过传递一个 numpy array ，时间索引以及列标签来创建一个 DataFrame ：
dates = pd.date_range('20130101', periods=6)
dates
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
df

# 1.3 通过传递一个能够被转换成类似序列结构的字典对象来创建一个 DataFrame ：
df2 = pd.DataFrame({ 'A': 1.,
                     'B': pd.Timestamp('20130102'),
                     'C': pd.Series(1, index=list(range(5)), dtype='float32'),
                     'D': np.array([3] * 5, dtype='int32'),
                     'E': pd.Categorical(["test", "train", "test", "train","add"]),
                     'F': 'foo' })
df2

# 1.4 查看不同列的数据类型：
df2.dtypes

# 2. 查看数据Viewing Data---------------------------------------------------

# 2.1 查看 DataFrame 中头部和尾部的行：十分钟搞定 Pandas
df.head()
df.tail(3)

# 2.2 显示索引. 列和底层的 numpy 数据：
df.index
df.columns
df.values

# 2.3 DataFrame.to_numpy（）给出了底层数据的NumPy表示。 请注意，当您的DataFrame具有不同数据类型的列时，他可能是一项昂贵的操作，这归结为pandas和NumPy之间的根本区别：NumPy数组对整个数组有一个dtype，而pandas DataFrames每列有一个dtype。 当您调用DataFrame.to_numpy（）时，pandas将找到可以容纳DataFrame中所有dtypes的NumPy dtype。 这可能最终成为对象，这需要将每个值都转换为Python对象。
df.to_numpy() # 对于df，我们的所有浮点值的DataFrame，DataFrame.to_numpy（）都很快，不需要复制数据。
df2.to_numpy()# 对于df2，具有多个dtypes的DataFrame，DataFrame.to_numpy（）相对昂贵。

# 2.4 describe() 函数对于数据的快速统计汇总：
df.describe()

# 2.5 对数据的转置：
df.T

# 2.6 按轴进行排序:
df.sort_index(axis=1, ascending=False)

# 2.7 按值进行排序:
df.sort_values(by='B')

# 3. 选择Selection-----------------------------------------------------------------
# 虽然标准的 Python/Numpy 的选择和设置表达式都能够直接派上用场，但是作为工程使用的代码，我们推荐使用经过优化的 pandas 数据访问方式： .at , .iat ,.loc , .iloc。

# 3.1 获取Getting
# 3.1.1 选择一个单独的列，这将会返回一个 Series ，等同于 df.A ：
df['A']
# 3.1.2 通过 [] 进行选择，这将会对行进行切片
df[0:3]
df['20130102':'20130104']

# 3.2 通过标签选择Selection by Label
# 3.2.1 使用标签来获取一个交叉的区域
df.loc[dates[0]]
# 3.2.2 通过标签来在多个轴上进行选择
df.loc[:, ['A', 'B']]
# 3.2.3 标签切片
df.loc['20130102':'20130104', ['A', 'B']]
# 3.2.4 对于返回的对象进行维度缩减
df.loc['20130102', ['A', 'B']]
# 3.2.5 获取一个标量
df.loc[dates[0], 'A']
# 3.2.6 快速访问一个标量（与上一个方法等价）
df.at[dates[0], 'A']

# 3.3 通过位置选择Selection by Position
# 3.3.1 通过传递数值进行位置选择（选择的是行）
df.iloc[3]
# 3.3.2 通过数值进行切片，与 numpy/python 中的情况类似
df.iloc[3:5, 0:2]
# 3.3.3 通过指定一个位置的列表，与 numpy/python 中的情况类似
df.iloc[[1, 2, 4], [0, 2]]
# 3.3.4 对行进行切片
df.iloc[1:3, :]
# 3.3.5 对列进行切片
df.iloc[:, 1:3]
# 3.3.6 获取特定的值
df.iloc[1, 1]
# 3.3.7 快速访问标量（等同于前一个方法）：
df.iat[1, 1]

# 3.4 布尔索引 Boolean Indexing
# 3.4.1 使用一个单独列的值来选择数据：
df[df.A > 0]
# 3.4.2 使用 where 操作来选择数据：
df[df > 0]
# 3.4.3 使用 isin() 方法来过滤：
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
df2
df2[df2['E'].isin(['two', 'four'])]

# 3.5 设置Setting
# 3.5.1 设置一个新的列：
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))
s1
# 3.5.2 通过标签设置新的值：
df.at[dates[0], 'A'] = 0
# 3.5.3 通过位置设置新的值：
df.iat[0, 1] = 0
# 3.5.4 通过一个numpy数组设置一组新值：
df.loc[:, 'D'] = np.array([5] * len(df))
df
# 3.5.5 通过where操作来设置新的值：
df2 = df.copy()
df2[df2 > 0] = -df2
df2

# 4. 缺失值处理Missing Data---------------------------------------------------------------
# 在 pandas 中，使用 np.nan 来代替缺失值，这些值将默认不会包含在计算中，详情请参阅：缺失的数据。

# 4.1 reindex() 方法可以对指定轴上的索引进行改变/增加/删除操作，这将返回原始数据的一个拷贝：
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1
df1

# 4.2 去掉包含缺失值的行：
df1.dropna(how='any')

# 4.3 对缺失值进行填充：
df1.fillna(value=5)

# 4.4 对数据进行布尔填充：
pd.isna(df1)

# 5. 相关操作Operations------------------------------------------------------------------
# 详情请参与 基本的二进制操作

# 5.1 统计（相关操作通常情况下不包括缺失值）.
# 5.1.1 执行描述性统计：
df.mean()
# 5.1.2 在其他轴上进行相同的操作：
df.mean(1)
# 5.1.3 对于拥有不同维度，需要对齐的对象进行操作。Pandas 会自动的沿着指定的维度进行广播：
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
df.sub(s, axis='index')

# 5.2 应用Apply
# 对数据应用函数：
df.apply(np.cumsum)
df.apply(lambda x: x.max() - x.min())

# 5.3直方图
s = pd.Series(np.random.randint(0, 7, size=10))
s
s.value_counts()

# 5.4字符串方法
# Series 对象在其 str 属性中配备了一组字符串处理方法，可以很容易的应用到数组中的每个元素，如下段代码所示。更多详情请参考：字符串向量化方法。
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()

# 6. 合并Merge---------------------------------------------------------------------------

# 6.1 Concat
# Pandas 提供了大量的方法能够轻松的对 Series ， DataFrame 和 Panel 对象进行各种符合各种逻辑关系的合并操作。具体请参阅：合并。
# 使用concat()连接pandas对象：
df = pd.DataFrame(np.random.randn(10, 4))
df
pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)

# 6.2 Join
# 类似于 SQL 类型的合并，具体请参阅：数据库风格的连接
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
left
right
pd.merge(left, right, on='key')
# 另一个例子：
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
left
right
pd.merge(left, right, on='key')

# 6.3Append
# 将一行连接到一个 DataFrame 上，具体请参阅附加：
df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
df
s = df.iloc[3]
df.append(s, ignore_index=True)

# 7. 分组Grouping------------------------------------------------------------------------------
# 对于”group by”操作，我们通常是指以下一个或多个操作步骤：
# （Splitting）按照一些规则将数据分为不同的组；
# （Applying）对于每组数据分别执行一个函数；
# （Combining）将结果组合到一个数据结构中；
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                          'foo', 'bar', 'foo', 'foo'],
                    'B': ['one', 'one', 'two', 'three',
                          'two', 'two', 'one', 'three'],
                    'C': np.random.randn(8),
                    'D': np.random.randn(8)})
df
# 7.1 分组并对每个分组执行 sum 函数：
df.groupby('A').sum()
# 7.2 通过多个列进行分组形成一个层次索引，然后执行函数：
df.groupby(['A', 'B']).sum()

# 8. 改变形状Reshaping-------------------------------------------------------------------------
# 详情请参阅 层次索引 和 改变形状。

# 8.1 栈方法 Stack ，二维数据与多索引的一维数据之间转变
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz','foo', 'foo', 'qux', 'qux'],
                     ['one', 'two', 'one', 'two','one', 'two', 'one', 'two'],
                     ['a', 'b', 'a', 'b','a', 'b', 'a', 'b']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second',"third"]) #三重索引
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4]
df2
# 8.1.1 stack()方法“压缩”DataFrame列中的级别（具有MultiIndex作为索引）。
stacked = df2.stack() #列转成索引，变成更多index的一维数据
stacked
# 8.1.2 使用“stacked”DataFrame或Series（具有MultiIndex作为索引），stack()的反向操作是unstack()，默认情况下取消堆栈最后一级：
stacked.unstack()   #默认最后一级索引变成列
stacked.unstack(1)  #第二层变成列
stacked.unstack(0)  #第一层变成列

# 8.2 数据透视表 Pivot Tables (其实就是把内容作为索引来展示二维数据)
df = pd.DataFrame({ 'A': ['one', 'one', 'two', 'three'] * 3,
                    'B': ['AA', 'BB', 'CC'] * 4,
                    'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                    'D': np.random.randn(12),
                    'E': np.random.randn(12)})
df
# 可以从这个数据中轻松的生成数据透视表：
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])

# 9. 时间序列--------------------------------------------------------------------------------
# Pandas 在对频率转换进行重新采样时拥有简单、强大且高效的功能（如将按秒采样的数据转换为按5分钟为单位进行采样的数据）。这种操作在金融领域非常常见。具体参考：时间序列。
rng = pd.date_range('1/1/2012', periods=100, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.resample('5Min').sum() #转成5min数据

# 9.1 时区表示：
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
ts
ts_utc = ts.tz_localize('UTC')
ts_utc

# 9.2 时区转换：
ts_utc.tz_convert('US/Eastern')

# 9.3 时间跨度转换：
rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts
ps = ts.to_period()
ps
ps.to_timestamp()

# 9.4 时期和时间戳之间的转换使得可以使用一些方便的算术函数。
prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
ts.head()

# 10. 分类 Categorical
# pandas 可以在 DataFrame 中支持 Categorical 类型的数据
df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                   "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})

# 10.1 将原始的 grade 转换为 Categorical 数据类型：
df["grade"] = df["raw_grade"].astype("category")
df["grade"]

# 10.2 将 Categorical 类型数据重命名为更有意义的名称：
df["grade"].cat.categories = ["very good", "good", "very bad"]

# 10.3 对类别进行重新排序，增加缺失的类别：
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium",
                                              "good", "very good"])
df["grade"]

# 10.4 排序是按照 Categorical 的顺序进行的而不是按照字典顺序进行：
df.sort_values(by="grade")

# 10.5 对 Categorical 列进行排序，也存在空的类别：
df.groupby("grade").size()

# 11. 画图 Plotting----------------------------------------------------------
ts = pd.Series(np.random.randn(1000),index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot();plt.show()
# 11.1 对于 DataFrame 来说， plot 是一种将所有列及其标签进行绘制的简便方法：
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
plt.figure()
df.plot()
plt.legend(loc='best');plt.show()

# 12. 导入和保存数据 Getting Data In/Out----------------------------------------
import os
# 目录引导结果在Console和run结果不同
print("os.getcwd():",os.getcwd())
print("abspath:", os.path.abspath("11.10MinutesToPandas0.24.2.py"))

# 12.1.1 写入 csv 文件：
df.to_csv("foo.csv")
# 12.1.2 从 csv 文件中读取：
pd.read_csv("foo.csv")

# 12.2 HDF5
# 12.2.1 写入 HDF5 存储：
df.to_hdf('foo.h5', 'df')
# 12.2.2 从 HDF5 存储中读取：
pd.read_hdf('foo.h5', 'df')

# 12.3 Excel
# 12.3.1 写入excel文件：
df.to_excel('foo.xlsx', sheet_name='Sheet1')
# 12.3.2 从excel文件中读取：
pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])

# 13. 陷阱 Gotchas--------------------------------------------------------
# 如果你尝试某个操作并且看到如下异常：(不能以Series化格式进行bool判定)
# if pd.Series([False, True, False]):
#      print("I was true")

