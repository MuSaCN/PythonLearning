# Author:Zhang Yuan
import pandas as pd
import numpy as np

#Pandas提供了两大数据结构：一维结构的Series类型、二维结构的DataFrame类型。

#Series对象本质上是Numpy对象，具有index和values两大属性。

#对于输入的valuas，Series会默认位置索引0、1、2、3...，还可以自定义标签索引。

#Series切片支持“标签切片”和“位置切片”。位置切片即Python切片，包括头不包括尾；但“标签切片”包括头包括尾。之所以这样设计是因为，通常我们不知道标签的顺序，无法知道末尾标签下一个标签是什么。

#时间序列Series，在索引和切片方面有优化：
from datetime import datetime
dates=[datetime(2016,1,1),datetime(2016,1,2),datetime(2016,1,3),datetime(2016,2,1)]
ts=pd.Series([1,2,3,4],index=dates)
print(ts["20160101"],ts["2016-01-01"],ts["01/01/2016"]) #时间序列多种字符串索引
print(ts["2016"])                                       #只传入年或年月来切片
print(ts["2016-01"])                                    #只传入年或年月来切片
print(ts["2016-01":"2016-03"])                          #切片字符串时间戳可以不必存于index中

#DataFrame是一个表格型的数据结构，每一列代表一个变量，每一行是一条记录。简单来说，DataFrame是共享同一个index的Series的集合。

#DataFrame对象的索引和切片---------------------------------------------------
dates=[datetime(2016,1,i) for i in range(1,10)]
df=pd.DataFrame(np.random.randn(9,4),index=dates,columns=list("ABCD"))
print(df[0:3])                      #对行切片
print(df["A"])                      #提取单独一列,Series.
print(df[["A","C"]])                #提取多列，DataFrame. df[["A"]]也是dataFrame
print(df[df["A"]>0])                #根据boolean值提取行
#PS注意：对列直接切片出错：df["A":"C"]；直接同时的操作行列也出错：df[1:3,"A"]
#如果要行列操作，需要用方法：标签索引和切片loc[]
print(df.loc[:,"A"])                   #提取一列
print(df.loc[:,"A":"C"])               #列切片
print(df.loc[dates[0:4],"A":"C"])      #行列切片
print(df.loc[dates[0],"A"])            #特定值
print(df.loc[df.loc[:,"A"]>0])         #根据boolean值提取
#如果要行列操作，需要用方法：位置索引和切片iloc[]
print(df.iloc[2])                 #提取行，相当于df.iloc[2,:]
print(df.iloc[:,2])               #提取列
print(df.iloc[[1,4],[2,3]])       #提取多个行列值，不是切片，类似numpy
print(df.iloc[1:5,2:4])           #切片
print(df.iloc[2,3])               #提取特定值
#-----------------------------------------------------------------------------

# ---修改数据时要注意，有时必须以列的形式定位数据再修改才可以修改成功
df["Volume"].iloc[0] = 0
# temp.Volume.iloc[0] = 0
# ---注意，多次 loc 或 iloc 无法修改原数据的数值
# deal_in_copy.loc[price, "Volume"].iloc[0] = 0
# deal_in_copy.loc[price, "Volume"].iloc[0] = 0
# deal_in_copy.loc[price].iloc[0]["Volume"] = 0
# ---注意，数据修改以一整行或一整列，行切片叠加列切片不一定能修改数据


#----------------------------------------------------------------------------
#Series与DataFrame对象的运算
#Series与Series是index匹配运算
s1=pd.Series([1,2,3],index=list("ABC"))
s2=pd.Series([4,5,6],index=list("BCD"))
s2-s1
#DataFrame与Series是DataFrame的column与Series的index匹配，PS:不是index匹配
df1=pd.DataFrame(np.arange(1,13).reshape(3,4),index=list("abc"),columns=list("ABCD"))
df1-s1
#DataFrame与DataFrame是同时对index与column匹配
df2=pd.DataFrame(np.arange(1,13).reshape(4,3),index=list("bcde"),columns=list("CDE"))
df1*df2

#DataFrame的轴axis与numpy一样。0轴-Y轴-列数据、1轴-X轴-行数据。

#Python代码的设计原则之一是“显示优于隐式”，使用loc和iloc可以让代码更容易维护，可读性更高。
#DataFrame的一级访问为列，这与NumPy的二维数据的一级访问不同，后者为行.

#stack堆积/unstack反堆积
#stack表示DataFrame-->Series方向。所以Series无stack()函数，*.stack().stack()...不能无限
#unstack表示Series-->DataFrame方向。当DataFrame的行索引都变成列索引时，系统会自动把列索引转成行索引，索引*.unstack().unstack()...能无限。


# ---Pandas高效率迭代:
### Series 循环效率最高
# --> for items in series.iteritems(): --> 遍历包含索引标签和系列中相应值的元组。
    # 效率不一定高于循环
# --> series 更高效率以 map 或 apply 形式，对 series 每个元素遍历
    # 效率不一定高于循环

### DataFrame: apply > iter > 循环
# --> 用 for i in range(len(df)) --> 低效率
# --> 效率比循环高
    # for i,row in df.iterrows(): --> 按行遍历。可修改df中的值
    # for row in df.itertuples(): --> 按行遍历。不可修改其值，返回对象。
    # for index, column in df.iteritems(): --> 按列遍历
# --> dataframe 更高效率以apply形式，对df每个行或列遍历。
    # 有时候apply会用到索引，索引可以添加到df中。
    # 如果函数f涉及到外部变量赋值，可以用列表。
    #         def f(row):
    #             print(row)
    #             return 1
    #         out = df.apply(f, axis=1) --> 按行遍历，且进行apply
    #         out = df.apply(f, axis=0) --> 按列遍历，且进行apply
# --> dataframe 用 applymap形式，对df每个元素遍历

### dataframe 的 groupby.apply 速度要慢于 df.apply

### rolling的使用
# df.rolling(..) --> 会对df中的每个列的Series进行rolling

### 经过检验，重复列名的df，无法赋值。
df = pd.DataFrame([[1,2],[2,4]])
df = pd.concat((df,df),axis=1)
df.columns = [0,1,1,2]
df
df[0][0] = 100 # 结果无法赋值！

### 小数和整数在一个df中，需要拼接，同时会都转成小数。
