# Author:Zhang Yuan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels as sm
import MyPackage

__mypath__ = MyPackage.MyClass_Path.MyClass_Path("\\利用Python进行数据分析")  #路径类
myfile = MyPackage.MyClass_File.MyClass_File()  #文件操作类
myplt = MyPackage.MyClass_Plot.MyClass_Plot()  #直接绘图类(单个图窗)
myfig = MyPackage.MyClass_Plot.MyClass_Figure()  #对象式绘图类(可多个图窗)
myfigpro = MyPackage.MyClass_PlotPro.MyClass_FigurePro()  #高级对象式绘图类
mynp = MyPackage.MyClass_Array.MyClass_NumPy()  #多维数组类(整合Numpy)
mypd = MyPackage.MyClass_Array.MyClass_Pandas()  #矩阵数组类(整合Pandas)
mypdpro = MyPackage.MyClass_ArrayPro.MyClass_PandasPro()  #高级矩阵数组类
mytime = MyPackage.MyClass_Time.MyClass_Time()  #时间类
#---------------------------------------------------------
path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\利用Python进行数据分析(第二版)代码\\"

#%%
import json
path = path+'datasets\\bitly_usagov\\example.txt'
records = [json.loads(line) for line in open(path)]


### Counting Time Zones with pandas
#%%
frame = pd.DataFrame(records)
frame.info()
frame['tz'][:10]
#%%
tz_counts = frame['tz'].value_counts()
tz_counts[:10]
#%%
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()
tz_counts[:10]
subset = tz_counts[:10]

myfigpro.bar_distribution(x=subset.values,y=subset.index)



#%%
frame['a'][1]
frame['a'][50]
frame['a'][51][:50]  # long line
#%%
results = pd.Series([x.split()[0] for x in frame.a.dropna()])
results[:5]
results.value_counts()[:8]
#%%
cframe = frame[frame.a.notnull()]
#%%
cframe = cframe.copy()
#%%
cframe['os'] = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
cframe['os'][:5]
#%%
by_tz_os = cframe.groupby(['tz', 'os'])
#%%
agg_counts = by_tz_os.size().unstack().fillna(0)
agg_counts[:10]
#%%
# Use to sort in ascending order
indexer = agg_counts.sum(1).argsort()
indexer[:10]
#%%
count_subset = agg_counts.take(indexer[-10:])
count_subset
#%%
agg_counts.sum(1).nlargest(10)
#%%
plt.figure()
#%%
# Rearrange the data for plotting
count_subset = count_subset.stack()
count_subset.name = 'total'
count_subset = count_subset.reset_index()
count_subset[:10]
sns.barplot(x='total', y='tz', hue='os',  data=count_subset)
#%%
def norm_total(group):
    group['normed_total'] = group.total / group.total.sum()
    return group

results = count_subset.groupby('tz').apply(norm_total)
#%%
plt.figure()
#%%
sns.barplot(x='normed_total', y='tz', hue='os',  data=results)
#%%
g = count_subset.groupby('tz')
results2 = count_subset.total / g.total.transform('sum')
#%% md
plt.show()






























