# Author:Zhang Yuan
import MyPackage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

#------------------------------------------------------------
__mypath__ = MyPackage.MyClass_Path.MyClass_Path("\\量化投资以Python为工具")  #路径类
myfile = MyPackage.MyClass_File.MyClass_File()  #文件操作类
myplt = MyPackage.MyClass_Plot.MyClass_Plot()  #直接绘图类(单个图窗)
myfig = MyPackage.MyClass_Plot.MyClass_Figure()  #对象式绘图类(可多个图窗)
mypltpro = MyPackage.MyClass_PlotPro.MyClass_PlotPro()  #Plot高级图系列
myfigpro = MyPackage.MyClass_PlotPro.MyClass_FigurePro()  #Figure高级图系列
mynp = MyPackage.MyClass_Array.MyClass_NumPy()  #多维数组类(整合Numpy)
mypd = MyPackage.MyClass_Array.MyClass_Pandas()  #矩阵数组类(整合Pandas)
mypdpro = MyPackage.MyClass_ArrayPro.MyClass_PandasPro()  #高级矩阵数组类
mytime = MyPackage.MyClass_Time.MyClass_Time()  #时间类
myDA = MyPackage.MyClass_DataAnalysis.MyClass_DataAnalysis()  #数据分析类
#MyPackage.MyClass_ToDefault.DefaultMatplotlibBackend()       #恢复默认设置(仅main主界面)
#------------------------------------------------------------
Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\027"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"


ssec2015 = pd.read_csv(Path+'\\ssec2015.csv')
ssec2015 = myfile.read_pd(Path+'\\ssec2015.csv',index_col="Date",parse_dates=True)
ssec2015 = ssec2015.iloc[:, 1:]
myDA.candle_ohlc(ssec2015)


# morning star
ssec2012 = pd.read_csv(Path+'\\ssec2012.csv')
ssec2012.index = ssec2012.iloc[:, 1]
ssec2012.index = pd.to_datetime(ssec2012.index, format='%Y-%m-%d')
ssec2012 = ssec2012.iloc[:, 2:]
ssec2012.head(2)
ssec2012.iloc[-2:, :]
Close = ssec2012.Close
Open = ssec2012.Open
ClOp = Close - Open
ClOp.head()
ClOp.describe()
Shape = [0, 0, 0]
lag1ClOp = ClOp.shift(1)
lag2ClOp = ClOp.shift(2)

for i in range(3, len(ClOp), 1):
    if all([lag2ClOp[i] < -11, abs(lag1ClOp[i]) < 2, ClOp[i] > 6, abs(ClOp[i]) > abs(lag2ClOp[i] * 0.5)]):
        Shape.append(1)
    else:
        Shape.append(0)

Shape.index(1)

lagOpen = Open.shift(1)
lagClose = Close.shift(1)
lag2Close = Close.shift(2)

Doji = [0, 0, 0]
for i in range(3, len(Open), 1):
    if all([lagOpen[i] < Open[i], lagOpen[i] < lag2Close[i],lagClose[i] < Open[i], (lagClose[i] < lag2Close[i])]):
        Doji.append(1)
    else:
        Doji.append(0)
Doji.count(1)

ret = Close / Close.shift(1) - 1
lag1ret = ret.shift(1)
lag2ret = ret.shift(2)
Trend = [0, 0, 0]
for i in range(3, len(ret)):
    if all([lag1ret[i] < 0, lag2ret[i] < 0]):
        Trend.append(1)
    else:
        Trend.append(0)

StarSig = []
for i in range(len(Trend)):
    if all([Shape[i] == 1, Doji[i] == 1, Trend[i] == 1]):
        StarSig.append(1)
    else:
        StarSig.append(0)

for i in range(len(StarSig)):
    if StarSig[i] == 1:
        print(ssec2012.index[i])

ssec201209 = ssec2012['2012-08-21':'2012-09-30']

# Need to specify path before import
myDA.CandlePlot_ohlc(ssec201209, title=' 上 证 综 指 2012 年9 月 份 的 日 K 线图 ')

# Dark Cloud Cover
# 提 取 读 入 上 证 综 指 年 的 日 交 易 数 据
import pandas as pd

ssec2011 = pd.read_csv(Path+'\\ssec2011.csv')
ssec2011.index = ssec2011.iloc[:, 1]
ssec2011.index = pd.to_datetime(ssec2011.index, format='%Y-%m-%d')
ssec2011 = ssec2011.iloc[:, 2:]

# 提 取 价 格 数 据
Close11 = ssec2011.Close
Open11 = ssec2011.Open

# 刻 画 捕 捉 符 合 “ 乌 云 盖 顶 ” 形 态 的 连 续 两 个 蜡 烛 实 体
lagClose11 = Close11.shift(1)
lagOpen11 = Open11.shift(1)
Cloud = pd.Series(0, index=Close11.index)
for i in range(1, len(Close11)):
    if all([Close11[i] < Open11[i], \
            lagClose11[i] > lagOpen11[i], \
            Open11[i] > lagClose11[i], \
            Close11[i] < 0.5 * (lagClose11[i] + lagOpen11[i]), \
            Close11[i] > lagOpen11[i]]):
        Cloud[i] = 1

# 定 义 前 期 上 升 趋 势
Trend = pd.Series(0, index=Close11.index)
for i in range(2, len(Close11)):
    if Close11[i - 1] > Close11[i - 2] > Close11[i - 3]:
        Trend[i] = 1

darkCloud = Cloud + Trend
darkCloud[darkCloud == 2]

# 绘 制 上 证 综 指 2011 年5月 19 日 附 近 的 K 线图
ssec201105 = ssec2011['2011-05-01':'2011-05-30']
myDA.CandlePlot_ohlc(ssec201105, \
                  title=' 上 证 综 指 2011 年5 月 份 的 日 K 线图 ')

# 绘 制 上 证 综 指 2011 年8月 16 日 附 近 的 K 线图
ssec201108 = ssec2011['2011-08-01':'2011-08-30']
myDA.CandlePlot_ohlc(ssec201108, \
                  title=' 上 证 综 指 2011 年8 月 份 的 日 K 线图 ')








