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
__mypath__ = MyPackage.MyClass_Path.MyClass_Path("\\test")  #路径类
myfile = MyPackage.MyClass_File.MyClass_File()  #文件操作类
myplt = MyPackage.MyClass_Plot.MyClass_Plot()  #直接绘图类(单个图窗)
myfig = MyPackage.MyClass_Plot.MyClass_Figure(AddFigure=False)  #对象式绘图类(可多个图窗)
mypltpro = MyPackage.MyClass_PlotPro.MyClass_PlotPro()  #Plot高级图系列
myfigpro = MyPackage.MyClass_PlotPro.MyClass_FigurePro(AddFigure=False)  #Figure高级图系列
mynp = MyPackage.MyClass_Array.MyClass_NumPy()  #多维数组类(整合Numpy)
mypd = MyPackage.MyClass_Array.MyClass_Pandas()  #矩阵数组类(整合Pandas)
mypdpro = MyPackage.MyClass_ArrayPro.MyClass_PandasPro()  #高级矩阵数组类
mytime = MyPackage.MyClass_Time.MyClass_Time()  #时间类
myDA = MyPackage.MyClass_DataAnalysis.MyClass_DataAnalysis()  #数据分析类
myBT = MyPackage.MyClass_BackTest.MyClass_BackTest()  #回测类
myWebQD = MyPackage.MyClass_WebCrawler.MyClass_WebQuotesDownload()  #金融行情下载类
#------------------------------------------------------------

# ---获得数据
Path = "C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\033"
CJSecurities = pd.read_csv(Path + '\\CJSecurities.csv', index_col=1, parse_dates=True)
CJSecurities = CJSecurities.iloc[:, 1:]
data = CJSecurities["2015"]

# ---基础设置
myBT = MyPackage.MyClass_BackTest.MyClass_BackTest()  #回测类
myBT.ValueCash(9999)
myBT.AddBarsData(data,fromdate=None,todate=None)

# ---策略函数
@myBT.OnInit
def __init__():
    print(myBT.close(0), "init")  # 此时没有递归，表示数据的最后一个

# ---增加策略1
@myBT.OnNext
def next():
    print("当前0", myBT.close(0), " 过去1", myBT.close(1))
myBT.addstrategy()

# ---增加策略2
@myBT.OnNext
def next():
    print("当前0", myBT.close(0), " 过去2", myBT.close(2))
myBT.addstrategy()

# ---运行
myBT.run(plot = True)




