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
__mypath__ = MyPackage.MyClass_Path.MyClass_Path("\\backtrader_test")  #路径类
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
data0 = CJSecurities
# ---基础设置
myBT = MyPackage.MyClass_BackTest.MyClass_BackTest()  #回测类
myBT.ValueCash(100000)
myBT.AddBarsData(data0,fromdate=None,todate=None)

# ---优化
for j in range(5,20):
    myBT.setPara(j)
    # ---策略开始
    @myBT.OnInit
    def __init__(i):
        print("init", myBT.Self(i) )
        myBT.add_indi_sma(i,0,period=myBT.Para[i][0])
        myBT.Self(i).barscount = 0

    # ---策略递归，next()执行完就进入下一个bar
    @myBT.OnNext
    def next(i):
        if not myBT.position(i):
            if myBT.close(0) > myBT.Self(i).SMA[0]:
                myBT.buy(i)
        else:
            if myBT.bars_executed(i) >= myBT.Self(i).barscount + 5:
                myBT.sell(i)

    # ---策略订单通知，已经进入下一个bar，且在next()之前执行
    @myBT.OnNotify_Order
    def notify_order(i):
        myBT.Self(i).barscount = myBT.bars_executed(i)

    @myBT.OnStop
    def stop(i):
        print("stop(): " , myBT.ValueCash(), myBT.Self(i).SMA[0])
    # ---
    myBT.addstrategy()
    # ---运行
myBT.run(maxcpus=1,plot = True)

