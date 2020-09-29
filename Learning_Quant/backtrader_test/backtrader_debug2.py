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
data0 = CJSecurities["2015"]

# ---基础设置
myBT = MyPackage.MyClass_BackTest.MyClass_BackTest()  #回测类
myBT.ValueCash(2000)
myBT.AddBarsData(data0,fromdate=None,todate=None)

# ---策略开始
@myBT.OnInit
def __init__():
    print("init检测无仓位 = ", not myBT.position())

# ---策略递归
order = []
@myBT.OnNext
def next():
    global order
    if myBT.bars_executed == 2:
        print(myBT.bars_executed ,"start buy")
        order.append(myBT.buy())
    if myBT.bars_executed == 3:
        print(myBT.bars_executed, "start sell")
        order.append(myBT.sell())
    if myBT.bars_executed == 4:
        print(myBT.bars_executed, "start buy and sell")
        order.append(myBT.buy())
        order.append(myBT.sell())

# ---策略订单触发订单通知，会在下一个bar的next()之前执行
@myBT.OnNotify_Order
def notify_order():
    print("notify_order 开始递交")
    print(myBT.bars_executed, "这是bar执行数量")
    if myBT.OrderStatusCheck(myBT.order_noti) == False:
        return
    print("notify_order 执行OK")

myBT.addstrategy()
# ---运行
myBT.run(plot = False)
print(order[0])


