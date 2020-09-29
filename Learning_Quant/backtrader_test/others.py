# Author:Zhang Yuan
from MyPackage import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
#------------------------------------------------------------
__mypath__ = MyPath.MyClass_Path()  # 路径类
myfile = MyFile.MyClass_File()  # 文件操作类
mytime = MyTime.MyClass_Time()  # 时间类
myplt = MyPlot.MyClass_Plot()  # 直接绘图类(单个图窗)
mypltpro = MyPlot.MyClass_PlotPro()  # Plot高级图系列
myfig = MyPlot.MyClass_Figure(AddFigure=False)  # 对象式绘图类(可多个图窗)
myfigpro = MyPlot.MyClass_FigurePro(AddFigure=False)  # Figure高级图系列
mynp = MyArray.MyClass_NumPy()  # 多维数组类(整合Numpy)
mypd = MyArray.MyClass_Pandas()  # 矩阵数组类(整合Pandas)
mypdpro = MyArray.MyClass_PandasPro()  # 高级矩阵数组类
myDA = MyDataAnalysis.MyClass_DataAnalysis()  # 数据分析类
# myMql = MyMql.MyClass_MqlBackups() # Mql备份类
# myDefault = MyDefault.MyClass_Default_Matplotlib() # matplotlib默认设置
# myBaidu = MyWebCrawler.MyClass_BaiduPan() # Baidu网盘交互类
myWebQD = MyWebCrawler.MyClass_WebQuotesDownload()  # 金融行情下载类
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector() # 向量型回测类
#------------------------------------------------------------


def decorator_maker_with_arguments(decorator_arg1, decorator_arg2):
    def my_decorator(func):
        def wrapped(function_arg1, function_arg2) :
            print (decorator_arg1, decorator_arg2, function_arg1, function_arg2)
            return func(function_arg1, function_arg2)
        return wrapped
    return my_decorator


def my_decorator(self):
    def wrapped(func) :
        def next(*args,**kwargs):
            print(*args,**kwargs)
        return next
    return wrapped


@my_decorator("self")
def decorated_function_with_arguments(self,d):
    print("OK")


decorated_function_with_arguments( "ABC","DEF")


class A():
    def __init__(self):
        self.a1 = [0]
        self.a2 = 2
    def produce(self):
        b = self.a1
        class AA:
            def __init__(self):
                b[0] = self
        AA()
    class AAA:
        def __init__(self):
            print(123)
