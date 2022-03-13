# Author:Zhang Yuan
from MyPackage import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# ------------------------------------------------------------
__mypath__ = MyPath.MyClass_Path("")  # 路径类
mylogging = MyDefault.MyClass_Default_Logging(activate=False)  # 日志记录类，需要放在上面才行
myfile = MyFile.MyClass_File()  # 文件操作类
myword = MyFile.MyClass_Word()  # word生成类
myexcel = MyFile.MyClass_Excel()  # excel生成类
myini = MyFile.MyClass_INI()  # ini文件操作类
mytime = MyTime.MyClass_Time()  # 时间类
myparallel = MyTools.MyClass_ParallelCal()  # 并行运算类
myplt = MyPlot.MyClass_Plot()  # 直接绘图类(单个图窗)
mypltpro = MyPlot.MyClass_PlotPro()  # Plot高级图系列
myfig = MyPlot.MyClass_Figure(AddFigure=False)  # 对象式绘图类(可多个图窗)
myfigpro = MyPlot.MyClass_FigurePro(AddFigure=False)  # Figure高级图系列
myplthtml = MyPlot.MyClass_PlotHTML() # 画可以交互的html格式的图
mynp = MyArray.MyClass_NumPy()  # 多维数组类(整合Numpy)
mypd = MyArray.MyClass_Pandas()  # 矩阵数组类(整合Pandas)
mypdpro = MyArray.MyClass_PandasPro()  # 高级矩阵数组类
myDA = MyDataAnalysis.MyClass_DataAnalysis()  # 数据分析类
myDefault = MyDefault.MyClass_Default_Matplotlib()  # 画图恢复默认设置类
# myMql = MyMql.MyClass_MqlBackups() # Mql备份类
# myBaidu = MyWebCrawler.MyClass_BaiduPan() # Baidu网盘交互类
# myImage = MyImage.MyClass_ImageProcess()  # 图片处理类
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector()  # 向量型回测类
myML = MyMachineLearning.MyClass_MachineLearning()  # 机器学习综合类
mySQL = MyDataBase.MyClass_MySQL(connect=False)  # MySQL类
mySQLAPP = MyDataBase.MyClass_SQL_APPIntegration()  # 数据库应用整合
myWebQD = MyWebCrawler.MyClass_QuotesDownload(tushare=False)  # 金融行情下载类
myWebR = MyWebCrawler.MyClass_Requests()  # Requests爬虫类
myWebS = MyWebCrawler.MyClass_Selenium(openChrome=False)  # Selenium模拟浏览器类
myWebAPP = MyWebCrawler.MyClass_Web_APPIntegration()  # 爬虫整合应用类
myEmail = MyWebCrawler.MyClass_Email()  # 邮箱交互类
myReportA = MyQuant.MyClass_ReportAnalysis()  # 研报分析类
myFactorD = MyQuant.MyClass_Factor_Detection()  # 因子检测类
myKeras = MyDeepLearning.MyClass_tfKeras()  # tfKeras综合类
myTensor = MyDeepLearning.MyClass_TensorFlow()  # Tensorflow综合类
myMT5 = MyMql.MyClass_ConnectMT5(connect=False)  # Python链接MetaTrader5客户端类
myMT5Pro = MyMql.MyClass_ConnectMT5Pro(connect=False)  # Python链接MT5高级类
myMT5Indi = MyMql.MyClass_MT5Indicator()  # MT5指标Python版
myMT5Report = MyMT5Report.MyClass_StratTestReport(AddFigure=False)  # MT5策略报告类
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=False)  # 固定杠杆仓位类
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=False)  # 浮动杠杆仓位类
myMT5run = MyMql.MyClass_RunningMT5()  # Python运行MT5
myMT5code = MyMql.MyClass_CodeMql5()  # Python生成MT5代码
myMoneyM = MyTrade.MyClass_MoneyManage()  # 资金管理类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------
# Jupyter Notebook 控制台显示必须加上：%matplotlib inline ，弹出窗显示必须加上：%matplotlib auto
# %matplotlib inline
# import warnings
# warnings.filterwarnings('ignore')

''' 详情请参考：https://gallery.pyecharts.org/#/README '''


#%% ### 3D曲面 ###
from typing import Union
def float_range(start: int, end: int, step: Union[int, float], round_number: int = 2):
    """
    浮点数 range
    :param start: 起始值
    :param end: 结束值
    :param step: 步长
    :param round_number: 精度
    :return: 返回一个 list
    """
    temp = []
    while True:
        if start < end:
            temp.append(round(start, round_number))
            start += step
        else:
            break
    return temp
def surface3d_data():
    import math
    for t0 in float_range(-3, 3, 0.05):
        y = t0
        for t1 in float_range(-3, 3, 0.05):
            x = t1
            z = math.sin(x ** 2 + y ** 2) * x / 3.14
            yield [x, y, z]
            # ---

data = list(surface3d_data())
data = pd.DataFrame(data)
data.columns = ["X","Y","Z"]
# ---
filepath = __mypath__.get_desktop_path()+r"\plot_surface3D.html"
surface3D = myplthtml.plot_surface3D(data=data, height=40, series_name="数据的名称", title="主题", savehtml = filepath)
import os
os.startfile(filepath)




#%% ### 3D散点图 ###
import random
data = [[i, j, random.randint(0, 12)] for i in range(24) for j in range(7)] # 小时，星期，value
data = pd.DataFrame(data)
data.columns = ["X","Y","Z"]
# ---
filepath = __mypath__.get_desktop_path()+r"\plot_scatter3D.html"
scatter3D = myplthtml.plot_scatter3D(data=data, series_name="数据的名称", title="主题", savehtml = filepath)
import os
os.startfile(filepath)


#%% ### 3D折线图 ###
import math
data = []
for t in range(0, 25000):
    _t = t / 1000
    x = (1 + 0.25 * math.cos(75 * _t)) * math.cos(_t)
    y = (1 + 0.25 * math.cos(75 * _t)) * math.sin(_t)
    z = _t + 2.0 * math.sin(75 * _t)
    data.append([x, y, z])
data = pd.DataFrame(data)
data.columns = ["X","Y","Z"]
# ---
filepath = __mypath__.get_desktop_path()+r"\line3d_autorotate.html"
line3D = myplthtml.plot_line3D(data=data, series_name="数据的名称", title="主题", savehtml = filepath)
import os
os.startfile(filepath)


# %% ### 3D柱状图 ###
hourslabel = ["12a","1a","2a", "3a","4a","5a","6a","7a","8a","9a","10a","11a","12p",  "1p", "2p", "3p","4p", "5p", "6p", "7p","8p","9p","10p","11p",]
weeklabel = ["Saturday", "Friday", "Thursday", "Wednesday", "Tuesday", "Monday", "Sunday"]
import random
data = [[i, j, random.randint(0, 12)] for i in range(24) for j in range(7)] # 小时，星期，value
# ---
data = pd.DataFrame(data)
data.columns = ["X","Y","Z"]
filepath = __mypath__.get_desktop_path()+r"\bar3d_punch_card.html"
bar3D = myplthtml.plot_bar3D(data=data, series_name="数据的名称", xlabellist=hourslabel, ylabellist=weeklabel, title="主题", savehtml = filepath)
import os
os.startfile(filepath)


#%% ### 画选项卡多图 ###
filepath = __mypath__.get_desktop_path()+r"\tab_base.html"
tab = myplthtml.plot_tab_chart(chart_list=[bar3D,line3D,scatter3D,surface3D],
                               tab_name_list=["bar3D","line3D","scatter3D","surface3D"],
                               savehtml=filepath)
import os
os.startfile(filepath)


