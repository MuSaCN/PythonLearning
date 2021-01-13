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
__mypath__ = MyPath.MyClass_Path("")  # 路径类
mylogging = MyDefault.MyClass_Default_Logging(activate=False)  # 日志记录类，需要放在上面才行
myfile = MyFile.MyClass_File()  # 文件操作类
myword = MyFile.MyClass_Word()  # word生成类
myexcel = MyFile.MyClass_Excel()  # excel生成类
mytime = MyTime.MyClass_Time()  # 时间类
myplt = MyPlot.MyClass_Plot()  # 直接绘图类(单个图窗)
mypltpro = MyPlot.MyClass_PlotPro()  # Plot高级图系列
myfig = MyPlot.MyClass_Figure(AddFigure=False)  # 对象式绘图类(可多个图窗)
myfigpro = MyPlot.MyClass_FigurePro(AddFigure=False)  # Figure高级图系列
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
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------

### 测试有问题，数据无法转换.
# -*- coding: utf-8 -*-
"""
Highstock Demos
Two panes, candlestick and volume: http://www.highcharts.com/stock/demo/candlestick-and-volume
"""
from highcharts import Highstock
from highcharts.highstock.highstock_helper import jsonp_loader
H = Highstock()

# data_url = 'http://www.highcharts.com/samples/data/jsonp.php?filename=aapl-ohlcv.json&callback=?'
# data = jsonp_loader(data_url, sub_d = r'(\/\*.*\*\/)')

data = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2000,12,4,0,0,0],[2020,12,5,0,0,0],index_time=True, col_capitalize=True)

time = data["Time"].apply(str)
open = data["Open"].apply(str)
high = data["High"].apply(str)
low = data["Low"].apply(str)
close = data["Close"].apply(str)
tick_volume = data["Tick_volume"].apply(str)

#%%
ohlc = []
volume = []
groupingUnits = [
['week', [1]],
['month', [1, 2, 3, 4, 6]]
]

#%%
for i in range(len(data)):
    ohlc.append(
        [
        time[i], # data[i][0], # the date
        open[i],# data[i][1], # open
        high[i],# data[i][2], # high
        low[i],# data[i][3], # low
        close[i],# data[i][4]  # close
        ]
        )
    volume.append(
        [
        time[i],# data[i][0], # the date
        int(tick_volume[i]),# data[i][5]  # the volume
        ]
    )

#%%
options = {
    'rangeSelector': {
                'selected': 1
            },

    'title': {
        'text': 'AAPL Historical'
    },

    'yAxis': [{
        'labels': {
            'align': 'right',
            'x': -3
        },
        'title': {
            'text': 'OHLC'
        },
        'height': '60%',
        'lineWidth': 2
    }, {
        'labels': {
            'align': 'right',
            'x': -3
        },
        'title': {
            'text': 'Volume'
        },
        'top': '65%',
        'height': '35%',
        'offset': 0,
        'lineWidth': 2
    }],
}

H.add_data_set(ohlc, 'candlestick', 'AAPL', dataGrouping = {
                    'units': groupingUnits
                }
)
H.add_data_set(volume, 'column', 'Volume', yAxis = 1, dataGrouping = {
                    'units': groupingUnits
                }
)

H.set_dict_options(options)

H.htmlcontent







