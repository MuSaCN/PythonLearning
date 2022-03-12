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

# %%
import random
from pyecharts import options as opts
from pyecharts.charts import Scatter3D
from pyecharts.faker import Faker

Scatter_data = [(random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)) for i in range(50)]
scatter = (
    Scatter3D(init_opts=opts.InitOpts(width='900px', height='600px'))  # 初始化
        .add("", Scatter_data,
             grid3d_opts=opts.Grid3DOpts(
                 width=100, depth=100, rotate_speed=500, is_rotate=True
             ))

        # 设置全局配置项
        .set_global_opts(
        title_opts=opts.TitleOpts(title="3D散点图"),  # 添加标题
        visualmap_opts=opts.VisualMapOpts(
            max_=50,  # 最大值
            pos_top=50,  # visualMap 组件离容器上侧的距离
            range_color=Faker.visual_color  # 颜色映射
        )
    )
        .render(__mypath__.get_desktop_path() + r"\3D散点图.html")
)



#%%
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import talib as ta
from pyecharts import Line, Kline, Pie, Grid, Overlap, Timeline, WordCloud

#%%
import math
from typing import Union

import pyecharts.options as opts
from pyecharts.charts import Surface3D

"""
Gallery 使用 pyecharts 1.1.0
参考地址: https://echarts.apache.org/examples/editor.html?c=surface-wave&gl=1

目前无法实现的功能:

1、暂时无法设置光滑表面 wireframe
2、暂时无法把 visualmap 进行隐藏
"""


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
    for t0 in float_range(-3, 3, 0.05):
        y = t0
        for t1 in float_range(-3, 3, 0.05):
            x = t1
            z = math.sin(x ** 2 + y ** 2) * x / 3.14
            yield [x, y, z]


(
    Surface3D(init_opts=opts.InitOpts(width="1600px", height="800px"))
    .add(
        series_name="",
        shading="color",
        data=list(surface3d_data()),
        xaxis3d_opts=opts.Axis3DOpts(type_="value"),
        yaxis3d_opts=opts.Axis3DOpts(type_="value"),
        grid3d_opts=opts.Grid3DOpts(width=100, height=40, depth=100),
    )
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(
            dimension=2,
            max_=1,
            min_=-1,
            range_color=[
                "#313695",
                "#4575b4",
                "#74add1",
                "#abd9e9",
                "#e0f3f8",
                "#ffffbf",
                "#fee090",
                "#fdae61",
                "#f46d43",
                "#d73027",
                "#a50026",
            ],
        )
    )
    .render(__mypath__.get_desktop_path()+r"\surface_wave.html")
)

#%%
from pyecharts import options as opts
from pyecharts.charts import Map3D
from pyecharts.globals import ChartType

example_data = [
    [[119.107078, 36.70925, 1000], [116.587245, 35.415393, 1000]],
    [[117.000923, 36.675807], [120.355173, 36.082982]],
    [[118.047648, 36.814939], [118.66471, 37.434564]],
    [[121.391382, 37.539297], [119.107078, 36.70925]],
    [[116.587245, 35.415393], [122.116394, 37.509691]],
    [[119.461208, 35.428588], [118.326443, 35.065282]],
    [[116.307428, 37.453968], [115.469381, 35.246531]],
]
c = (
    Map3D()
    .add_schema(
        maptype="山东",
        itemstyle_opts=opts.ItemStyleOpts(
            color="rgb(5,101,123)",
            opacity=1,
            border_width=0.8,
            border_color="rgb(62,215,213)",
        ),
        light_opts=opts.Map3DLightOpts(
            main_color="#fff",
            main_intensity=1.2,
            is_main_shadow=False,
            main_alpha=55,
            main_beta=10,
            ambient_intensity=0.3,
        ),
        view_control_opts=opts.Map3DViewControlOpts(center=[-10, 0, 10]),
        post_effect_opts=opts.Map3DPostEffectOpts(is_enable=False),
    )
    .add(
        series_name="",
        data_pair=example_data,
        type_=ChartType.LINES3D,
        effect=opts.Lines3DEffectOpts(
            is_show=True,
            period=4,
            trail_width=3,
            trail_length=0.5,
            trail_color="#f00",
            trail_opacity=1,
        ),
        linestyle_opts=opts.LineStyleOpts(is_show=False, color="#fff", opacity=0),
    )
    .set_global_opts(title_opts=opts.TitleOpts(title="Map3D-Lines3D"))
    .render(__mypath__.get_desktop_path()+r"\map3d_with_lines3d.html")
)


#%%
from pyecharts.charts import Bar

bar = Bar()
bar.add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
bar.add_yaxis("商家A", [5, 20, 36, 10, 75, 90])
# render 会生成本地 HTML 文件，默认会在当前目录生成 render.html 文件
# 也可以传入路径参数，如 bar.render("mycharts.html")
bar.render() # 会默认到当前工作位置，要输入路径



