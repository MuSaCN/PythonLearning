# Author:Zhang Yuan
import warnings
warnings.filterwarnings('ignore')

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
# myDefault.set_backend_default("agg")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------
# Jupyter Notebook 控制台显示必须加上：%matplotlib inline ，弹出窗显示必须加上：%matplotlib auto
# %matplotlib inline

# %%
import warnings
warnings.filterwarnings('ignore')

#%%

# ---多进程必须要在这里执行
if __name__ == '__main__':
    from MyPackage.MyProjects.MT5回测结果过滤.MT5_report_filter import MT5_Report_Filter
    myDefault.set_backend_default("agg")  # 设置图片输出方式，这句必须放到类下面.
    plt.show()
    # ---从桌面加载
    outdesktopfile = __mypath__.get_desktop_path() + r"\通用过滤参数.csv"
    # 读取
    readfile = myfile.read_pd(outdesktopfile, sep=";")
    readfile.set_index(keys="0",drop=True,inplace=True)
    # 加载参数
    file = "通用过滤.htm"
    direct = readfile.loc["direct"][0]
    filtermode = readfile.loc["filtermode"][0]
    tf_indi = readfile.loc["tf_indi"][0]
    core_num = int(readfile.loc["core_num"][0])
    print(readfile)
    # ---输入调整
    file = "ReportTester.xlsx" if file == "" else file
    direct = "All" if direct == "" else direct
    if filtermode == "" or filtermode == "-1":
        filtermode = "all" # 所有的都测试
    elif filtermode == "0":
        filtermode = "range"
    elif filtermode == "1":
        filtermode = "2side"
    tf_indi = "TIMEFRAME_H1" if tf_indi == "" else tf_indi

    # ---如果是都测试
    if filtermode == "all":
        print("所有的都测试filtermode=", filtermode)
        # ===范围过滤===
        print("===开始范围过滤===")
        c_report_filter = MT5_Report_Filter()
        myDefault.set_backend_default("agg")  # 设置图片输出方式，这句必须放到类下面.
        plt.show()
        # ---外部赋值
        c_report_filter.core_num = core_num
        c_report_filter.file = __mypath__.get_desktop_path() + "\\" + file
        c_report_filter.direct = direct  # 方向 "All","BuyOnly","SellOnly"
        c_report_filter.filtermode = "range"  # 过滤模式 "range","2side"
        c_report_filter.tf_indi = tf_indi  # 指标的时间框，可以与报告的不同
        # ---读取报告，设定各种变量
        c_report_filter.load_report()
        # ---并行运算，输出过滤的文本文档
        c_report_filter.main_filter_and_xlsx()
        # ---参数过滤自动选择，且画图、输出xlsx。
        c_report_filter.main_auto_kalman_choose()
        # ---并行运算，卡尔曼选择后策略回测
        c_report_filter.main_auto_kalman_stratgy_test()

        # ===两侧过滤===
        plt.close()
        plt.show()
        plt.close() # 必须要先释放下，不然多进程分别测试各个模式会出错。
        print("===开始两侧过滤===")
        c_report_filter = MT5_Report_Filter()
        myDefault.set_backend_default("agg")  # 设置图片输出方式，这句必须放到类下面.
        plt.show()
        # ---外部赋值
        c_report_filter.core_num = core_num
        c_report_filter.file = __mypath__.get_desktop_path() + "\\" + file
        c_report_filter.direct = direct  # 方向 "All","BuyOnly","SellOnly"
        c_report_filter.filtermode = "2side"  # 过滤模式 "range","2side"
        c_report_filter.tf_indi = tf_indi  # 指标的时间框，可以与报告的不同
        # ---读取报告，设定各种变量
        c_report_filter.load_report()
        # ---并行运算，输出过滤的文本文档
        c_report_filter.main_filter_and_xlsx()
        # ---参数过滤自动选择，且画图、输出xlsx。
        c_report_filter.main_auto_kalman_choose()
        # ---并行运算，卡尔曼选择后策略回测
        c_report_filter.main_auto_kalman_stratgy_test()

    # ---仅测试指定的模式
    else:
        c_report_filter = MT5_Report_Filter()
        myDefault.set_backend_default("agg")  # 设置图片输出方式，这句必须放到类下面.
        plt.show()
        # ---外部赋值
        c_report_filter.core_num = core_num
        c_report_filter.file = __mypath__.get_desktop_path() + "\\" + file
        c_report_filter.direct = direct  # 方向 "All","BuyOnly","SellOnly"
        c_report_filter.filtermode = filtermode  # 过滤模式 "range","2side"
        c_report_filter.tf_indi = tf_indi  # 指标的时间框，可以与报告的不同
        # ---读取报告，设定各种变量
        c_report_filter.load_report()
        # ---并行运算，输出过滤的文本文档
        c_report_filter.main_filter_and_xlsx()
        # ---参数过滤自动选择，且画图、输出xlsx。
        c_report_filter.main_auto_kalman_choose()
        # ---并行运算，卡尔曼选择后策略回测
        c_report_filter.main_auto_kalman_stratgy_test()
    # ---
    # input("All Finished, any input to exit!")







