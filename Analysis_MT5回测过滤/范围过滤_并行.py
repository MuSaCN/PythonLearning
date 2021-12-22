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
myMT5Report = MyMql.MyClass_StratTestReport(AddFigure=False)  # MT5策略报告类
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=False)  # 固定杠杆仓位类
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=False)  # 浮动杠杆仓位类
myMT5run = MyMql.MyClass_RunningMT5()  # Python运行MT5
myMT5code = MyMql.MyClass_CodeMql5()  # Python生成MT5代码
myMoneyM = MyTrade.MyClass_MoneyManage()  # 资金管理类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------
# Jupyter Notebook 控制台显示必须加上：%matplotlib inline ，弹出窗显示必须加上：%matplotlib auto
# %matplotlib inline


# %%
import warnings
warnings.filterwarnings('ignore')

from MyPackage.MyProjects.MT5回测结果过滤.MT5_report_filter import MT5_Report_Filter
c_report_filter = MT5_Report_Filter()

# ---外部赋值
c_report_filter.file = __mypath__.get_desktop_path() + "\\ReportTester.xlsx"
c_report_filter.direct = "All"  # 方向 "All","BuyOnly","SellOnly"
c_report_filter.filtermode = "range"  # 过滤模式 "range","2side"

tf_indi="TIMEFRAME_H1"

#%%
# ---设置图片输出方式
myDefault.set_backend_default("agg") # 这句必须放到类下面

# ---读取报告，设定各种变量
c_report_filter.load_report()

# ---(用于并行)执行过滤，且输出过滤后的结果series.
# indiname="@RSI"
# para = [55]
# paralist = [tf_indi, indiname, *para]
# c_report_filter.run_filter(paralist)


#%%
# ---多进程必须要在这里执行
if __name__ == '__main__':
    indi_name_list = myBTV.indiMT5.indi_name_rangefilter()
    params_dict = myBTV.indiMT5.indi_params_scale1D(indi_name_list)
    # ---
    multi_params = []
    indi_name_list = [indi_name_list[0]]
    for indi_name in indi_name_list:  # indi_name = indi_name_list[0]
        params = params_dict[indi_name]
        params["tf_indi"] = tf_indi
        params = params[[params.columns[-1]] + params.columns[0:-1].tolist()] # 列排序重置下
        multi_params = multi_params + params.values.tolist()
    # ---开始多核执行
    xlsxname = "过滤结果_{0}_{1}.xlsx".format(c_report_filter.direct, c_report_filter.filtermode)
    myBTV.muiltcore.run_concat_dataframe(c_report_filter.run_filter, multi_params,
                                         filepath=c_report_filter.savefolder+"\\"+xlsxname,
                                         core_num=-1)
    print("过滤结束.")






