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

#------------------------------------------------------------
__mypath__ = MyPath.MyClass_Path("")  # 路径类
mylogging = MyDefault.MyClass_Default_Logging(activate=True, filename=__mypath__.get_desktop_path()+"\\指标范围过滤输出文档.log") # 日志记录类，需要放在上面才行

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


'''
# 说明
# 1.根据信号的利润，运用其他指标来过滤，从累计利润角度进行过滤。可以分析出 其他指标的值 的哪些区间对于累计利润是正的贡献、哪些区间是负的贡献。所用的思想为“求积分(累积和)来进行噪音过滤”。
# 2.根据训练集获取过滤区间，然后作用到训练集，不是整个样本。
# 3.一个策略参数有许多个指标，每个指标有许多指标参数，这些结果都放到一个表格中。
# 4.有许多个指标，所以通过并行运算。并行是对一个品种、一个时间框下、一个方向下，不同指标的不同参数进行并行。
# 5.表格文档存放到硬盘路径"_**研究\过滤指标参数自动选择\symbol.timeframe"，以便于下一步极值分析。
# 6.由于属于大型计算，并行运算时间长，防止出错要输出日志。
# 7.后期要通过动态读取文件来解析品种、时间框、方向、策略参数名、策略参数值等
'''

#%%
from MyPackage.MyProjects.向量化策略测试.Range_Filter import Range_Filter_Output
rf_out = Range_Filter_Output()

#%% ************ 需要修改的部分 ************
rf_out.strategy_para_name = ["n", "holding", "lag_trade"]
rf_out.symbol_list = myMT5Pro.get_mainusd_symbol_name_list()
rf_out.total_folder = "F:\\工作---策略研究\\2.公开的海龟策略\\_海龟动量研究"
rf_out.readfile_suffix = ".better"

#%% ******修改这个函数******
#  策略的当期信号(不用平移)：para_list策略参数，默认-1为lag_trade，-2为holding。
def stratgy_signal(dataframe, para_list=list or tuple):
    return myBTV.stra.turtle_momentum(dataframe, para_list[0], price_arug= ["High", "Low", "Close"])
rf_out.stratgy_signal = stratgy_signal

#%%
rf_out.core_num = -1
if __name__ == '__main__':
    # ---
    rf_out.main_func()









