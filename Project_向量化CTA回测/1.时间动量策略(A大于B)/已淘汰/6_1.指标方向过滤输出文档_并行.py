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
mylogging = MyDefault.MyClass_Default_Logging(activate=True, filename=__mypath__.get_desktop_path()+"\\指标方向过滤输出文档.log") # 日志记录类，需要放在上面才行

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
myMT5Pro = MyMql.MyClass_ConnectMT5Pro(connect = False) # Python链接MT5高级类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------


'''
# 说明：
# 1.根据趋势性指标进行策略方向性过滤。价格在指标上方，只做多、不做空；价格在指标下方，只做空，不做多。
# 2.根据训练集获取过滤区间，然后作用到训练集，不是整个样本。
# 3.一个策略参数有许多个指标，每个指标有许多指标参数，这些结果都放到一个表格中。
# 4.有许多个指标，所以通过并行运算。并行是对一个品种、一个时间框下、一个方向下，不同指标的不同参数进行并行。
# 5.表格文档存放到硬盘路径"_**研究\过滤指标参数自动选择\symbol.timeframe"，以便于下一步极值分析。
# 6.由于属于大型计算，并行运算时间长，防止出错要输出日志。
# 7.后期要通过动态读取文件来解析品种、时间框、方向、策略参数名、策略参数值等
'''

#%%
from MyPackage.MyProjects.向量化策略测试.Direct_Filter import Direct_Filter_Output
df_out = Direct_Filter_Output()

#%% ******修改这里******
# 策略参数名称，用于文档中解析参数 ***修改这里***
df_out.strategy_para_name = ["k", "holding", "lag_trade"]
df_out.symbol_list = myMT5Pro.get_mainusd_symbol_name_list()
df_out.total_folder = "F:\\工作---策略研究\\简单的动量反转\\_动量研究"


#%% ******修改这个函数******
#  sig_mode方向、stra_mode策略模式(默认值重要，不明写)、para_list策略参数
def stratgy_signal(dataframe, para_list=list or tuple, stra_mode="Continue"):
    price = dataframe["Close"]
    return myBTV.stra.momentum(price=price, k=para_list[0], stra_mode=stra_mode)
df_out.stratgy_signal = stratgy_signal


#%%
df_out.core_num = -1
if __name__ == '__main__':
    # ---
    df_out.main_func()


