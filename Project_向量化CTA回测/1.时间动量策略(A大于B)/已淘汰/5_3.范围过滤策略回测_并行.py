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
mylogging = MyDefault.MyClass_Default_Logging(activate=True, filename=__mypath__.get_desktop_path()+"\\范围过滤策略回测.log") # 日志记录类，需要放在上面才行

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
# 说明
# 这里的策略回测是建立在前面已经对指标的范围过滤做了参数选择。
# 前面对每个具体策略都通过指标过滤方式，算出了各个指标过滤效果的极值。我们根据极值对应的指标值做回测。
# 画的图中，min-max表示 "max最大的以max之前的min最小" 或 "min最小的以min之后的max最大"，start-end表示上涨额度最大的区间。
# 根据训练集获取过滤区间，然后作用到整个样本。
# 并行以品种来并行，以时间框来分组。
# 由于指标较多，并行运算时间长，防止出错输出日志。
'''

#%%
from MyPackage.MyProjects.向量化策略测试.Range_Filter import Range_Filter_BackTest
rf_bt = Range_Filter_BackTest()
myplt.set_backend("agg")  # agg 后台输出图片，不占pycharm内存

#%% ************ 需要修改的部分 ************
rf_bt.symbol_list = myMT5Pro.get_mainusd_symbol_name_list()
rf_bt.total_folder = "F:\\工作---策略研究\\简单的动量反转\\_动量研究"

#%% ******修改函数******
# sig_mode方向、stra_mode策略模式(默认值重要，不明写)、para_list策略参数
def stratgy_signal(dataframe, para_list=list or tuple, stra_mode="Continue"):
    price = dataframe["Close"]
    return myBTV.stra.momentum(price=price, k=para_list[0], stra_mode=stra_mode)
rf_bt.stratgy_signal = stratgy_signal

#%%
rf_bt.core_num = -1 # 注意，M1时间框数据量较大时，并行太多会爆内存。
if __name__ == '__main__':
    # ---
    rf_bt.main_func()












