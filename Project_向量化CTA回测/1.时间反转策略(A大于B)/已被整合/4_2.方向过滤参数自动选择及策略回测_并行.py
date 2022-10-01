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
mylogging = MyDefault.MyClass_Default_Logging(activate=True, filename=__mypath__.get_desktop_path()+"\\方向过滤策略回测.log") # 日志记录类，需要放在上面才行

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
# 1.根据前面 信号利润过滤测试 输出的文档，解析文档名称，读取参数，选择极值。
# 2.一个特定的策略参数作为一个目录，存放该下面所有指标的结果。
# 3.不同名称的指标会自动判断极值，且输出图片。最后会输出表格文档，整理这些极值。
# 4.由于不是大型计算，并行是一次性所有并行。
# 5.并行运算注意内存释放，并且不要一次性都算完，这样容易爆内存。分组进行并行。
'''
'''
# 说明
# 这里的策略回测是建立在前面已经对指标的范围过滤做了参数选择。
# 前面对每个具体策略都通过指标过滤方式，算出了各个指标过滤效果的极值。我们根据极值对应的指标值做回测。
# 画的图中，分别展示 过滤前训练集价格和指标、过滤前训练集策略、过滤后全集价格和指标、过滤后全集策略以及训练集策略。
# 方向过滤作用到整个样本。
# 并行以品种来并行，以时间框来分组。
# 由于指标较多，并行运算时间长，防止出错输出日志。
'''

#%%
from MyPackage.MyProjects.向量化策略测试.Direct_Filter import Auto_Choose_DFilter_Param
choo_para = Auto_Choose_DFilter_Param()
myDefault.set_backend_default("agg")


#%% ******需要修改******
choo_para.symbol_list = myMT5Pro.get_mainusd_symbol_name_list()
choo_para.total_folder = "F:\\工作---策略研究\\1.简单的动量反转\\_反转研究"
choo_para.core_num = -1


#%%
from MyPackage.MyProjects.向量化策略测试.Direct_Filter import Direct_Filter_BackTest
rf_bt = Direct_Filter_BackTest()
myplt.set_backend("agg")  # agg 后台输出图片，不占pycharm内存


#%%
rf_bt.symbol_list = choo_para.symbol_list
rf_bt.total_folder = choo_para.total_folder
rf_bt.core_num = -1


#%% ******修改函数******
#  策略的当期信号(不用平移)：para_list策略参数，默认-1为lag_trade，-2为holding。
def stratgy_signal(dataframe, para_list=list or tuple, stra_mode="Reverse"):
    price = dataframe["Close"]
    return myBTV.stra.momentum(price=price, k=para_list[0], stra_mode=stra_mode)
rf_bt.stratgy_signal = stratgy_signal


#%%
# ---多进程必须要在这里执行
if __name__ == '__main__':
    # ---
    print("开始方向过滤参数自动选择：")
    choo_para.main_func()
    print("开始方向过滤策略回测：")
    rf_bt.main_func()









