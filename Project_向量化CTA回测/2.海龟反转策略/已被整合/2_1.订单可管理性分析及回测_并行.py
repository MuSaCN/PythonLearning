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
mylogging = MyDefault.MyClass_Default_Logging(activate=True, filename=__mypath__.get_desktop_path()+"\\订单可管理性分析.log") # 日志记录类，需要放在上面才行
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
# 订单可管理性：如果一个策略在未来1期持仓表现不错，同时在未来多期持仓也表现不错。这就表明，这个策略的交易订单在时间伸展上能够被管理，我们称作为订单具备可管理性。
# 对训练集进行多holding回测，展示结果的夏普比曲线和胜率曲线。
# 采用无重复持仓模式和重复持仓模式。
# 如果前3个夏普都是递增的，则选择之。输出测试图片。否则不认为具有可管理性，则弃之。
# 并行运算以品种来并行
'''
'''
# 0.这里的回测是建立在前面已经对策略的参数做了选择。
# 1.根据前面整理的自动选择的最佳参数表格文档，读取参数，再做原始的策略测试。
# 2.策略结果保存到 "策略参数自动选择\品种\auto_para_1D_{order}\原始策略回测_filter1" 文件夹下面。
# 3.策略测试所用的区间要增大。
# 4.回测结果较多，构成策略库供后续选择研究。
# 5.并行运算注意内存释放，并且不要一次性都算完，这样容易爆内存。分组进行并行。
'''

#%%
from MyPackage.MyProjects.向量化策略测试.More_Holding import Auto_More_Holding
more_h = Auto_More_Holding()
myDefault.set_backend_default("agg")

#%% ******修改这里******
more_h.strategy_para_name = ["n", "holding", "lag_trade"]
more_h.symbol_list = myMT5Pro.get_mainusd_symbol_name_list()
more_h.total_folder = "F:\\工作---策略研究\\2.公开的海龟策略\\_海龟反转研究"
more_h.readfile_suffix = ".original" # 输入的文档加后缀
more_h.outfile_suffix = ".holdingtest" # 输出的文档加后缀
more_h.core_num = -1
more_h.holding_testcount = 3  # 测试到的holding数量

#%% ******修改函数******
#  策略的当期信号(不用平移)：para_list策略参数，默认-1为lag_trade，-2为holding。
def stratgy_signal(dataframe, para_list=list or tuple):
    return myBTV.stra.turtle_reverse(dataframe, para_list[0], price_arug= ["High", "Low", "Close"])
more_h.stratgy_signal = stratgy_signal


#%%
from MyPackage.MyProjects.向量化策略测试.More_Holding import Strategy_BackTest
strat_bt = Strategy_BackTest()
myDefault.set_backend_default("agg")

#%% ************ 需要修改的部分 ************
# 策略内参数(非策略参数 symbol、timeframe、direct 会自动解析) ******修改这里******
strat_bt.para_name = more_h.strategy_para_name
strat_bt.symbol_list = more_h.symbol_list
strat_bt.total_folder = more_h.total_folder
strat_bt.readfile_suffix = ".holdingtest" # 输入的文档加后缀
strat_bt.core_num = -1 # -1表示留1个进程不执行运算。
strat_bt.stratgy_signal = stratgy_signal


#%%

if __name__ == '__main__':
    # ---
    print("开始订单可管理性分析： ")
    more_h.main_func()
    print("开始筛选后策略自动回测： ")
    strat_bt.main_func()

