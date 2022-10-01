# Author:Zhang Yuan
import warnings
warnings.filterwarnings('ignore')
#
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

mylogging = MyDefault.MyClass_Default_Logging(activate=True, filename=__mypath__.get_desktop_path()+"\\参数优化.log") # 日志记录类，需要放在上面才行

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
""
# 策略说明：
'''
# 反转策略，采用最简单的反转计算形式：
# 当天的收盘价A1 小于 过去某一期的收盘价B1，表示下跌会反转，则触发买入信号；
# 当天的收盘价A2 大于 过去某一期的收盘价B2，表示上涨会反转，则触发卖出信号；
# 信号触发后，下一期(或下n期)进行交易。持有仓位周期为1根K线。
'''

'''
# 参数优化说明：
# 参数优化部分，需要专门设定训练集和测试集。由于参数较多，不可能都通过图示。所以，通过训练集来计算出各个参数下策略结果，安全起见保存结果到硬盘。
# 再根据训练集参数优化的结果，计算对应参数下测试集策略结果，把结果保存到硬盘。
# 整合两个结果到一张表格。
# 需要注意的是，由于 训练集和测试集 信号计算时依赖的数据集不同，所以要设定两个函数。
# 由于并行运算的原理，参数分为 策略参数 + 非策略参数
# 为了提高运算速度，可以只测试训练集，然后再通过后面的分析筛选。
# 由于并行运算时间长，防止出错输出日志。
'''

#%%
from MyPackage.MyProjects.向量化策略测试.Strategy_Param_Opt import Strategy_Param_Opt_OutPut
opt = Strategy_Param_Opt_OutPut()


#%% ************ 需要修改的部分 ************
# 策略参数，设置范围的最大值，按顺序保存在 para 的前面
opt.strategy_para_names = ["k", "holding", "lag_trade"]  # 顺序不能搞错了，要与信号函数中一致
opt.para1_end = 300         # 动量向左参数
opt.holding_end = 1         # 持有期参数，可以不同固定为1
opt.lag_trade_end = 1       # 信号出现滞后交易参数，参数不能大
# 非策略参数
opt.direct_para = ["BuyOnly", "SellOnly"] # direct_para = ["BuyOnly", "SellOnly", "All"]
opt.symbol_list = myMT5Pro.get_mainusd_symbol_name_list()
opt.total_folder = "F:\\工作---策略研究\\1.简单的动量反转\\_反转研究"
opt.filename_prefix = "反转"

#%% ******修改函数******
#  策略的当期信号(不用平移)：para_list策略参数，默认-1为lag_trade，-2为holding。
def stratgy_signal(dataframe, para_list=list or tuple, stra_mode="Reverse"):
    price = dataframe["Close"]
    return myBTV.stra.momentum(price=price, k=para_list[0], stra_mode=stra_mode)
opt.stratgy_signal = stratgy_signal

#%% ******修改函数******
# 获取策略参数范围(direct、timeframe、symbol参数必须设置在-3、-2、-1的位置)
def get_strat_para_scope(direct, timeframe, symbol):
    return [(k, holding, lag_trade, direct, timeframe, symbol)
            for k in range(1, opt.para1_end + 1)
            for holding in range(1, opt.holding_end + 1)
            for lag_trade in range(1, opt.lag_trade_end + 1)]
opt.get_strat_para_scope = get_strat_para_scope

# 策略退出条件，strat_para = (k, holding, lag_trade)。
def strat_break(strat_para):
    if strat_para[1] > strat_para[0]:
        return True
opt.strat_break = strat_break


#%%
opt.core_num = 11 # 具体指定，不能是-1，要显示进度
# ---多进程必须要在这里执行
if __name__ == '__main__':
    # ---
    opt.main_func(run_testset=False)





