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

mylogging = MyDefault.MyClass_Default_Logging(activate=True, filename=__mypath__.get_desktop_path()+"\\信号利润过滤及测试.log") # 日志记录类，需要放在上面才行

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
myPjMT5 = MyProject.MT5_MLLearning()  # MT5机器学习项目类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------

'''
# 说明
# 根据信号的利润，运用其他指标来过滤，从累计利润角度进行过滤。可以分析出 其他指标的值 的哪些区间对于累计利润是正的贡献、哪些区间是负的贡献。所用的思想为“求积分(累积和)来进行噪音过滤”。
# 画的图中，min-max表示max最大的以max之前的min最小，start-end表示上涨额度最大的区间。
# 根据训练集获取过滤区间，然后作用到整个样本。
# 由于并行运算时间长，防止出错输出日志。
'''

myplt.set_backend("agg")  # agg 后台输出图片，不占pycharm内存

#%% 分析到此部分，基本确定了 某个品种、某个时间框、某个方向 的策略参数，并行主要体现在多个指标上。

# 不同方向 BuyOnly、SellOnly、All 的策略参数，根据前面分析后设置固定值。
strategy_para_direct = [[101,1,1], [101,1,1]] # 值对应["k", "holding", "lag_trade"]，且索引对应 direct_para

# 技术指标名称，参数设置在 -4 的位置，具体的参数指定，在 if __name__ == '__main__': 中
indi_name_list=["rsi"]

# 方向参数："BuyOnly" "SellOnly" "All"，保存在 para 的 -3 位置
direct_para = ["BuyOnly","SellOnly"]

# timeframe、symbol 参数设置在 -2、-1 的位置
timeframe_list = ["TIMEFRAME_D1"]
symbol_list = ["EURUSD"]


#%%
# para 传递指标的参数和策略的参数，结果输出图片。
def run_plot(para):
    # 显示进度
    print("\r", "当前执行参数为：", para, end="", flush=True)
    # 非策略参数
    # para = ("Close", 20) + ("rsi", "BuyOnly", "TIMEFRAME_D1", "EURUSD")
    indi_name = para[-4]
    direct = para[-3]
    timeframe = para[-2]
    symbol = para[-1]

    # ---获取数据
    date_from, date_to = myPjMT5.get_date_range(timeframe)
    data_total = myPjMT5.getsymboldata(symbol, timeframe, date_from, date_to, index_time=True, col_capitalize=True)
    # 由于信号利润过滤是利用训练集的，所以要区分训练集和测试集
    data_train, data_test = myPjMT5.get_train_test(data_total, train_scale=0.8)
    # 测试不需要把数据集区分训练集、测试集，仅画区间就可以了
    train_x0 = data_train.index[0]
    train_x1 = data_train.index[-1]
    # 把训练集的时间进行左右扩展
    bound_left, bound_right = myPjMT5.extend_train_time(train_t0=train_x0, train_t1=train_x1, extend_scale=0)
    # 再次重新加载下全部的数据
    data_total = myPjMT5.getsymboldata(symbol, timeframe, bound_left, bound_right, index_time=True, col_capitalize=True)

    # ---加载固定的参数
    k, holding, lag_trade = strategy_para_direct[direct_para.index(direct)]

    # ---获取训练集和整个样本的信号
    # 获取训练集的信号 ***(修改这里)***
    signaldata_train = myBTV.stra.momentum(data_train.Close, k=k, holding=holding, sig_mode=direct, stra_mode="Continue")
    signal_train = signaldata_train[direct]
    # 计算整个样本的信号 ***(修改这里)***
    signaldata_all = myBTV.stra.momentum(data_total.Close, k=k, holding=holding, sig_mode=direct, stra_mode="Continue")
    signal_all = signaldata_all[direct]

    # ---(核心，在库中添加)获取指标
    indicator = myBTV.indi.multicore_get_indicator(data_total, indi_name, para)

    # ---信号利润过滤及测试
    # 总目录 ***(修改这里)***
    folder = __mypath__.get_desktop_path() + "\\_动量研究\\指标过滤\\{}.{}".format(symbol, timeframe)
    savefig = folder + "\\{}\\{}\\{}{}.png".format(indi_name, direct, indi_name, para[:-4])
    # 过滤及测试后，输出图片
    myBTV.plot_signal_indicator_filter_and_quality(signal_train=signal_train, signal_all=signal_all, indicator=indicator, train_x0=train_x0, train_x1=train_x1, price_DataFrame=data_total, price_Series=data_total.Close, holding=holding, lag_trade=lag_trade, noRepeatHold=True, indi_name="%s(%s)" % (indi_name, para[:-4]), savefig=savefig)


#%%
core_num = -1
if __name__ == '__main__':
    for timeframe in timeframe_list:
        finish_symbol = [] # 记录品种完成进度
        for symbol in symbol_list:
            for direct in direct_para:
                finish_indi = [] # 由于指标很多，记录指标完成进度
                for indi_name in indi_name_list:
                    # ---(核心部分)不同名称的技术指标，设定不同的多核运算参数范围
                    if indi_name == "rsi":
                        multi_params = [("Close", i) + (indi_name, direct, timeframe, symbol) for i in range(5, 100 + 1)]
                    # ---开始多核执行
                    import timeit
                    t0 = timeit.default_timer()
                    myBTV.multi_processing(run_plot, multi_params, core_num=core_num)
                    t1 = timeit.default_timer()
                    print("\n", '{}.{}.{}.{} 耗时为：'.format(symbol, timeframe, direct, indi_name), t1 - t0)
                    # ---记录指标完成
                    finish_indi.append(indi_name)
                    # 由于并行时间长，要记录到logging
                    mylogging.warning("indi finished: {} {} {} {}".format(timeframe, symbol, direct, finish_indi))
            finish_symbol.append(symbol)
            mylogging.warning("symbol finished: {} {}".format(timeframe, finish_symbol))













