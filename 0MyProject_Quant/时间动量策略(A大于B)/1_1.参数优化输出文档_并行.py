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
myMT5Pro = MyMql.MyClass_ConnectMT5Pro(connect = False) # Python链接MT5高级类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------
""
# 策略说明：
'''
# 动量策略，采用最简单的动量计算形式：
# 当天的收盘价A1 大于 过去某一期的收盘价B1，表示上涨动量会持续，则触发买入信号；
# 当天的收盘价A2 小于 过去某一期的收盘价B2，表示下跌动量会持续，则触发卖出信号；
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

#%% ************ 需要修改的部分 ************
# 策略参数，设置范围的最大值，按顺序保存在 para 的前面。******修改这里******
strategy_para_names = ["k", "holding", "lag_trade"]  # 顺序不能搞错了，要与信号函数中一致
k_end = 400             # 动量向左参数
holding_end = 1         # 持有期参数，可以不同固定为1
lag_trade_end = 1       # 信号出现滞后交易参数，参数不能大
# 非策略参数，******修改这里******
direct_para = ["BuyOnly", "SellOnly"] # direct_para = ["BuyOnly", "SellOnly", "All"]
symbol_list = myMT5Pro.get_main_symbol_name_list()
total_folder = "F:\\工作---策略研究\\简单的动量反转\\_动量研究test"
filename_prefix = "动量"
# 获取策略参数范围，******修改函数******(direct、timeframe、symbol参数必须设置在-3、-2、-1的位置)
def get_strat_para_scope(direct, timeframe, symbol):
    return [(k, holding, lag_trade, direct, timeframe, symbol) for k in range(1, k_end + 1) for holding in range(1, holding_end + 1) for lag_trade in range(1, lag_trade_end + 1)]
# 策略退出条件，strat_para = (k, holding, lag_trade)。******修改函数******
def strat_break(strat_para):
    if strat_para[1] > strat_para[0]:
        return True
# ******修改这个函数****** sig_mode方向、stra_mode策略模式(默认值重要，不明写)、para_list策略参数。
def stratgy_signal(price, sig_mode, stra_mode="Continue", para_list=list or tuple):
    return myBTV.stra.momentum(price=price, k=para_list[0], holding=para_list[1], sig_mode=sig_mode, stra_mode=stra_mode)

#%% ################# 信号函数部分，或多个函数、或多个参数 #####################
temp = 0  # 用来显示进度，必须放在这里
# 必须把总结果写成函数，且只能有一个参数，所以参数以列表或元组形式传递。内部参数有的要依赖于外部。
# ---训练集 计算信号，不重复持仓
def signalfunc_NoRepeatHold_train(para):
    # 策略参数
    # para = (101, 1, 1, "BuyOnly", "TIMEFRAME_D1", "EURUSD")
    strat_para = para[0:-3] # -1必为lag_trade，-2必为holding
    # 打印进度
    global temp
    temp += 1
    print("\r", "{}/{} train: ".format(temp * core_num, k_end * holding_end * lag_trade_end), para, end="", flush=True)
    # 非策略参数
    direct = para[-3] # "BuyOnly","SellOnly","All"
    timeframe = para[-2]
    symbol = para[-1]
    # 获取数据
    date_from, date_to = myMT5Pro.get_date_range(timeframe) # 不同时间框架加载的时间范围不同
    data_total = myMT5Pro.getsymboldata(symbol, timeframe, date_from, date_to, index_time=True, col_capitalize=True)
    data_train, data_test = myMT5Pro.get_train_test(data_total, train_scale=0.8)
    # 退出条件
    if strat_break(strat_para): return None
    # 获取信号数据
    signaldata = stratgy_signal(data_train.Close,sig_mode=direct,para_list=strat_para)
    # 信号分析
    outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signaldata[direct], price_DataFrame=data_train, holding=strat_para[-2], lag_trade=strat_para[-1], plotRet=False, plotStrat=False)
    # 设置信号统计
    result = myBTV.filter_strategy(outStrat, outSignal, para, strategy_para_names)
    return result

# ---测试集 计算信号，不重复持仓
def signalfunc_NoRepeatHold_test(para):
    # 策略参数
    # para = (101, 1, 1, "BuyOnly", "TIMEFRAME_D1", "EURUSD")
    strat_para = para[0:-3] # -1必为lag_trade，-2必为holding
    # 打印进度
    global temp
    temp += 1
    print("\r", "{}/{} test: ".format(temp * core_num, k_end * holding_end * lag_trade_end), para, end="", flush=True)
    # 非策略参数
    direct = para[-3]  # "BuyOnly","SellOnly","All"
    timeframe = para[-2]
    symbol = para[-1]
    # 获取数据
    date_from, date_to = myMT5Pro.get_date_range(timeframe) # 不同时间框架加载的时间范围不同
    data_total = myMT5Pro.getsymboldata(symbol, timeframe, date_from, date_to, index_time=True, col_capitalize=True)
    data_train, data_test = myMT5Pro.get_train_test(data_total, train_scale=0.8)
    # 退出条件
    if strat_break(strat_para): return None
    # !!!获取信号数据(注意这里信号是全数据段)
    signaldata = stratgy_signal(data_total.Close, sig_mode=direct, para_list=strat_para)
    # !!!信号分析(注意这里价格数据是测试集)
    outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signaldata[direct], price_DataFrame=data_test, holding=strat_para[-2], lag_trade=strat_para[-1], plotRet=False, plotStrat=False)
    # 设置信号统计
    result = myBTV.filter_strategy(outStrat, outSignal, para, strategy_para_names)
    return result

#%%
core_num = 7
# ---多进程必须要在这里执行
if __name__ == '__main__':
    # ---主要函数，run_testset是否计算下测试集
    def main_func(core_num, run_testset=False):
        timeframe_list = ["TIMEFRAME_D1","TIMEFRAME_H12","TIMEFRAME_H8","TIMEFRAME_H6",
                          "TIMEFRAME_H4","TIMEFRAME_H3","TIMEFRAME_H2","TIMEFRAME_H1",
                          "TIMEFRAME_M30","TIMEFRAME_M20","TIMEFRAME_M15","TIMEFRAME_M12",
                          "TIMEFRAME_M10","TIMEFRAME_M6","TIMEFRAME_M5","TIMEFRAME_M4",
                          "TIMEFRAME_M3","TIMEFRAME_M2","TIMEFRAME_M1"]
        # ---开始并行运算
        for timeframe in timeframe_list:
            # 已经执行过的则跳过，用于长期运算时出错后不至于重头再算。
            # if timeframe in ["TIMEFRAME_D1"]:
            #     continue
            finish_symbol = []
            for symbol in symbol_list:
                # 已经执行过的则跳过，用于长期运算时出错后不至于重头再算。
                # if timeframe ==  "TIMEFRAME_M1" and symbol in ['AUDCAD']:
                #     finish_symbol.append(symbol)
                #     continue
                # 设置输出目录：one symbol + one timeframe + three direct --> one folder
                folder = total_folder + "\\{}.{}".format(symbol, timeframe)
                # 仅做多、仅做空、多空都做，保存在一个目录下
                for direct in direct_para:
                    # 设定并行参数，只需要指定策略参数的范围即可
                    para_muilt = get_strat_para_scope(direct, timeframe, symbol)
                    # 文件位置
                    filepath = folder + "\\{}_{}.xlsx".format(filename_prefix,direct)
                    # 分析训练集(并行)，会把参数优化结果生成文档。
                    myBTV.muiltcore.run_concat_dataframe(signalfunc_NoRepeatHold_train, para_muilt, filepath, core_num)
                    # 分析测试集(并行)，会内部解析训练集文档中的参数。
                    if run_testset == True:
                        myBTV.muiltcore.run_parse_xlsx(signalfunc_NoRepeatHold_test, filepath, strategy_para_names, [direct,timeframe,symbol], core_num)
                finish_symbol.append(symbol)
                mylogging.warning("finished: {} {}".format(timeframe, finish_symbol))
    # ---
    main_func(core_num,run_testset=False)











