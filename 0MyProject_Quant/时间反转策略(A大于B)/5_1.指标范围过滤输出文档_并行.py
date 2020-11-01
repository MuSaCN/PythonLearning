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
myPjMT5 = MyProject.MT5_MLLearning()  # MT5机器学习项目类
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
def run_range_filter_result(para):
    print("\r", "当前执行参数为：", para, end="", flush=True)
    # para = ('Close', 135, 'roc', [314, 1, 1], 'SellOnly', 'TIMEFRAME_H1', 'AUDNZD')
    symbol = para[-1]
    timeframe = para[-2]
    direct = para[-3]
    [k, holding, lag_trade] = para[-4]
    indi_name = para[-5]
    indi_para = para[0:-5]  # ("Close", 30)

    # ---获取数据
    date_from, date_to = myPjMT5.get_date_range(timeframe, to_Timestamp=True)
    data_total = myPjMT5.getsymboldata(symbol, timeframe, date_from, date_to, index_time=True, col_capitalize=True)
    # 由于信号利润过滤是利用训练集的，所以要区分训练集和测试集
    data_train, data_test = myPjMT5.get_train_test(data=data_total, train_scale=0.8)

    # ---获取训练集和整个样本的信号
    # 获取训练集的信号 ******(修改这里)******
    signaldata_train = myBTV.stra.momentum(data_train.Close, k=k, holding=holding, sig_mode=direct,stra_mode="Reverse")
    signal_train = signaldata_train[direct]

    # ---(核心，在库中添加)获取指标
    indicator = myBTV.indi.get_oscillator_indicator(data_total, indi_name, indi_para)

    # ---信号利润范围过滤及测试
    result = myBTV.signal_range_filter_and_quality(signal_train=signal_train, signal_all=signal_train, indicator=indicator, price_DataFrame=data_total, price_Series=data_total.Close, holding=1, lag_trade=1, noRepeatHold=True, indi_name=indi_name, indi_para=indi_para)
    return result


#%%
core_num = -1
if __name__ == '__main__':
    # 策略参数名称，用于文档中解析参数 ***修改这里***
    strategy_para_name = ["k", "holding", "lag_trade"]
    symbol_list = myPjMT5.get_all_symbol_name().tolist()
    # ---
    finish_symbol = []
    for symbol in symbol_list: # symbol = "EURUSD"
        # if symbol in ['EURUSD', 'GBPUSD']:
        #     finish_symbol.append(symbol)
        #     continue

        # ---定位文档 ******修改这里******
        in_file = __mypath__.get_desktop_path() + "\\_反转研究\\策略参数自动选择\\{}\\{}.total.{}.xlsx".format(symbol, symbol, "filter1")   # 固定只分析 filter1
        filecontent = pd.read_excel(in_file)
        # ---解析，显然没有内容则直接跳过
        for i in range(len(filecontent)):  # i=0
            # ---解析文档
            # 获取各参数
            timeframe = filecontent.iloc[i]["timeframe"]
            direct = filecontent.iloc[i]["direct"]
            # 策略参数 ******修改这里******
            k = filecontent.iloc[i][strategy_para_name[0]]
            holding = filecontent.iloc[i][strategy_para_name[1]]
            lag_trade = filecontent.iloc[i][strategy_para_name[2]]
            strat_para = [k, holding, lag_trade]

            # 过滤下参数，反转策略，如果k太小，交易频率太高无意义。 ******修改这里******
            # if k in [1, 2, 3, 4]:
            #     continue
            # 过滤规则为：只有主力品种才全部检测，其他品种只检测大的时间框。
            if symbol not in ["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCAD","USDCHF","XAUUSD","XAGUSD"]:
                if timeframe not in ["TIMEFRAME_D1","TIMEFRAME_H12","TIMEFRAME_H8","TIMEFRAME_H6","TIMEFRAME_H4","TIMEFRAME_H3","TIMEFRAME_H2","TIMEFRAME_H1"]:
                    continue


            # 输出的文档路径
            suffix = myBTV.string_strat_para(strategy_para_name, strat_para)
            # ******修改这里******
            out_file = __mypath__.get_desktop_path() + "\\_反转研究\\范围指标参数自动选择\\{}.{}".format(symbol, timeframe) + "\\{}.{}.xlsx".format( direct, suffix)
            # ---设定并行参数，分别设定再合并
            rsi_params = [("Close", i) + ("rsi", strat_para, direct, timeframe, symbol) for i in range(5, 144 + 1)]
            roc_params = [("Close", i) + ("roc", strat_para, direct, timeframe, symbol) for i in range(5, 144 + 1)]
            multi_params = rsi_params + roc_params
            # ---开始多核执行
            myBTV.run_concat_dataframe(run_range_filter_result, multi_params, filepath=out_file, core_num=core_num)
            print("para finished:", symbol, timeframe, direct, suffix)
        # ---记录对应时间框下完成的品种
        finish_symbol.append(symbol)
        mylogging.warning("symbol finished: {}".format(finish_symbol))












