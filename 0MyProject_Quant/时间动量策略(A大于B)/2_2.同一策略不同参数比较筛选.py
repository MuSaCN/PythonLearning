# Author:Zhang Yuan
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
# ------------------------------------------------------------

#%%
strategy_para_name = ["k", "holding", "lag_trade"]
symbol_list = myMT5Pro.get_main_symbol_name_list()
total_folder = "F:\\工作---策略研究\\简单的动量反转\\_动量研究"
readfile_suffix = ".original" # 输入的文档加后缀 .holdingtest
core_num = -1

#  策略的当期信号(不用平移)：para_list策略参数，默认-1为lag_trade，-2为holding。
def stratgy_signal(dataframe, para_list=list or tuple, stra_mode="Continue"):
    price = dataframe["Close"]
    return myBTV.stra.momentum(price=price, k=para_list[0], stra_mode=stra_mode)
stratgy_signal = stratgy_signal

#%%
para = ("EURUSD",)
symbol = para[0]
in_file = total_folder + "\\策略参数自动选择\\{}\\{}.total.{}{}.xlsx".format(symbol, symbol, "filter1", readfile_suffix)  # 固定只分析 filter1
out_folder = __mypath__.dirname(in_file, 0)  # "...\\策略参数自动选择\\EURUSD"
filecontent = pd.read_excel(in_file)
# ---
timeframe_list = ["TIMEFRAME_D1", "TIMEFRAME_H12", "TIMEFRAME_H8", "TIMEFRAME_H6",
                  "TIMEFRAME_H4", "TIMEFRAME_H3", "TIMEFRAME_H2", "TIMEFRAME_H1",
                  "TIMEFRAME_M30", "TIMEFRAME_M20", "TIMEFRAME_M15","TIMEFRAME_M12",                        "TIMEFRAME_M10", "TIMEFRAME_M6", "TIMEFRAME_M5", "TIMEFRAME_M4",
                  "TIMEFRAME_M3", "TIMEFRAME_M2", "TIMEFRAME_M1"]
direct_list = ["BuyOnly", "SellOnly"]
for timeframe in timeframe_list: # timeframe = "TIMEFRAME_H1"
    for direct in direct_list: # direct = "SellOnly"
        index = (filecontent["timeframe"] == timeframe) & (filecontent["direct"] == direct)
        samestrat = filecontent[index]
        if len(samestrat) <= 1: # 没有或只有1个则跳过
            continue
        # 准备数据
        date_from, date_to = myMT5Pro.get_date_range(timeframe)
        data_total = myMT5Pro.getsymboldata(symbol, timeframe, date_from, date_to, index_time=True, col_capitalize=True)
        data_train, data_test = myMT5Pro.get_train_test(data_total, train_scale=0.8)
        # ---展开holding参数
        holding_test = [holding for holding in range(1, 20 + 1)]
        label_list = ["sharpe", "winRate", "maxDD", "cumRet", "calmar_ratio", "SQN"]
        df = pd.DataFrame([], index=holding_test)
        #%%
        for i in range(len(samestrat)):  # i=0
            # df增加列
            for label in label_list:
                df[label+"_"+str(i)] = np.nan # 不重复持仓列
                df[label + "_re" + str(i)] = np.nan # 重复持仓列
            # 策略参数
            strat_para = [samestrat.iloc[i][strategy_para_name[j]] for j in range(len(strategy_para_name))]
            # 满足3个标的指标都是递增才输出路径
            suffix = myBTV.string_strat_para(strategy_para_name, strat_para)
            # ---填充df数据
            for holding in holding_test:  # holding=1
                # 策略参数更换 -2位置的holding参数
                para_list = strat_para[0:-2] + [holding] + [strat_para[-1]]
                # 获取信号数据
                signal = stratgy_signal(data_train, para_list=para_list)
                # 信号分析，无重复持仓模式: signal_quality_NoRepeatHold / signal_quality
                outStrat_NoRe, outSignal_NoRe = myBTV.signal_quality_NoRepeatHold(signal=signal[direct], price_DataFrame=data_train, holding=holding, lag_trade=para_list[-1], plotStrat=False)
                # 信号分析，可重复持仓模式：
                outStrat_Re, outSignal_Re = myBTV.signal_quality(signal=signal[direct], price_DataFrame=data_train, holding=holding, lag_trade=para_list[-1], plotStrat=False)
                # 赋值
                for label in label_list:  # label = label_list[0]
                    df.loc[holding][label + "_" + str(i)] = outStrat_NoRe[direct][label]
                    df.loc[holding][label + "_re" + str(i)] = outStrat_Re[direct][label]
        #%%
        # ---画图筛选
        for label in label_list:
            for i in range(len(samestrat)):
                ax = plt.gca()
                df[label + "_" + str(i)].plot(ax = ax)
            plt.legend()
            plt.show()
            for i in range(len(samestrat)):
                ax = plt.gca()
                df[label + "_re" + str(i)].plot(ax = ax)
            plt.legend()
            plt.show()
        # ---(淘汰，混乱)顺序图，需排序
        # lab1,lab2 = "winRate","maxDD" # lab1为排序词缀
        # for i in range(len(samestrat)):
        #     dfsort = df.sort_values(by = lab1+"_"+str(i))
        #     myplt.plot(dfsort[lab1+"_"+str(i)], dfsort[lab2+"_"+str(i)], objectname=str(i), show=False, PlotLabel=["",lab1,lab2])
        # plt.show()
        # for i in range(len(samestrat)):
        #     dfsort = df.sort_values(by = lab1+"_re"+str(i))
        #     myplt.plot(dfsort[lab1+"_re"+str(i)], dfsort[lab2+"_re"+str(i)], objectname=str(i), show=False, PlotLabel=["",lab1,lab2])
        # plt.show()










