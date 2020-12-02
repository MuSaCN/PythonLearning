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
myMT5Pro = MyMql.MyClass_ConnectMT5Pro(connect = False) # Python链接MT5高级类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------


'''
# 订单可管理性：如果一个策略在未来1期持仓表现不错，同时在未来多期持仓也表现不错。这就表明，这个策略的交易订单在时间伸展上能够被管理，我们称作为订单具备可管理性。
# 对训练集进行多holding回测，展示结果的夏普比曲线和胜率曲线。
# 采用无重复持仓模式和重复持仓模式。
# 并行运算以品种来并行
'''

#%%
# 策略参数名称，用于文档中解析参数 ******修改这里******
strategy_para_name = ["k", "holding", "lag_trade"]

#%%
# ---并行执行订单可管理性分析
def run_holding_extend(para):
    symbol = para[0]  # symbol = "EURUSD"
    print("%s 开始订单可管理性分析..." % symbol)
    # ---定位策略参数自动选择文档，获取各组参数 ******修改这里******
    total_folder = "F:\\工作---策略研究\\简单的动量反转" + "\\_反转研究"
    pool_file = total_folder + "\\策略池整合\\{}\\{}_strategy_pool.xlsx".format(symbol, symbol)  # 固定只分析 filter1
    pool_filecontent = pd.read_excel(pool_file, header=[0,1]) # 多层表头
    pool_filecontent = pool_filecontent["original"] if len(pool_filecontent) > 0 else pool_filecontent # 定位到无过滤内容
    for i in range(len(pool_filecontent)):  # i=0
        # ---解析文档
        # 获取各参数
        timeframe = pool_filecontent.iloc[i]["timeframe"]
        direct = pool_filecontent.iloc[i]["direct"]
        # 策略参数 ******修改这里******
        k = pool_filecontent.iloc[i][strategy_para_name[0]]
        holding = pool_filecontent.iloc[i][strategy_para_name[1]]
        lag_trade = pool_filecontent.iloc[i][strategy_para_name[2]]
        strat_para = [k, holding, lag_trade]
        # 输出的路径
        suffix = myBTV.string_strat_para(strategy_para_name, strat_para)
        out_folder = __mypath__.dirname(pool_file) + "\\{}.{}.{}".format(timeframe, direct, suffix)
        __mypath__.makedirs(out_folder, exist_ok=True)

        # ---准备数据
        date_from, date_to = myMT5Pro.get_date_range(timeframe)
        data_total = myMT5Pro.getsymboldata(symbol, timeframe, date_from, date_to, index_time=True, col_capitalize=True)
        data_train, data_test = myMT5Pro.get_train_test(data_total, train_scale=0.8)
        # 展开holding参数
        holding_range = [holding for holding in range(1, 20 + 1)]

        # ---策略训练集多holding回测，选择夏普比和胜率来分析，下面的信号质量计算是否重复持仓都要分析。重复持仓主要看胜率。
        label1, label2 = "sharpe", "winRate" # 让sharpe为红色，更明显
        out_list1_NoRe, out_list2_NoRe, out_list1_Re, out_list2_Re = [] ,[] ,[], []
        # ---
        for holding in holding_range:
            # 获取信号数据 ******修改这里******
            signal = myBTV.stra.momentum(data_train.Close, k=k, holding=holding, sig_mode=direct, stra_mode="Reverse")
            # 信号分析，无重复持仓模式: signal_quality_NoRepeatHold / signal_quality
            outStrat_NoRe, outSignal_NoRe = myBTV.signal_quality_NoRepeatHold(signal=signal[direct],price_DataFrame=data_train, holding=holding,lag_trade=lag_trade, plotStrat=False,)
            out_list1_NoRe.append(outStrat_NoRe[direct][label1])
            out_list2_NoRe.append(outStrat_NoRe[direct][label2])
            # 信号分析，可重复持仓模式：
            outStrat_Re, outSignal_Re = myBTV.signal_quality(signal=signal[direct],price_DataFrame=data_train,holding=holding, lag_trade=lag_trade,plotStrat=False, )
            out_list1_Re.append(outStrat_Re[direct][label1])
            out_list2_Re.append(outStrat_Re[direct][label2])
        # ---
        myfig.__init__(nrows=1,ncols=11,figsize=[1300,600],GridSpec=["[0:4]","[6:-1]"],AddFigure=True)
        myplt.plot_twoline_twinx(out_list1_NoRe, out_list2_NoRe, label1, label2, color1= "r", color2="b", ax=myfig.axeslist[0],title="NoRepeatHold", show=False)
        myplt.plot_twoline_twinx(out_list1_Re, out_list2_Re, label1, label2, color1= "r", color2="b", ax=myfig.axeslist[1],title="RepeatHold", show=False)
        pic_path = out_folder + "\\订单可管理性.png"
        myfig.savefig(pic_path)
        # 并行运算释放内存
        myfig.close(check=False)
        plt.close()
        plt.show()
        del data_total, data_train, data_test, out_list1_NoRe, out_list2_NoRe, out_list1_Re, out_list2_Re
        # 打印进度
        print("finished:",symbol,timeframe,direct,suffix)
    print(symbol, "finished!!!")
    mylogging.warning("symbol finished: {}".format(symbol))

#%%
core_num = -1
if __name__ == '__main__':
    symbol_list = myMT5Pro.get_all_symbol_name().tolist()
    mylogging.warning("symbol_list: {}".format(symbol_list))
    # finished_symbol = []
    # for symbol in symbol_list:
    #     run_strategy_pool((symbol,))
    #     finished_symbol.append(symbol)
    #     print(finished_symbol)
    para_muilt = [(symbol,) for symbol in symbol_list]
    import timeit
    # ---开始多核执行
    t0 = timeit.default_timer()
    myBTV.muiltcore.multi_processing(run_holding_extend, para_muilt, core_num=core_num)
    t1 = timeit.default_timer()
    print("\n", ' 耗时为：', t1 - t0)


