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
myini = MyFile.MyClass_INI()  # ini文件操作类
mytime = MyTime.MyClass_Time()  # 时间类
myparallel = MyTools.MyClass_ParallelCal()  # 并行运算类
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
myMT5Report = MyMql.MyClass_StratTestReport()  # MT5策略报告类
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=False)  # 固定杠杆仓位类
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=False)  # 浮动杠杆仓位类
myMT5run = MyMql.MyClass_RunningMT5()  # Python运行MT5
myMT5code = MyMql.MyClass_CodeMql5()  # Python生成MT5代码
myMoneyM = MyTrade.MyClass_MoneyManage()  # 资金管理类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------

#%%
strat_para_name0 = "k"
in_folder0 = r"F:\工作---Python策略研究\1.简单的动量反转\_动量研究\策略池整合"
out_folder = myMT5code.experts_folder + r"\My_Experts\简单动量策略(AB比较)_Test"
strategy_signal = "Momentum"

#%%
# ---
def autocode():
    # 判断是否存在，不存在则返回
    if __mypath__.path_exists(in_folder0) == False:
        return
    folder0_dir = __mypath__.listdir(in_folder0)
    for foldname in folder0_dir:  # foldname = folder0_dir[0] # foldname = "EURUSD"
        # 如果是文件，不是文件夹，则跳过
        if __mypath__.is_folder_or_file(in_folder0 + "\\" + foldname, check_folder=False):
            continue
        symbol = foldname
        in_folder1 = in_folder0 + "\\" + foldname
        if __mypath__.path_exists(in_folder1) == False:
            continue
        in_file = in_folder1 + "\\{}_strategy_pool.xlsx".format(symbol)
        if __mypath__.path_exists(in_file) == False:
            continue
        # 读取策略池整合的xlsx文件
        filecontent = pd.read_excel(in_file, header=[0,1])
        # 原内容因为 d_mode 所以每两行是一样。
        sameindex = filecontent["original"].duplicated()
        content1 = filecontent[~sameindex]
        content2 = filecontent[sameindex]
        # ---解析，显然没有内容则直接跳过
        for i in range(len(content1)):  # i=0
            # ---获取各个参数
            timeframe = content1.iloc[i]["original", "timeframe"]
            direct = content1.iloc[i]["original", "direct"]
            length0 = content1.iloc[i]["original", strat_para_name0]
            # 范围过滤参数
            rindi_name = content1.iloc[i]["range_filter_only", "indi_name"]
            rindi_para0 = content1.iloc[i]["range_filter_only", "indi_para0"]
            rindi_left = content1.iloc[i]["range_filter_only", "indi_start"]
            rindi_right = content1.iloc[i]["range_filter_only", "indi_end"]
            # 方向过滤参数
            def func_d_mode(content, d_mode_value):
                d_mode = content.iloc[i]["direct_filter_only", "d_mode"]
                dindi_name = content.iloc[i]["direct_filter_only", "indi_name"] if d_mode == d_mode_value else "None"
                dindi_para0 = content.iloc[i]["direct_filter_only", "indi_para0"] if d_mode == d_mode_value else 0
                # "MA"指标要根据方法调整下
                if dindi_name == "MA":
                    if content.iloc[i]["direct_filter_only", "indi_para2"] == "MODE_SMMA":
                        dindi_name = "SMMA"
                    elif content.iloc[i]["direct_filter_only", "indi_para2"] == "MODE_SMA":
                        dindi_name = "SMA"
                return dindi_name, dindi_para0
            dindi1_name, dindi1_para0 = func_d_mode(content1, 1)
            dindi2_name, dindi2_para0 = func_d_mode(content2, 2)

            # ---各过滤各平仓模式
            # 声明为各过滤各平仓模式，要显示的对各变量赋值
            myMT5code.__init__()
            myMT5code.declare_mode_filter_close()

            # ---变量复制
            # ***(需修改)策略参数***
            myMT5code.length0 = length0
            myMT5code.symbol0 = myMT5code.to_mql5string(symbol)
            myMT5code.timeframe0 = myMT5code.timeframe_to_mql5(timeframe) # "PERIOD_D1"
            myMT5code.direct0 = myMT5code.to_mql5string(direct) # "BuyOnly"
            # ***(需修改)范围过滤参数***
            myMT5code.rindi_name = myMT5code.to_mql5string(rindi_name) # "WPR"
            myMT5code.rindi_para0 = rindi_para0
            myMT5code.rindi_left = rindi_left
            myMT5code.rindi_right = rindi_right
            # ***(需修改)方向过滤模式1参数***
            myMT5code.dindi1_name = myMT5code.to_mql5string(dindi1_name) # "TEMA"
            myMT5code.dindi1_para0 = dindi1_para0
            # ***(需修改)方向过滤模式2参数***
            myMT5code.dindi2_name = myMT5code.to_mql5string(dindi2_name) # "VIDYA"
            myMT5code.dindi2_para0 = dindi2_para0

            # ---设置信号策略：
            myMT5code.set_signal_strategy(strat=strategy_signal)

            # ---生成各过滤各平仓模式代码
            myMT5code.code_mode_filter_close()

            # ---输出mq5文件
            filename = symbol + "." + myMT5code.timeframe_to_affix(timeframe) + "." \
                       + myMT5code.direct_to_affix(direct) + "." + "各过滤各平仓.mq5"
            mq5file = out_folder + r"\{}\{}".format(symbol, filename)
            # 会智能化命名，不覆盖存在的。
            myMT5code.write_mq5(mq5file=mq5file, autoname=True)
# ---
autocode()
