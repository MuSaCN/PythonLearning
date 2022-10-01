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
myplthtml = MyPlot.MyClass_PlotHTML()  # 画可以交互的html格式的图
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
myMT5Report = MyMT5Report.MyClass_StratTestReport(AddFigure=False)  # MT5策略报告类
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=False)  # 固定杠杆仓位类
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=False)  # 浮动杠杆仓位类
myMT5run = MyMql.MyClass_RunningMT5()  # Python运行MT5
myMT5code = MyMql.MyClass_CodeMql5()  # Python生成MT5代码
myMoneyM = MyTrade.MyClass_MoneyManage()  # 资金管理类
# myDefault.set_backend_default("agg")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------
# Jupyter Notebook 控制台显示必须加上：%matplotlib inline ，弹出窗显示必须加上：%matplotlib auto
# %matplotlib inline
# import warnings
# warnings.filterwarnings('ignore')

#%%
'''
注意信号是 Cross 还是 Momentum，若要修改需要到 EA 中修改。
'''
import warnings
warnings.filterwarnings('ignore')
myDefault.set_backend_default("agg")  # 设置图片输出方式，这句必须放到类下面.
plt.show()

#%% ###### 通用参数 ######
symbol_list = myMT5Pro.get_mainusd_symbol_name_list()
timeframe_list = ["TIMEFRAME_D1","TIMEFRAME_H12","TIMEFRAME_H8","TIMEFRAME_H6",
                  "TIMEFRAME_H4","TIMEFRAME_H3","TIMEFRAME_H2","TIMEFRAME_H1",
                  "TIMEFRAME_M30","TIMEFRAME_M20","TIMEFRAME_M15","TIMEFRAME_M12",
                  "TIMEFRAME_M10","TIMEFRAME_M6","TIMEFRAME_M5","TIMEFRAME_M4",
                  "TIMEFRAME_M3","TIMEFRAME_M2","TIMEFRAME_M1"]
timeframe_list = ["TIMEFRAME_M10","TIMEFRAME_M6","TIMEFRAME_M5"]


def muiltPinbar(symbol, timeframe): # symbol=symbol_list[0] ; timeframe="TIMEFRAME_H4"
    if timeframe == "TIMEFRAME_M10" and symbol == "EURUSD":
        return


    experfolder = "My_Experts\\Strategy\\K线形态CTA"
    expertfile = "Pinbar裸K策略.ex5"
    expertname = experfolder + "\\" + expertfile
    fromdate = "2010.01.01"
    todate = "2020.01.01"
    symbol = symbol # "EURUSD"
    timeframe = timeframe # "TIMEFRAME_H4"
    totalfolder = r"F:\工作(同步)\工作---MT5策略研究\Pinbar裸K策略"
    reportfolder = totalfolder + "\\{}.{}\\{}".format(symbol, timeframe, expertfile.rsplit(sep=".", maxsplit=1)[0])

    ###### Step1.0 无筛选.固定持仓=1单次回测 ######
    fixedholding = 1
    # 单一测试不需要.xml后缀
    reportfile = reportfolder + "\\1.a.无筛选.Fixed={}".format(fixedholding)
    optimization = 0 # 0 禁用优化, 1 "慢速完整算法", 2 "快速遗传算法", 3 "所有市场观察里选择的品种"
    # ---
    myMT5run.__init__()
    myMT5run.config_Tester(expertname, symbol, timeframe, fromdate=fromdate, todate=todate,
                           delays=0, optimization=optimization, reportfile=reportfile)

    def PinbarSetting():
        # ======Pinbar指标参数======
        myMT5run.input_set("Inp_CombinMax", "3||1||1||3||N")
        myMT5run.input_set("Inp_RiFaCompare", "true||false||0||true||N")
        myMT5run.input_set("Inp_RStatPeriod", "100||100||1||1000||N")
        myMT5run.input_set("Inp_RQuantile", "0.5||0.5||0.05||5.0||N")
        # ======Pinbar必要筛选======
        myMT5run.input_set("IsConti", "true||false||0||true||N")
        myMT5run.input_set("IsExtrema", "true||false||0||true||N")
        # ======Pinbar细节筛选======
        myMT5run.input_set("NeedDetail", "0||0||1||4||Y")
        myMT5run.input_set("IsSizeLarge", "false||false||0||true||Y")
        myMT5run.input_set("IsFalseBreak", "false||false||0||true||Y")
        myMT5run.input_set("IsEyeBody", "false||false||0||true||Y")
        myMT5run.input_set("IsEyeRange", "false||false||0||true||Y")
        myMT5run.input_set("IsFitTrend", "false||false||0||true||Y")
    def CommonSetting():
        # ======(通用)用于分析======
        myMT5run.input_set("CustomMode", "0") # 设置自定义的回测结果 0-TB, 42-最大连亏, 4-SQN_MT5_No
        # ------1.固定持仓------
        myMT5run.input_set("FixedHolding", "%s||1||1||10||N"%fixedholding) # 0不是固定持仓模式，>0固定周期持仓
        # ------2.信号过滤------
        myMT5run.input_set("FilterMode", "0") # 0-NoFilter, 1-Range, 2-TwoSide
        myMT5run.input_set("FilterIndiName", "过滤指标名称") # 过滤指标名称
        myMT5run.input_set("FilterIndiTF", "TIMEFRAME_H1") # 过滤指标时间框字符串
        myMT5run.input_set("FilterIndiPara0", "0") # 过滤指标首个参数
        myMT5run.input_set("FilterLeftValue", "0") # 过滤指标左侧的值
        myMT5run.input_set("FilterRightValue", "0") # 过滤指标右侧的值
        # ------3.止损止盈------
        # 3.1 止损设置
        myMT5run.input_set("Init_SLMode", "0") # 设置初始止损模式
        myMT5run.input_set("SL_Point", "100||100||100||1000||N") # SLMode_POINT模式：指定止损点.
        myMT5run.input_set("SL_PreBar", "1||1||1||3||Y") # SLMode_BAR模式：信号前的bar数量.
        myMT5run.input_set("SL_atr_Period", "7||7||1||70||N") # SLMode_ATR模式：止损ATR周期.
        myMT5run.input_set("SL_atr_N", "3||3||0.3||30||N") # SLMode_ATR模式：ATR倍数.
        myMT5run.input_set("SL_Adjust", "100||20||20||100||Y") # SLMode_*模式：调节点数.
        # 3.2 止盈设置
        myMT5run.input_set("Init_TPMode", "0") # 设置初始止盈模式
        myMT5run.input_set("TP_Point", "0||0||1||10||N") # TPMode_POINT模式：0表示没有.
        myMT5run.input_set("TP_SLMultiple", "2.0||1.0||0.2||2.0||Y") # TPMode_PnLRatio模式：止损盈亏比.
        # ------4.直接交易或挂单交易------
        myMT5run.input_set("Is_DirectTrade", "true||false||0||true||N") # Is_DirectTrade=true直接进场；false挂单进场.
        myMT5run.input_set("Pending_PreBar", "1||1||1||10||N") # 挂单：在之前的N根极值处挂单
        myMT5run.input_set("Pending_Adjust", "20||20||20||100||Y") # 挂单：以点数修正下挂单位置
        myMT5run.input_set("Pending_ExpireTF", "0||0||0||49153||N") # 挂单：挂单有效的时间框
        myMT5run.input_set("Pending_ExpireBar", "3||1||1||5||Y") # 挂单：挂单有效的Bar个数
        # ------5.重复入场------
        myMT5run.input_set("Is_ReSignal", "true") # true允许信号重复入场，false不允许信号重复入场。

    PinbarSetting()
    CommonSetting()
    # ---检查参数输入是否匹配优化的模式，且写出配置结果。
    myMT5run.check_inputs_and_write()
    myMT5run.run_MT5()

    ###### Step2.0 通用过滤：范围过滤和两侧过滤 ######
    core_num = -1
    tf_indi = timeframe # 过滤指标的时间框 timeframe "TIMEFRAME_H1" "TIMEFRAME_M30"

    # ====== 操作都默认从桌面操作 ======
    # ---把 .htm 文件复制到桌面 通用过滤.htm
    filepath2 = reportfile + ".htm" # file = reportfolder + "\\1.b.信号=100.0.Fixed=1.htm"
    filehtm = __mypath__.get_desktop_path() + r"\通用过滤.htm"
    myfile.copy_dir_or_file(source=filepath2, destination=filehtm, DirRemove=True)

    # ---输出参数csv到项目目录和桌面
    dfpara = []
    dfpara.append(["filepath",filepath2])
    dfpara.append(["direct","All"])
    dfpara.append(["filtermode","-1"])
    dfpara.append(["tf_indi",tf_indi])
    dfpara.append(["core_num",core_num])
    dfpara = pd.DataFrame(dfpara)
    dfpara.set_index(keys=0,drop=True,inplace=True)
    # 添加到指定目录
    outfile = reportfolder + r"\2.无筛选.Fixed={}.通用过滤参数.csv".format(fixedholding)
    dfpara.to_csv(outfile, sep=";")
    # 添加到桌面，从桌面加载
    outdesktopfile = __mypath__.get_desktop_path() + r"\通用过滤参数.csv"
    dfpara.to_csv(outdesktopfile, sep=";")
    # 休息
    import time
    time.sleep(1)
    print("通用过滤参数输出完成！")

    # ---需要 run 中运行，ipython中不行。
    myDefault.set_backend_default("agg")
    FilterScript = __mypath__.get_user_path()+r"\PycharmProjects\PythonLearning\Project_Python驱动MT5回测\CommonScript\自动MT5reportFilter.py"
    import os
    os.system("python "+FilterScript)
    time.sleep(1)
    print("通用过滤执行完成！")

    # ---剪切桌面的结果到项目目录 reportfolder
    # 移动桌面 通用过滤.range 通用过滤.2side 到项目目录
    filterfolder1 = __mypath__.get_desktop_path() + "\\通用过滤.range"
    tofilterfolder1 = reportfolder + "\\2.无筛选.Fixed={}.通用过滤.range".format(fixedholding)
    filterfolder2 = __mypath__.get_desktop_path() + "\\通用过滤.2side"
    tofilterfolder2 = reportfolder + "\\2.无筛选.Fixed={}.通用过滤.2side".format(fixedholding)
    if __mypath__.path_exists(filterfolder1):
        myfile.copy(src=filterfolder1,dst=tofilterfolder1,cover=True)
    else:
        print("{}不存在！！！".format(filterfolder1))
    if __mypath__.path_exists(filterfolder2):
        myfile.copy(src=filterfolder2,dst=tofilterfolder2,cover=True)
    else:
        print("{}不存在！！！".format(filterfolder2))
    #
    time.sleep(3)
    print("移动桌面 通用过滤.range 通用过滤.2side 到项目目录 %s"%reportfolder)
    # 必须删除 filehtm, outdesktopfile, filterfolder1, filterfolder2
    if __mypath__.path_exists(filehtm):
        myfile.remove_dir_or_file(filehtm)
    else:
        print("{}不存在！！！".format(filehtm))
    if __mypath__.path_exists(outdesktopfile):
        myfile.remove_dir_or_file(outdesktopfile)
    else:
        print("{}不存在！！！".format(outdesktopfile))
    if __mypath__.path_exists(filterfolder1):
        myfile.remove_dir_or_file(filterfolder1, onlyContent=False)
    else:
        print("{}不存在！！！".format(filterfolder1))
    if __mypath__.path_exists(filterfolder2):
        myfile.remove_dir_or_file(filterfolder2, onlyContent=False)
    else:
        print("{}不存在！！！".format(filterfolder2))
    time.sleep(1)
    print("删除桌面 {}, {}, {}, {}".format(filehtm, outdesktopfile, filterfolder1, filterfolder2))
    # 关闭 MT5 进程
    import os
    os.system("TASKKILL /F /IM terminal64.exe") # os.system("taskkill /F /IM 进程名")
    time.sleep(5)


for timeframe in timeframe_list:
    for symbol in symbol_list:
        muiltPinbar(symbol, timeframe)
        print(symbol, timeframe, "finished!")



