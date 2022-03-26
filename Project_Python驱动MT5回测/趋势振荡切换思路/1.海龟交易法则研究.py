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
myplthtml = MyPlot.MyClass_PlotHTML() # 画可以交互的html格式的图
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
experfolder = "My_Experts\\Strategy走势分类研究\海龟交易法则趋势振荡分类"
expertfile = "1.海龟交易法则研究.Trend.ex5"
expertname = experfolder + "\\" + expertfile
fromdate = "2010.01.01"
todate = "2017.01.01"
symbol = "EURUSD"
timeframe = "TIMEFRAME_M5"
totalfolder = r"F:\工作(同步)\工作---MT5策略研究\趋势振荡分类研究_海龟交易法则_Momentum"
reportfolder = totalfolder + "\\{}.{}\\{}".format(symbol, timeframe, expertfile.rsplit(sep=".", maxsplit=1)[0])


#%% ###### Step1.0 优化信号参数和固定持仓 ######
reportfile = reportfolder + "\\1.opt信号固定持仓.xml"
optimization = 1 # 0 禁用优化, 1 "慢速完整算法", 2 "快速遗传算法", 3 "所有市场观察里选择的品种"
# ---
myMT5run.__init__()
myMT5run.config_Tester(expertname, symbol, timeframe, fromdate=fromdate, todate=todate,
                       delays=0, optimization=optimization, reportfile=reportfile)

myMT5run.input_set("Inp_ChannelPeriod", "24||5||1||100||Y")
# ======(通用)用于分析======
myMT5run.input_set("CustomMode", "0") # 设置自定义的回测结果 0-TB, 42-最大连亏
myMT5run.input_set("backtestmode", "0") # 0-FitnessPerformance
# ------1.固定持仓------
myMT5run.input_set("FixedHolding", "0||1||1||10||Y") # 0表示不是固定持仓模式，>0表示固定周期持仓。
# ------2.信号过滤------
myMT5run.input_set("FilterMode", "0") # 0-NoFilter, 1-Range, 2-TwoSide
myMT5run.input_set("FilterIndiName", "过滤指标名称") # 过滤指标名称
myMT5run.input_set("FilterIndiTF", "TIMEFRAME_H1") # 过滤指标时间框字符串
myMT5run.input_set("FilterIndiPara0", "0") # 过滤指标首个参数
myMT5run.input_set("FilterLeftValue", "0") # 过滤指标左侧的值
myMT5run.input_set("FilterRightValue", "0") # 过滤指标右侧的值
# ------3.止盈止损------
myMT5run.input_set("InitTPPoint", "0||200||100||2000||N")
myMT5run.input_set("SL_Init", "false") # false表示没有初期止损，true表示有初期止损。
# ---止损设置 SL_Init=true 启用---
myMT5run.input_set("atr_Period", "5||3||1||36||N") # 止损ATR周期.
myMT5run.input_set("atr_N", "1.0||0.5||0.2||3.0||N") # ATR倍数.
# ------4.重复入场------
myMT5run.input_set("Is_ReSignal", "true") # true允许信号重复入场，false不允许信号重复入场。

# ---检查参数输入是否匹配优化的模式，且写出配置结果。
myMT5run.check_inputs_and_write()
myMT5run.run_MT5()


#%% ###### Step1.1 找寻随着持仓周期增加策略表现递增的信号参数 ######
filepath = reportfolder+r"\1.a.3D信号固定持仓.html" # 输出3D图的位置
fixedholding = 10 # 选择固定持仓的周期，有时候小时间框需要固定持仓大一点的

# ---
# 读取优化opt结果
opt = myMT5Report.read_opt_xml(reportfile)
columns = opt.columns
columns = columns.insert(loc=columns.get_loc("Sharpe Ratio")+1, item="Sqn") # 插入到指定位置
# 由于时间是一样的，所以不需要考虑年化交易数量
opt["Sqn_No"] = opt["Sharpe Ratio"] * np.power(opt["Trades"], 0.5)
# 必须要重新排列下
opt = opt[columns.tolist()]

# ---分别画3D图
chart_list, tab_name_list = [], []
for name in ['Profit', 'Expected Payoff', 'Profit Factor', 'Recovery Factor', 'Sharpe Ratio', 'Sqn_No', 'Custom', 'Equity DD %', 'Trades']:
    # 参数1为信号参数，排除 "FixedHolding" 后剩下的；参数2为固定持仓；参数Z为策略表现；
    para1,para2,paraZ = opt.columns[-2:].drop("FixedHolding")[0], "FixedHolding",  name
    data3D = opt[[para1,para2,paraZ]]
    # 画3D图
    surface3D = myplthtml.plot_surface3D(data=data3D, height=100, series_name=paraZ, title=paraZ, savehtml = None)
    chart_list.append(surface3D)
    tab_name_list.append(name)
# 输出到html
tab = myplthtml.plot_tab_chart(chart_list=chart_list,tab_name_list=tab_name_list,savehtml=filepath)
# import os
# os.startfile(filepath)


# ---选择固定持仓，且交易数量平均每天1次的
para1 = opt.columns[-2:].drop("FixedHolding")[0]
opt1 = opt[(opt["FixedHolding"]==fixedholding) & (opt["Trades"]>=250*7)]
opt1.set_index(keys=para1, drop=False, inplace=True)

# ---分别输出各策略结果的卡尔曼过滤选择结果
myDefault.set_backend_default("agg") # 后台输出图片
totalindex = []
for name in ['Profit', 'Expected Payoff', 'Profit Factor', 'Recovery Factor', 'Sharpe Ratio', "Sqn_No", 'Custom']: # name = "Expected Payoff"
    # 有时候夏普比会都为-5
    if len(opt1[name].unique())==1:
        continue
    # 输出结果
    index, values, ax = myDA.plot_kalman_and_extrema(array = opt1[name], arrayX = opt1[para1], restore_nan=False, comparator=np.greater_equal, order=30, filterlevel=1, ylabel = name,savefig=reportfolder+"\\1.a.信号{}期.{}.jpg".format(fixedholding, name), batch=True, show=True)
    # 特殊结果占权重两个
    indexlist = index[0].tolist() * 2 if name in ["Sharpe Ratio", "Custom", "Sqn_No"] else index[0].tolist()
    totalindex = totalindex + indexlist

# ---选择考虑权重后的均值距离最近的作为信号参数1，距离算法保证了结果只有一个。
mean = pd.Series(totalindex).mean() #
dist = np.abs(totalindex - mean)
signalpara1 = totalindex[dist.argmin()] # signalpara1 = 17.0




#%% ###### Step1.2 单独一次回测 ######
""
# 输出结果，不需要.xml后缀
reportfile_1b = reportfolder + "\\1.b.信号={}.Fixed={}".format(signalpara1, fixedholding)
optimization = 0 # 0 禁用优化, 1 "慢速完整算法", 2 "快速遗传算法", 3 "所有市场观察里选择的品种"
# ---
myMT5run.__init__()
myMT5run.config_Tester(expertname, symbol, timeframe, fromdate=fromdate, todate=todate,
                       delays=0, optimization=optimization, reportfile=reportfile_1b)

myMT5run.input_set("Inp_ChannelPeriod", "{}||5||1||100||N".format(signalpara1))
# ======(通用)用于分析======
myMT5run.input_set("CustomMode", "0") # 设置自定义的回测结果 0-TB, 42-最大连亏
myMT5run.input_set("backtestmode", "0") # 0-FitnessPerformance
# ------1.固定持仓------
myMT5run.input_set("FixedHolding", "1||1||1||10||N") # 0表示不是固定持仓模式，>0表示固定周期持仓。
# ------2.信号过滤------
myMT5run.input_set("FilterMode", "0") # 0-NoFilter, 1-Range, 2-TwoSide
myMT5run.input_set("FilterIndiName", "过滤指标名称") # 过滤指标名称
myMT5run.input_set("FilterIndiTF", "TIMEFRAME_H1") # 过滤指标时间框字符串
myMT5run.input_set("FilterIndiPara0", "0") # 过滤指标首个参数
myMT5run.input_set("FilterLeftValue", "0") # 过滤指标左侧的值
myMT5run.input_set("FilterRightValue", "0") # 过滤指标右侧的值
# ------3.止盈止损------
myMT5run.input_set("InitTPPoint", "0||200||100||2000||N")
myMT5run.input_set("SL_Init", "false") # false表示没有初期止损，true表示有初期止损。
# ---止损设置 SL_Init=true 启用---
myMT5run.input_set("atr_Period", "5||3||1||36||N") # 止损ATR周期.
myMT5run.input_set("atr_N", "1.0||0.5||0.2||3.0||N") # ATR倍数.
# ------4.重复入场------
myMT5run.input_set("Is_ReSignal", "true") # true允许信号重复入场，false不允许信号重复入场。


# ---检查参数输入是否匹配优化的模式，且写出配置结果。
myMT5run.check_inputs_and_write()
myMT5run.run_MT5()



#%% ###### Step2.0 通用过滤：范围过滤和两侧过滤 ######
core_num = -1
tf_indi = timeframe # 过滤指标的时间框 timeframe "TIMEFRAME_H1" "TIMEFRAME_M30"

# ====== 操作都默认从桌面操作 ======
# ---把 .htm 文件复制到桌面 通用过滤.htm
filepath2 = reportfile_1b + ".htm" # file = reportfolder + "\\1.b.信号=100.0.Fixed=1.htm"
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
outfile = reportfolder + r"\2.信号={}.Fixed={}.通用过滤参数.csv".format(signalpara1,fixedholding)
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
tofilterfolder1 = reportfolder + "\\2.信号={}.Fixed={}.通用过滤.range".format(signalpara1,fixedholding)
filterfolder2 = __mypath__.get_desktop_path() + "\\通用过滤.2side"
tofilterfolder2 = reportfolder + "\\2.信号={}.Fixed={}.通用过滤.2side".format(signalpara1,fixedholding)
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
# 删除 filehtm, outdesktopfile
# if __mypath__.path_exists(filehtm):
#     myfile.remove_dir_or_file(filehtm)
# else:
#     print("{}不存在！！！".format(filehtm))
# if __mypath__.path_exists(outdesktopfile):
#     myfile.remove_dir_or_file(outdesktopfile)
# else:
#     print("{}不存在！！！".format(outdesktopfile))
time.sleep(1)
print("删除桌面 {}, {}".format(filehtm, outdesktopfile))



