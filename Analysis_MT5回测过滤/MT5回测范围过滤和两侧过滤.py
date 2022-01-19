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
myMT5Report = MyMT5Report.MyClass_StratTestReport(AddFigure=False)  # MT5策略报告类
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=False)  # 固定杠杆仓位类
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=False)  # 浮动杠杆仓位类
myMT5run = MyMql.MyClass_RunningMT5()  # Python运行MT5
myMT5code = MyMql.MyClass_CodeMql5()  # Python生成MT5代码
myMoneyM = MyTrade.MyClass_MoneyManage()  # 资金管理类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------
# Jupyter Notebook 控制台显示必须加上：%matplotlib inline ，弹出窗显示必须加上：%matplotlib auto
# %matplotlib inline
'''
MT5上回测策略必须要求可重复持仓
'''

# %% ############ MT5上回测策略必须要求可重复持仓 ############
import warnings
warnings.filterwarnings('ignore')

file = __mypath__.get_desktop_path() + "\\CrazyBreakMT5.EURUSD.H1.xlsx"

# 读取报告，加载品种信息到 self.symbol_df。注意部分平仓不适合deal_standard = True修正。
strat_setting, strat_result, dict_order_content, dict_deal_content = myMT5Report.read_report_xlsx(filepath=file, result_vert=True, deal_standard=False, onlytestsymbol=False)

# 解析下词缀
symbol = strat_setting.loc["Symbol:"][0]
timeframe, timefrom, timeto = myMT5Report.parse_period(strat_setting)
# 获取数据
data = myMT5Pro.getsymboldata(symbol, timeframe, timefrom, timeto, index_time=True, col_capitalize=True)

# 设置为指定品种的内容
order_content = dict_order_content[symbol]
deal_content = dict_deal_content[symbol]

# 分析交易单元，分为 unit_total、unit_buyonly、unit_sellonly。
unit_total = myMT5Report.content_to_unit_order(order_content=order_content, deal_content=deal_content, sortby="Order0")
unit_buyonly, unit_sellonly = myMT5Report.content_to_direct_unit_order(order_content=order_content, deal_content=deal_content, sortby="Order0")

result = myMT5Report.cal_result_no_money_manage(unit_order=unit_total)[0]

# ---绘制策略报告的资金走势结果，按all、buyonly、sellonly绘制。注意order和deal有区别，order是以整体单来算，deal是实际情况。
myMT5Report.plot_report_balance(unit_total, unit_buyonly, unit_sellonly, savefig=None, show=True)
deal_content["Balance"][1:-1].plot()
plt.show()



#%% #############################
#---获取与交易单元起始时间匹配的指标值：shift_indi=1表示信号确认，结果以"Order0"排序
tf_indi = "TIMEFRAME_H1"
indiname = "@ADXW"
para = [22]

new_unit_total = myMT5Report.indi_matching_unit(unit_total, symbol, timefrom, timeto, timeframe, tf_indi, 1, indiname, *para)
new_unit_buy = myMT5Report.indi_matching_unit(unit_buyonly, symbol, timefrom, timeto, timeframe, tf_indi, 1, indiname, *para)
new_unit_sell = myMT5Report.indi_matching_unit(unit_sellonly, symbol, timefrom, timeto, timeframe, tf_indi, 1, indiname, *para)


#%% #############################
# ======范围过滤======
# ---得到报告过滤的策略结果
result
result_filter, indi_start, indi_end = myMT5Report.get_report_filter_result(new_unit=new_unit_total, mode="range")

# ---画报告过滤的结果。参数中 new_unit 为 indi_matching_unit() 的输出结果。data为报告对应的原数据；mode="range"范围过滤；"2side"两侧过滤；
myMT5Report.plot_report_filter_analysis(data=data, new_unit=new_unit_total, mode="range",savefig=None, batch=False)


#%% #############################
# ======两侧过滤======
result
# ---得到报告过滤的策略结果
result_filter, indi_end, indi_start = myMT5Report.get_report_filter_result(new_unit=new_unit_total, mode="2side")

# ---画报告过滤的结果。参数中 new_unit 为 indi_matching_unit() 的输出结果。data为报告对应的原数据；mode="range"范围过滤；"2side"两侧过滤；
myMT5Report.plot_report_filter_analysis(data=data, new_unit=new_unit_total, mode="2side",savefig=None, batch=False)



