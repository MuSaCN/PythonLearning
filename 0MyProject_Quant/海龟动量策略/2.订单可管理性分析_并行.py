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
myMT5Pro = MyMql.MyClass_ConnectMT5Pro(connect=False)  # Python链接MT5高级类
myMT5Indi = MyMql.MyClass_MT5Indicator()  # MT5指标Python版
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------

'''
# 订单可管理性：如果一个策略在未来1期持仓表现不错，同时在未来多期持仓也表现不错。这就表明，这个策略的交易订单在时间伸展上能够被管理，我们称作为订单具备可管理性。
# 对训练集进行多holding回测，展示结果的夏普比曲线和胜率曲线。
# 采用无重复持仓模式和重复持仓模式。
# 如果前3个夏普都是递增的，则选择之。输出测试图片。否则不认为具有可管理性，则弃之。
# 并行运算以品种来并行
'''

#%%
from MyPackage.MyProjects.向量化策略测试.More_Holding import Auto_More_Holding
more_h = Auto_More_Holding()

#%% ******修改这里******
strategy_para_name = ["n", "holding", "lag_trade"]
symbol_list = myMT5Pro.get_main_symbol_name_list()
total_folder = "F:\\工作---策略研究\\公开的海龟策略\\_海龟动量研究"
label1, label2 = "sharpe", "winRate"  # 策略训练集多holding回测，选择夏普比和胜率来分析
readfile_suffix = ".original" # 输入的文档加后缀，""表示不加，加词缀要加点号".original"
outfile_suffix = ".holdingtest" # 输出的文档加后缀

#%% ******修改函数******
#  策略的当期信号(不用平移)：para_list策略参数，默认-1为lag_trade，-2为holding。
def stratgy_signal(dataframe, para_list=list or tuple):
    return myBTV.stra.turtle_momentum(dataframe, para_list[0], price_arug= ["High", "Low", "Close"])
stratgy_signal = stratgy_signal

#%%
para = ["EURUSD"]
symbol = para[0]  # symbol = "EURUSD"
print("%s 开始订单可管理性分析..." % symbol)
# ---定位策略参数自动选择文档，获取各组参数
in_file = total_folder + "\\策略参数自动选择\\{}\\{}.total.{}{}.xlsx".format(symbol, symbol, "filter1", readfile_suffix)   # 固定只分析 filter1
out_folder = __mypath__.dirname(in_file, 0)  # "...\\策略参数自动选择\\EURUSD"
filecontent = pd.read_excel(in_file)
copyfile = filecontent.copy() # 备份以便于操作

#%%
for i in range(len(filecontent)):  # i=0
    # ---解析文档
    # 获取各参数
    timeframe = filecontent.iloc[i]["timeframe"]
    direct = filecontent.iloc[i]["direct"]
    # 策略参数
    strat_para = [filecontent.iloc[i][strategy_para_name[j]] for j in
                  range(len(strategy_para_name))]
    # 满足3个标的指标都是递增才输出路径
    suffix = myBTV.string_strat_para(strategy_para_name, strat_para)

    # ---准备数据
    date_from, date_to = myMT5Pro.get_date_range(timeframe)
    data_total = myMT5Pro.getsymboldata(symbol, timeframe, date_from, date_to, index_time=True, col_capitalize=True)
    data_train, data_test = myMT5Pro.get_train_test(data_total, train_scale=0.8)

    # ---信号的时间可管理性分析，以3个holding是否递增来判断。
    # 展开holding参数
    holding_test = [holding for holding in range(1, 3 + 1)]
    # 下面的信号质量计算是否重复持仓都要分析。重复持仓主要看胜率。
    out_list1_NoRe, out_list2_NoRe, out_list1_Re, out_list2_Re = [], [], [], []

    # ---
    for holding in holding_test:  # holding=1
        # 策略参数更换 -2位置的holding参数
        para_list = strat_para[0:-2] + [holding] + [strat_para[-1]]
        # 获取信号数据
        signal = stratgy_signal(data_train, para_list=para_list)
        # 信号分析，无重复持仓模式: signal_quality_NoRepeatHold / signal_quality
        outStrat_NoRe, outSignal_NoRe = myBTV.signal_quality_NoRepeatHold(signal=signal[direct], price_DataFrame=data_train, holding=holding, lag_trade=para_list[-1], plotStrat=False)
        out_list1_NoRe.append(outStrat_NoRe[direct][label1])
        out_list2_NoRe.append(outStrat_NoRe[direct][label2])
        # 信号分析，可重复持仓模式：
        outStrat_Re, outSignal_Re = myBTV.signal_quality(signal=signal[direct], price_DataFrame=data_train, holding=holding, lag_trade=para_list[-1], plotStrat=False)
        out_list1_Re.append(outStrat_Re[direct][label1])
        out_list2_Re.append(outStrat_Re[direct][label2])
    # ---判断是否单调增
    condition1 = out_list1_NoRe[0]<=out_list1_NoRe[1] and out_list1_NoRe[1] <=out_list1_NoRe[2]
    condition2 = out_list1_Re[0]<=out_list1_Re[1] and out_list1_Re[1]<=out_list1_Re[2]
    # ---如果不是单调增，则备份的内容删除对应行
    if(not (condition1 and condition2)):
        copyfile = copyfile.drop(i)

#%%
# ---备份的内容重建索引
copyfile.reset_index(inplace=True)
copyfile.to_excel(out_folder + "\\{}.total.{}{}.xlsx".format(symbol, "filter1", outfile_suffix))













#%%
more_h.core_num = -1
if __name__ == '__main__':
    # ---
    more_h.main_func()

