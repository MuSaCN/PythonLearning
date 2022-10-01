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

'''
# 1.同一个策略、同一个时间框、同一个方向下，不同的参数之间进行比较筛选。
# 2.筛选最佳的占优策略 或排除最差的策略。思路：模式1：先分析词缀sharpe和cumRet下是否有指定比率(比如80%)领先者，若有则领先者为最佳，否则进入模式2；模式2：先分析某个词缀(比如sharpe)下哪个策略的优势超过指定比率(比如80%)，该策略得1分。对所有词缀进行分析，若某个策略的得分最大且超过指定数量(词缀个数*2*80%)，则该策略认为是最佳的占优策略。
# 3.排除最差策略思想与上述相反。
# 4.反复筛选，直到剩余1个 或 找不到最佳最差 或 找到最佳。
'''

#%%
from MyPackage.MyProjects.向量化策略测试.More_Holding import Strategy_Better
s_better = Strategy_Better()
myDefault.set_backend_default("agg")


#%% ******修改这里******
s_better.strategy_para_name = ["n", "holding", "lag_trade"]
s_better.symbol_list = myMT5Pro.get_mainusd_symbol_name_list()
s_better.total_folder = "F:\\工作---策略研究\\2.公开的海龟策略\\_海龟反转研究"
s_better.readfile_suffix = ".holdingtest" # 输入的文档加后缀 .holdingtest
s_better.outfile_suffix = ".better" # 输出的文档加后缀
s_better.core_num = -1


#%% ******修改函数******
#  策略的当期信号(不用平移)：para_list策略参数，默认-1为lag_trade，-2为holding。
def stratgy_signal(dataframe, para_list=list or tuple):
    return myBTV.stra.turtle_momentum(dataframe, para_list[0], price_arug= ["High", "Low", "Close"])
s_better.stratgy_signal = stratgy_signal


#%%
if __name__ == '__main__':
    # ---
    print("开始同策同框同向不同参数比较筛选： ")
    s_better.main_func()










