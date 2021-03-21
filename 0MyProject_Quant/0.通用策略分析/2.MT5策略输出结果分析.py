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

file = __mypath__.get_desktop_path() + "\\MT5test.xlsx"

total_file = pd.read_excel(file, header=None)
# ---获取Orders和Deals的序号
index_order = total_file[0][total_file[0] == "Orders"].index[0]
index_deal = total_file[0][total_file[0] == "Deals"].index[0]
index_setting = total_file[0][total_file[0] == "Settings"].index[0]
index_result = total_file[0][total_file[0] == "Results"].index[0]

# ---整理下策略设置
strat_setting = total_file[index_setting+1:index_result]
strat_setting.dropna(axis=0,how='all',inplace=True)
strat_setting.dropna(axis=1,how='all',inplace=True)
strat_setting.reset_index(drop=True,inplace=True)
strat_setting.columns = [i for i in range(len(strat_setting.columns))]

# ---整理下策略结果
strat_result = total_file[index_result+1:index_order]
strat_result.dropna(axis=0,how='all',inplace=True)
strat_result.dropna(axis=1,how='all',inplace=True)
strat_result.reset_index(drop=True,inplace=True)
strat_result.columns = [i for i in range(len(strat_result.columns))]
# 是否需要转换竖排格式
strat_result1 = pd.DataFrame([],columns=[0,1])
for i in range(len(strat_result)): # i=0
    for j in range(0, len(strat_result.columns)-1, 2): # j=0
        temp = strat_result.iloc[i:i+1,j:j+2]
        temp.columns = strat_result1.columns
        strat_result1 = pd.concat((strat_result1,temp), axis=0, ignore_index=True)
strat_result1.dropna(axis=0,how='all',inplace=True)
strat_result1.reset_index(drop=True,inplace=True)






# ---整理下Orders
order_file = total_file[index_order:index_deal]
order_columns = order_file.iloc[1] # 第一次的列名，包括 NaN 列
order_content = order_file[2:]
order_content.columns = order_columns.tolist()
order_columns = order_columns.dropna() # 第二次的列名，丢弃NaN列
order_content = order_content[order_columns]
order_content.reset_index(drop=True,inplace=True)

# ---整理下Deals
deal_file = total_file[index_deal:]
deal_columns = deal_file.iloc[1] # 第一次的列名，包括 NaN 列
deal_content = deal_file[2:]
deal_content.columns = deal_columns.tolist()
deal_columns = deal_columns.dropna() # 第二次的列名，丢弃NaN列
deal_content = deal_content[deal_columns]
deal_content.reset_index(drop=True,inplace=True)





