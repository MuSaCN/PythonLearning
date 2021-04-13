# Author:Zhang Yuan
from MyPackage import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

#------------------------------------------------------------
__mypath__ = MyPath.MyClass_Path("\\0MyProject_Quant")  # 路径类
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
# myMql = MyMql.MyClass_MqlBackups() # Mql备份类
# myMT5 = MyMql.MyClass_ConnectMT5(connect=False) # Python链接MetaTrader5客户端类
# myDefault = MyDefault.MyClass_Default_Matplotlib() # matplotlib默认设置
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
#------------------------------------------------------------

# 测试结论：python直接连接MT5速度 比 Python连接mysql速度 更快
mySQL.__init__("forextimefxtm-demo01",3308,True);
import time

start = time.process_time()
query = "SELECT * FROM eurusd_period_m2"
eurusd_m1 = mySQL.execute_fetchall_commit(query)
print("Time used:", (time.process_time() - start)) # Time used: 72.46875
data = pd.DataFrame(eurusd_m1)

start = time.process_time()
query = "SELECT * FROM eurusd_period_m2 WHERE Date>='2020-01-01 00:00:00' AND Date<'2020-06-01 00:00:00'"
eurusd_m1 = mySQL.execute_fetchall_commit(query)
print("Time used:", (time.process_time() - start)) # Time used: 1.4375
data = pd.DataFrame(eurusd_m1)
mySQL.close()

###################################################
myMT5 = MyMql.MyClass_ConnectMT5(connect=True) # Python链接MetaTrader5客户端类
import time

start = time.process_time()
data1 = myMT5.copy_rates_range("EURUSD",myMT5.mt5.TIMEFRAME_M2,[2020,1,1,0,0,0],[2020,6,1,0,0,0])
print("Time used:", (time.process_time() - start)) # Time used: 0.09375
data1 = myMT5.rates_to_DataFrame(data1,True)


start = time.process_time()
data1 = myMT5.copy_rates_range("EURUSD",myMT5.mt5.TIMEFRAME_M2,[1970,1,1,0,0,0],[2022,6,1,0,0,0])
print("Time used:", (time.process_time() - start)) # Time used: 0.0625
data1 = myMT5.rates_to_DataFrame(data1,True) # 3877059 rows x 8 columns
print("Time used:", (time.process_time() - start)) # Time used: 0.0625


myMT5.shutdown()
