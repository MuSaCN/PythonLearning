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
__mypath__ = MyPath.MyClass_Path("\\python金融大数据挖掘与分析全流程详解")  # 路径类
myfile = MyFile.MyClass_File()  # 文件操作类
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
myWebQD = MyWebCrawler.MyClass_WebQuotesDownload()  # 金融行情下载类
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector()  # 向量型回测类
myML = MyMachineLearning.MyClass_MachineLearning()  # 机器学习综合类
myWebR = MyWebCrawler.MyClass_Requests()  # Requests爬虫类
mySQL = MyDataBase.MyClass_MySQL(connect=False)  # MySQL类
myWebAPP = MyWebCrawler.MyClass_APPIntegration() # 整合应用类
#------------------------------------------------------------

# 5.1 数据去重及清洗优化
company = "腾讯"
url = 'https://www.thepaper.cn/newsDetail_forward_6366027'
text = myWebR.get(url).text
print(text)
myWebR.findall(company[0] + '.{0,5}' + company[-1], text)
text = myWebR.no_messy_code(text)
print(text)
myWebR.findall(company[0] + '.{0,5}' + company[-1], text)
text = myWebR.get_href_content(url)
print(text)
myWebR.findall(company[0] + '.{0,5}' + company[-1], text)

myWebAPP.news_baidu("阿里巴巴")
myWebAPP.news_sogou("阿里巴巴")
myWebAPP.news_sina("阿里巴巴")


mySQL.__init__("quant")
sql = 'SELECT * FROM news WHERE company = "阿里巴巴" AND (date BETWEEN DATE_FORMAT("2020-03-08","%Y-%m-%d 00:00:00") AND DATE_FORMAT("2020-03-08","%Y-%m-%d 23:59:59") or date = "2020-03-08")'
sql = 'SELECT * FROM news WHERE company = "阿里巴巴" AND (date = "2020-03-08" or date BETWEEN DATE_FORMAT("2020-03-08","%Y-%m-%d 00:00:00") AND DATE_FORMAT("2020-03-08","%Y-%m-%d 23:59:59"))'

import time
today = time.strftime("%Y-%m-%d")
company = "阿里巴巴"
sql = 'SELECT * FROM news WHERE company = "阿里巴巴" AND (date = "2020-03-08" or date BETWEEN DATE_FORMAT("2020-03-08","%Y-%m-%d 00:00:00") AND DATE_FORMAT("2020-03-08","%Y-%m-%d 23:59:59"))'

sql = 'SELECT * FROM news WHERE company = "'+company+'" AND (date = "'+today+'" or date BETWEEN DATE_FORMAT("'+today+'","%Y-%m-%d 00:00:00") AND DATE_FORMAT("'+today+'","%Y-%m-%d 23:59:59"))'

mySQL.execute_fetchall_commit(sql)


mySQL.deletetable_content("news")
mySQL.close()

keywords = ['违约', '诉讼', '兑付', '阿里', '百度', '京东', '互联网']
myWebAPP.news_baidu("阿里巴巴",rtt=1,scorekeyword=keywords,checkhref=True,word_href=None,database="quant.news")
myWebAPP.news_sogou("阿里巴巴",sort=0,scorekeyword=keywords,checkhref=True,word_href=None,database="quant.news")
myWebAPP.news_sina("阿里巴巴",sort="time",scorekeyword=keywords,checkhref=True,word_href=None,database="quant.news")

mySQLAPP = MyDataBase.MyClass_SQL_APPIntegration()

# 5.4.3 从数据库汇总每日评分
mySQLAPP.totalscore_daily("quant.news",word="阿里巴巴",today="2020-03-06")

