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
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector()  # 向量型回测类
myML = MyMachineLearning.MyClass_MachineLearning()  # 机器学习综合类
mySQL = MyDataBase.MyClass_MySQL(connect=False)  # MySQL类
myWebQD = MyWebCrawler.MyClass_WebQuotesDownload(tushare=False)  # 金融行情下载类
myWebR = MyWebCrawler.MyClass_Requests()  # Requests爬虫类
myWebS = MyWebCrawler.MyClass_Selenium() # Selenium模拟浏览器类
#------------------------------------------------------------

# 8.2 爬虫进阶2-爬虫利器selenium库详解

# 1.打开及关闭网页+网页最大化
myWebS.__init__(defaultbrowser=True)
myWebS.get("https://www.baidu.com/")
myWebS.quit()

# 2.xpath方法 / css_selector方法 来定位元素
pages = []
myWebS.__init__(openChrome=True)
myWebS.get("https://www.baidu.com/")

pages.append(myWebS.page_source())
myWebS.find_element('//*[@id="kw"]').send_keys('量化投资')
myWebS.find_element('#su').click()
import datetime
now = datetime.datetime.now()
while (datetime.datetime.now() - now).seconds < 2 :
    data = myWebS.page_source()
    if pages[-1] != data:
        pages.append(data)
len(pages)
data = myWebS.page_source(sleep=3)
myWebS.quit()

# 5.browser.page_source方法来获取新浪财经股票信息
myWebS.__init__(openChrome=True)
myWebS.get("http://finance.sina.com.cn/realstock/company/sh000001/nc.shtml")
data = myWebS.page_source()
print(data)
myWebS.quit()

# 6.Chrome Headless无界面浏览器设置
myWebS.__init__(openChrome=True,headless=True)
myWebS.get("http://finance.sina.com.cn/realstock/company/sh000001/nc.shtml")
data = myWebS.page_source()
print(data)
myWebS.quit()




