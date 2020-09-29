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
__mypath__ = MyPath.MyClass_Path()  # 路径类
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
myWebQD = MyWebCrawler.MyClass_QuotesDownload(tushare=False)  # 金融行情下载类
myWebR = MyWebCrawler.MyClass_Requests()  # Requests爬虫类
myWebS = MyWebCrawler.MyClass_Selenium(openChrome = False)  # Selenium模拟浏览器类
myWebAPP = MyWebCrawler.MyClass_APPIntegration() # 爬虫整合应用类
#------------------------------------------------------------

# 9.1 新浪股票实时数据挖掘实战
myWebAPP.sina_realstock("sh000001",quit=True)


# 9.2 东方财富网数据挖掘实战
myWebAPP.news_eastmoney("阿里巴巴", quit=True, database="quant.news")


# 9.3 裁判文书网数据挖掘实战
'''2019年8月份之后裁判文书网改版，其反爬非常强，所以模拟键盘鼠标操作后等待很久也等不到刷新，
所以这里主要给大家练习下如何通过selenium库模拟键盘鼠标操作。'''
from selenium import webdriver
import time
browser = webdriver.Chrome()
browser.get('http://wenshu.court.gov.cn/')
browser.find_element_by_xpath('//*[@id="_view_1540966814000"]/div/div[1]/div[2]/input').clear()  # 清空原搜索框
browser.find_element_by_xpath('//*[@id="_view_1540966814000"]/div/div[1]/div[2]/input').send_keys('房地产')  # 在搜索框内模拟输入'房地产'三个字
browser.find_element_by_xpath('//*[@id="_view_1540966814000"]/div/div[1]/div[3]').click()  # 点击搜索按钮
time.sleep(10)  # 如果还是获取不到你想要的内容，你可以把这个时间再稍微延长一些，现在裁判文书网反爬非常厉害，所以可能等待也等不到刷新，所以这里主要给大家练习下模拟键盘鼠标操作
data = browser.page_source
browser.quit()
print(data)

# 9.4 巨潮资讯网数据挖掘实战
myWebAPP.news_cninfo("阿里巴巴")
myWebAPP.quit()

myWebAPP.news_cninfo("阿里巴巴",database="quant.news")

