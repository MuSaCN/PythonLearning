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
myWebC = MyWebCrawler.MyClass_WebCrawler()  # 综合网络爬虫类
#------------------------------------------------------------


# 3.1 百度新闻数据挖掘
myWebC.news_baidu(word="量化投资", rtt=1)


# 3.2.1 批量爬取多家公司新闻
companys = ['华能信托', '阿里巴巴', '万科集团', '百度集团', '腾讯', '京东']
for i in companys:  # 这个i只是个代号，可以换成其他内容
    myWebC.news_baidu(i,file="百度新闻爬取.txt")
    print(i + '百度新闻爬取成功')

# 3.3 异常处理及24小时数据挖掘实战
import time
# while True:
#     companys = ['华能信托', '阿里巴巴', '万科集团', '百度集团', '腾讯', '京东']
#     for i in companys:
#         try:
#             myWebC.news_baidu(i)
#             print(i + '百度新闻爬取成功')
#         except:
#             print(i + '百度新闻爬取失败')
#     time.sleep(20)  # 每10800秒运行一次，即3小时运行一次，注意缩进


# 3.4.2-1 一家公司批量爬取多页

for i in range(10):
    myWebC.news_baidu("阿里巴巴",page = i)
    print('第' + str(i+1) + '页爬取成功')

# 3.4.2-2 多家公司批量爬取多页
companys = ['华能信托', '阿里巴巴', '万科集团', '百度集团', '腾讯', '京东']
for company in companys:
    for i in range(10):  # i是从0开始的序号，所以下面要写i+1，这里一共爬取了20页
        myWebC.news_baidu(company, page = i)
        print(company + ' 第' + str(i+1) + '页爬取成功')

# 3.5.1 搜狗新闻数据挖掘实战
myWebC.news_sogou('阿里巴巴',sort=1,page=2,file=None)


# 3.5.2 新浪财经数据挖掘实战
myWebC.news_sina('阿里巴巴')

