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
mySQL = MyDataBase.MyClass_MySQL(connect=False) # MySQL类
#------------------------------------------------------------


# 4.3.3 数据插入数据库
# 先预定义些变量
company = '阿里巴巴'
title = '测试标题'
href = '测试链接'
source = '测试来源'
date = '测试日期'

# 连接数据库，必须打开数据库服务器才行
mySQL.__init__(database='quant')
sql = "INSERT INTO table0(company, title, href, source, date) VALUES (%s, %s, %s, %s, %s)"
mySQL.execute_commit(sql, (company, title, href, source, date))
mySQL.close()


# 4.3.4 连接数据库并提取数据
# 1.根据1个条件查找并提取
company = '阿里巴巴'
mySQL.__init__(database='quant')
sql = 'SELECT * FROM news WHERE company = %s'  # 编写SQL语句
data = mySQL.execute_fetchall_commit(sql, company)
mySQL.close()


mySQL.__init__(database='quant')
mySQL.deletetable("news")
mySQL.deletetable_content("news")


# 2.根据2个条件查找并提取
company = '阿里巴巴'
title = '标题1'
mySQL.__init__(database='quant')
sql = 'SELECT * FROM news WHERE company = %s AND title = %s'  # 编写SQL语句
data = mySQL.execute_fetchall_commit(sql, (company, title))
mySQL.close()

# 4.3.5 连接数据库并删除数据
company = '阿里巴巴'
mySQL.__init__(database='quant')
sql = 'DELETE FROM table0 WHERE company = %s'  # 编写SQL语句
mySQL.execute_commit(sql, company)
mySQL.close()

# 4.4 把数据挖掘到数据存入数据库
myWebC.news_baidu("阿里巴巴",database="quant.table0",port=3308)

companys = ['华能信托', '阿里巴巴', '百度集团', '腾讯', '京东']
for company in companys:
    try:
        myWebC.news_baidu(company,database="quant.table0",port=3308)
        print(company + '爬取并存入数据库成功')
    except:
        print(company + '爬取并存入数据库失败')

myWebC.news_baidu("阿里巴巴",database="quant.news",port=3308)
myWebC.news_sogou("量化投资",database="quant.news",port=3308)
myWebC.news_sina("习近平",database="quant.news",port=3308)





