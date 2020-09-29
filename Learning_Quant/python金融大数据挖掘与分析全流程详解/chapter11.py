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
myWebQD = MyWebCrawler.MyClass_QuotesDownload(tushare=False)  # 金融行情下载类
myWebR = MyWebCrawler.MyClass_Requests()  # Requests爬虫类
myWebS = MyWebCrawler.MyClass_Selenium(openChrome=False)  # Selenium模拟浏览器类
myWebAPP = MyWebCrawler.MyClass_APPIntegration()  # 爬虫整合应用类
myEmail = MyWebCrawler.MyClass_Email() # 邮箱交互类
#------------------------------------------------------------

myEmail.__init__(to = "435116098@qq.com")
myEmail.set_subject_maintext(subject='测试邮件主题!',maintext='测试邮件正文内容')
myEmail.send_message()



# 1.编写邮件正文内容
mail_msg = '''
<p>这个是一个常规段落</p>
<p><a href="https://www.baidu.com">这是一个包含链接的段落</a></p>
'''
myEmail.__init__()
myEmail.attach_subject_maintext('html主题!',mail_msg,"html","utf-8")
myEmail.send_message()

# 11.1.4 发送邮件附件
mail_msg = '''
<p>这个是一个常规段落</p>
<p><a href="https://www.baidu.com">这是一个包含链接的段落</a></p>
'''
myEmail.__init__()
myEmail.attach_subject_maintext('html主题!',mail_msg,"html","utf-8")
myEmail.attach_file("公司A理财公告.PDF","A.PDF")
myEmail.attach_file("test.csv")
myEmail.send_message()

# 11.2.1 案例实战-自动发送数据分析邮件
company = "阿里巴巴"
date = "2020-03-08"
mySQLAPP = MyDataBase.MyClass_SQL_APPIntegration()
mySQLAPP.send_email_database(database="quant.news",word=company,date=date,to=None)

import schedule
schedule.every(1).minutes.do( myWebAPP.send_email_database,database="quant.news",word=company,date=date,to=None )

while True:
    schedule.run_pending()

