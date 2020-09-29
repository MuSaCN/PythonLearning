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
myword = MyFile.MyClass_Word() # word生成类
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
mySQLAPP = MyDataBase.MyClass_SQL_APPIntegration() # 数据库应用整合
myWebQD = MyWebCrawler.MyClass_QuotesDownload(tushare=False)  # 金融行情下载类
myWebR = MyWebCrawler.MyClass_Requests()  # Requests爬虫类
myWebS = MyWebCrawler.MyClass_Selenium(openChrome=False)  # Selenium模拟浏览器类
myWebAPP = MyWebCrawler.MyClass_Web_APPIntegration()  # 爬虫整合应用类
myEmail = MyWebCrawler.MyClass_Email()  # 邮箱交互类
myReportA = MyQuant.MyClass_ReportAnalysis()  # 研报分析类
#------------------------------------------------------------

# 创建Word对象
myword.__init__("")
myword.add_heading("三行情诗3",level=2)
myword.add_paragraph_run('螃蟹在剥我的壳，笔记本在写我',size=26,RGB=(54, 95, 145))
myword.add_paragraph_run('漫天的我落在枫叶上雪花上',bold=True,italic=True,underline=True)
myword.add_paragraph_run('而你在想我',alignment="JUSTIFY")
myword.add_paragraph_run('设置首行缩进示例文字',first_line_indent=0.32)
myword.add_paragraph_run('设置行距示例文字',line_spacing=16)
myword.add_paragraph_run('设置段前段后距示例文字',space_before=14)

myword.add_paragraph_list('点序号', style='List Bullet')
myword.add_paragraph_list('数字序号', style='List Number')

table = myword.add_table(rows=2, cols=3, style='Light Shading Accent 1')
table.cell(0, 0).text = '第一句'  # 第一行第一列
table.cell(0, 1).text = '第二句'  # 第一行第二列
table.cell(0, 2).text = '第三句'  # 第一行第三列
table.cell(1, 0).text = '克制'  # 第二行第一列
table.cell(1, 1).text = '再克制'  # 第二行第二列
table.cell(1, 2).text = '"在吗"'  # 第三行第三列

myword.add_picture('水墨.png',widthInches=3,heightInches=3,alignment="CENTER")
myword.save("三行情书3.docx")

# 13.3 案例实战-生成数据分析报告
mySQLAPP.creat_docx_database(database="quant.news",word="阿里巴巴",date ="2020-03-08")





