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
__mypath__ = MyPath.MyClass_Path("\\TensorFlow实战Google深度学习框架")  # 路径类
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
#------------------------------------------------------------

import tensorflow as tf
a = tf.constant([1,2])
b = tf.constant([3,4])
result = a+b
sess = tf.compat.v1.Session()
sess.run(result)

#### 1. 定义两个不同的图
import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    v = tf.compat.v1.get_variable("v", [1], initializer=tf.zeros_initializer())  # 设置初始值为0

g2 = tf.Graph()
with g2.as_default():
    v = tf.compat.v1.get_variable("v", [1], initializer=tf.ones_initializer())  # 设置初始值为1

with tf.compat.v1.Session(graph=g1) as sess:
    tf.compat.v1.global_variables_initializer().run()
    with tf.compat.v1.variable_scope("", reuse=True):
        print(sess.run(tf.compat.v1.get_variable("v")))

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

# %% md

#### 2. 张量的概念

# %%

import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b
print(result)

sess = tf.InteractiveSession()
print(result.eval())
sess.close()

# %% md

#### 3. 会话的使用

# %% md

3.1
创建和关闭会话

# %%

# 创建一个会话。
sess = tf.Session()

# 使用会话得到之前计算的结果。
print(sess.run(result))

# 关闭会话使得本次运行中使用到的资源可以被释放。
sess.close()

# %% md

3.2
使用with
statement
来创建会话

# %%

with tf.Session() as sess:
    print(sess.run(result))

# %% md

3.3
指定默认会话

# %%

sess = tf.Session()
with sess.as_default():
    print(result.eval())

# %%

sess = tf.Session()

# 下面的两个命令有相同的功能。
print(sess.run(result))
print(result.eval(session=sess))

# %% md

#### 4. 使用tf.InteractiveSession构建会话

# %%

sess = tf.InteractiveSession()
print(result.eval())
sess.close()

# %% md

#### 5. 通过ConfigProto配置会话

# %%

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)

