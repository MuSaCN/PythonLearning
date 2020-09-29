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
__mypath__ = MyPath.MyClass_Path("\\Python大战机器学习")  # 路径类
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
myML = MyMachineLearning.MyClass_MachineLearning() # 机器学习综合类
#------------------------------------------------------------
""" 广义线性模型 LinearRegression """
data = myML.DataPre.load_datasets(mode="diabetes")
X_train,X_test,Y_train,Y_test = myML.DataPre.train_test_split(data.data,data.target,test_size=0.25,random_state=0)
from sklearn import  linear_model

# ---LinearRegression线性回归
regr = linear_model.LinearRegression().fit(X_train, Y_train)
myML.LinearModel.showModelTest(regr, X_test, Y_test)

# ---岭回归 Ridge
regr = linear_model.Ridge().fit(X_train, Y_train)
myML.LinearModel.showModelTest(regr, X_test, Y_test)
# 测试alpha
alphas=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
myML.plotML.PlotParam_Score(X_train,X_test,Y_train,Y_test,"linear_model.Ridge()",drawParam=1,logX=True,alpha=alphas)

# ---Lasso
regr = linear_model.Lasso().fit(X_train, Y_train)
myML.LinearModel.showModelTest(regr, X_test, Y_test)
# test alpha
alphas=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
myML.plotML.PlotParam_Score(X_train,X_test,Y_train,Y_test,"linear_model.Lasso()",logX=True,alpha=alphas)

# ---ElasticNet
data = myML.DataPre.load_datasets(mode="diabetes")
X_train,X_test,Y_train,Y_test = myML.DataPre.train_test_split(data.data,data.target,test_size=0.25,random_state=0)
from sklearn import  linear_model
regr = linear_model.ElasticNet().fit(X_train, Y_train)
myML.LinearModel.showModelTest(regr, X_test, Y_test)
# test alpha and rhos
alphas=np.logspace(-1,1)
rhos=np.linspace(0.01,1)
myML.plotML.PlotParam_Score(X_train,X_test,Y_train,Y_test,"linear_model.ElasticNet()",drawParam=2,plot3D=True,alpha=alphas,l1_ratio=rhos)

# ---logistic 回归
data = myML.DataPre.load_datasets(mode="iris")
X_train,X_test,Y_train,Y_test = myML.DataPre.train_test_split(data.data,data.target,test_size=0.25,random_state=0)
from sklearn import  linear_model
# 有多少个分类，就有多少个优化函数
regr = linear_model.LogisticRegression().fit(X_train, Y_train)
myML.LinearModel.showModelTest(regr, X_test, Y_test)
# 测试 LogisticRegression 的预测性能随 multi_class 参数的影响
regr = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs').fit(X_train, Y_train)
myML.LinearModel.showModelTest(regr, X_test, Y_test)
# 测试 LogisticRegression 的预测性能随  C  参数的影响
Cs=np.logspace(-2,4,num=100)
myML.plotML.PlotParam_Score(X_train,X_test,Y_train,Y_test,"linear_model.LogisticRegression()",drawParam=1,logX=True,C=Cs)


# ---线性判别分析
data = myML.DataPre.load_datasets(mode="iris")
X_train,X_test,Y_train,Y_test = myML.DataPre.train_test_split(data.data,data.target,test_size=0.25,random_state=0)
#
from sklearn import discriminant_analysis
lda = discriminant_analysis.LinearDiscriminantAnalysis().fit(X_train, Y_train)
myML.LinearModel.showModelTest(lda, X_test, Y_test)
# 绘制经过 LDA 转换后的数据(4维降到3维)
myML.LinearModel.plot_LDA(X_train, Y_train)




