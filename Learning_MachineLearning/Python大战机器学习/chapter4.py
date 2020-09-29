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
myML = MyMachineLearning.MyClass_MachineLearning()  # 机器学习综合类
#------------------------------------------------------------

# ---KNN分类模型
digits = myML.DataPre.load_datasets("digits")# 使用 scikit-learn 自带的手写识别数据集 Digit Dataset
X_train, X_test, y_train, y_test =myML.DataPre.train_test_split(digits.data, digits.target,test_size=0.25, random_state=0,stratify=digits.target)

from sklearn import neighbors
clf=neighbors.KNeighborsClassifier().fit(X_train,y_train)
myML.KNN.showModelTest(clf,X_train,y_train)
myML.KNN.showModelTest(clf,X_test,y_test)

# 测试 KNeighborsClassifier 中 n_neighbors 和 weights 参数的影响
X_train,X_test,y_train,y_test=myML.DataPre.train_test_split(digits.data, digits.target,test_size=0.25, random_state=0,stratify=digits.target)
Ks=[1,10,20,50,100,200,500]
weights=['uniform','distance']
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"neighbors.KNeighborsClassifier()",drawParam=2,logX=False,label="Test",show=False,n_neighbors=Ks,weights=weights)
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"neighbors.KNeighborsClassifier()",drawParam=2,logX=False,label="Train",show=True,n_neighbors=Ks,weights=weights)

# 测试 KNeighborsClassifier 中 n_neighbors 和 p 参数的影响
Ks=[1,10,20,50,100,200,500]
Ps=[1,2,10]
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"neighbors.KNeighborsClassifier()",drawParam=2,logX=False,label="Test",show=False,n_neighbors=Ks,p=Ps)
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"neighbors.KNeighborsClassifier()",drawParam=2,logX=False,label="Train",show=True,n_neighbors=Ks,p=Ps)


# ---KNN回归模型
from sklearn import neighbors
n=1000
X =5 * np.random.rand(n, 1)
y = np.sin(X).ravel()
y[::5] += 1 * (0.5 - np.random.rand(int(n/5))) # 每隔 5 个样本就在样本的值上添加噪音
X_train,X_test,y_train,y_test = myML.DataPre.train_test_split(X, y,test_size=0.25,random_state=0)# 进行简单拆分，测试集大小占 1/4

# 测试 KNeighborsRegressor 的用法
regr=neighbors.KNeighborsRegressor()
regr.fit(X_train,y_train)
print("Training Score:%f"%regr.score(X_train,y_train))
print("Testing Score:%f"%regr.score(X_test,y_test))

# 测试 KNeighborsRegressor 中 n_neighbors 和 weights 参数的影响
Ks=[1,10,20,50,100,200,500]
weights=['uniform','distance']
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"neighbors.KNeighborsRegressor()",drawParam=2,logX=False,label="Test",show=False,n_neighbors=Ks,weights=weights)
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"neighbors.KNeighborsRegressor()",drawParam=2,logX=False,label="Train",show=True,n_neighbors=Ks,weights=weights)

# 测试 KNeighborsRegressor 中 n_neighbors 和 p 参数的影响
Ks=[1,10,20,50,100,200,500]
Ps=[1,2,10]
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"neighbors.KNeighborsRegressor()",drawParam=2,logX=False,label="Test",show=False,n_neighbors=Ks,p=Ps)
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"neighbors.KNeighborsRegressor()",drawParam=2,logX=False,label="Train",show=True,n_neighbors=Ks,p=Ps)
