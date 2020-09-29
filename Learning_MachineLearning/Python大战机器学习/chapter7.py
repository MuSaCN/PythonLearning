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

# ---LinearSVC
from sklearn import  svm
iris=myML.DataPre.load_datasets("iris") # 使用 scikit-learn 自带的 iris 数据集
X_train,X_test,y_train,y_test = myML.DataPre.train_test_split(iris.data, iris.target,test_size=0.25, random_state=0,stratify=iris.target) # 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4

# 测试 LinearSVC 的用法
cls=svm.LinearSVC()
cls.fit(X_train,y_train)
myML.SVM.showModelTest(cls,X_test, y_test)

# 测试 LinearSVC 的预测性能随损失函数的影响
losses=['hinge','squared_hinge']
for loss in losses:
    cls=svm.LinearSVC(loss=loss)
    cls.fit(X_train,y_train)
    print("Loss:%s"%loss)
    myML.SVM.showModelTest(cls, X_test, y_test)

# 测试 LinearSVC 的预测性能随正则化形式的影响
L12=['l1','l2']
for p in L12:
    cls=svm.LinearSVC(penalty=p,dual=False)
    cls.fit(X_train,y_train)
    print("penalty:%s"%p)
    myML.SVM.showModelTest(cls, X_test, y_test)

# 测试 LinearSVC 的预测性能随参数 C 的影响
Cs=np.logspace(-2,1)
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"svm.LinearSVC()",drawParam=1,label="test",logX=True,show=False,C=Cs)
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"svm.LinearSVC()",drawParam=1,label="train",logX=True,show=True,C=Cs)


# ---SVC
from sklearn import  svm
iris=myML.DataPre.load_datasets("iris") # 使用 scikit-learn 自带的 iris 数据集
X_train,X_test,y_train,y_test = myML.DataPre.train_test_split(iris.data, iris.target,test_size=0.25, random_state=0,stratify=iris.target) # 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4

# 测试 SVC 的用法。这里使用的是最简单的线性核
cls=svm.SVC(kernel='linear')
cls.fit(X_train,y_train)
myML.SVM.showModelTest(cls,X_test,y_test)

# 测试多项式核的 SVC 的预测性能随 degree、gamma、coef0 的影响.
### 测试 degree ####
degrees=range(1,10)
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"svm.SVC()",drawParam=1,label="test",show=False,degree=degrees,kernel=['poly'])
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"svm.SVC()",drawParam=1,label="train",show=True, degree=degrees,kernel=['poly'])

### 测试 gamma ，此时 degree 固定为 3####
gammas=range(1,20)
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"svm.SVC()",drawParam=1,label="test",show=False,gamma=gammas,degree=[3],kernel=['poly'])
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"svm.SVC()",drawParam=1,label="train",show=True,gamma=gammas,degree=[3],kernel=['poly'])

### 测试 r ，此时 gamma固定为10 ， degree 固定为 3######
rs=range(0,20)
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"svm.SVC()",drawParam=1,label="test",show=False,coef0=rs,gamma=[10],degree=[3],kernel=['poly'])
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"svm.SVC()",drawParam=1,label="train",show=True,coef0=rs,gamma=[10],degree=[3],kernel=['poly'])

# 测试 高斯核的 SVC 的预测性能随 gamma 参数的影响
gammas=range(1,20)
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"svm.SVC()",drawParam=1,label="test",show=False,gamma=gammas,kernel=['rbf'])
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"svm.SVC()",drawParam=1,label="train",show=True,gamma=gammas,kernel=['rbf'])

# 测试 sigmoid 核的 SVC 的预测性能随 gamma、coef0 的影响
fig=plt.figure()
### 测试 gamma ，固定 coef0 为 0 ####
gammas=np.logspace(-2,1)
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"svm.SVC()",drawParam=1,logX=True,label="test",show=False,gamma=gammas,kernel=['sigmoid'],coef0=[0])
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"svm.SVC()",drawParam=1,logX=True,label="train",show=True,gamma=gammas,kernel=['sigmoid'],coef0=[0])

### 测试 r，固定 gamma 为 0.01 ######
rs=np.linspace(0,5)
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"svm.SVC()",drawParam=1,label="test",show=False,coef0=rs,gamma=[0.01],kernel=['sigmoid'])
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"svm.SVC()",drawParam=1, label="train",show=True,coef0=rs,gamma=[0.01],kernel=['sigmoid'])



# ---LinearSVR
from sklearn import  svm
diabetes=myML.DataPre.load_datasets("diabetes") # 使用 scikit-learn 自带的 iris 数据集
X_train,X_test,y_train,y_test = myML.DataPre.train_test_split(diabetes.data, diabetes.target,test_size=0.25, random_state=0) # 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4

# 测试 LinearSVR 的用法
regr=svm.LinearSVR()
regr.fit(X_train,y_train)
myML.SVM.showModelTest(regr,X_test, y_test)

# 测试 LinearSVR 的预测性能随不同损失函数的影响
losses=['epsilon_insensitive','squared_epsilon_insensitive']
for loss in losses:
    regr=svm.LinearSVR(loss=loss)
    regr.fit(X_train,y_train)
    print("loss：%s"%loss)
    myML.SVM.showModelTest(regr,X_test, y_test)

# 测试 LinearSVR 的预测性能随 epsilon 参数的影响
epsilons=np.logspace(-2,2)
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"svm.LinearSVR()",drawParam=1,logX=True,label="test",show=False,epsilon=epsilons,loss=['squared_epsilon_insensitive'])
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"svm.LinearSVR()",drawParam=1,logX=True,label="train",show=True,epsilon=epsilons,loss=['squared_epsilon_insensitive'])

# 测试 LinearSVR 的预测性能随 C 参数的影响
Cs=np.logspace(-1,2)
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"svm.LinearSVR()",drawParam=1,logX=True,label="test",show=False,C=Cs,epsilon=[0.1],loss=['squared_epsilon_insensitive'])
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"svm.LinearSVR()",drawParam=1,logX=True,label="train",show=True,C=Cs,epsilon=[0.1],loss=['squared_epsilon_insensitive'])



# ---SVR
from sklearn import  svm
diabetes=myML.DataPre.load_datasets("diabetes") # 使用 scikit-learn 自带的 iris 数据集
X_train,X_test,y_train,y_test = myML.DataPre.train_test_split(diabetes.data, diabetes.target,test_size=0.25, random_state=0) # 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4

# 测试 SVR 的用法。这里使用最简单的线性核
regr=svm.SVR(kernel='linear')
regr.fit(X_train,y_train)
print('Coefficients:%s, intercept %s'%(regr.coef_,regr.intercept_))
print('Score: %.2f' % regr.score(X_test, y_test))

# 测试 多项式核的 SVR 的预测性能随  degree、gamma、coef0 的影响.
### 测试 degree ####
degrees=range(1,20)
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"svm.SVR()",drawParam=1,label="test",show=False,degree=degrees,kernel=['poly'],coef0=[1])
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"svm.SVR()",drawParam=1,label="test",show=True,degree=degrees,kernel=['poly'],coef0=[1])

### 测试 gamma，固定 degree为3， coef0 为 1 ####
gammas=range(1,40)
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"svm.SVR()",drawParam=1,label="test",show=False,gamma=gammas,kernel=['poly'],coef0=[1],degree=[3])
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"svm.SVR()",drawParam=1,label="test",show=True,gamma=gammas,kernel=['poly'],coef0=[1],degree=[3])

### 测试 r，固定 gamma 为 20，degree为 3 ######
rs=range(0,20)
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"svm.SVR()",drawParam=1,label="test",show=False,coef0=rs,gamma=[20],kernel=['poly'],degree=[3])
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"svm.SVR()",drawParam=1,label="test",show=True,coef0=rs,gamma=[20],kernel=['poly'],degree=[3])

# 测试 高斯核的 SVR 的预测性能随 gamma 参数的影响
gammas=range(1,20)
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"svm.SVR()",drawParam=1,label="test",show=False,gamma=gammas,kernel=['rbf'])
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"svm.SVR()",drawParam=1,label="train",show=True,gamma=gammas,kernel=['rbf'])

# 测试 sigmoid 核的 SVR 的预测性能随 gamma、coef0 的影响.
fig=plt.figure()
### 测试 gammam，固定 coef0 为 0.01 ####
gammas=np.logspace(-1,3)
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"svm.SVR()",drawParam=1,label="test",logX=True,show=False,gamma=gammas,kernel=['sigmoid'],coef0=[0.01])
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"svm.SVR()",drawParam=1,label="train",logX=True,show=True,gamma=gammas,kernel=['sigmoid'],coef0=[0.01])

### 测试 r ，固定 gamma 为 10 ######
rs=np.linspace(0,5)
myML.plotML.PlotParam_Score(X_train,X_test,y_train,y_test,"svm.SVR()",drawParam=1,label="test",show=False,coef0=rs,gamma=[10],kernel=['sigmoid'])
myML.plotML.PlotParam_Score(X_train,X_train,y_train,y_train,"svm.SVR()",drawParam=1,label="train",show=True,coef0=rs,gamma=[10],kernel=['sigmoid'])


