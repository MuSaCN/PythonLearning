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

# ------半监督学习

# ---LabelPropagation
from sklearn import semi_supervised
digits = myML.DataPre.load_datasets("digits")
######   混洗样本　########
rng = np.random.RandomState(0)
indices = np.arange(len(digits.data)) # 样本下标集合
rng.shuffle(indices) # 混洗样本下标集合
X = digits.data[indices]
y = digits.target[indices]
###### 生成未标记样本的下标集合 ####
n_labeled_points = int(len(y)/10) # 只有 10% 的样本有标记
unlabeled_indices = np.arange(len(y))[n_labeled_points:] # 后面 90% 的样本未标记

X,y,unlabeled_indices

# 测试 LabelPropagation 的用法
y_train=np.copy(y) # 必须拷贝，后面要用到 y
y_train[unlabeled_indices]=-1 # 未标记样本的标记设定为 -1
clf=semi_supervised.LabelPropagation(max_iter=100,kernel='rbf',gamma=0.1)
clf.fit(X,y_train)
### 获取预测准确率
true_labels = y[unlabeled_indices] # 真实标记
myML.Semi.showModelTest(clf,X[unlabeled_indices],true_labels)

# 测试 LabelPropagation 的 rbf 核时，预测性能随 gamma 的变化
y_train=np.copy(y) # 必须拷贝，后面要用到 y
y_train[unlabeled_indices]=-1 # 未标记样本的标记设定为 -1
gammas=np.logspace(-2,2,num=10)
myML.plotML.PlotParam_Score(X,X[unlabeled_indices],y_train,y[unlabeled_indices],
                            "semi_supervised.LabelPropagation()",drawParam=1,logX=True,
                            gamma=gammas,max_iter=[100],kernel=['rbf'])


# 测试 LabelPropagation 的 knn 核时，预测性能随 alpha 和 n_neighbors 的变化
y_train=np.copy(y) # 必须拷贝，后面要用到 y
y_train[unlabeled_indices]=-1 # 未标记样本的标记设定为 -1
Ks=[1,2,3,4,5,8,10,15,20,25,30,35,40,50]
myML.plotML.PlotParam_Score(X,X[unlabeled_indices],y_train,y[unlabeled_indices],
                            "semi_supervised.LabelPropagation()",drawParam=1,logX=True,
                            n_neighbors=Ks,max_iter=[100],kernel=['knn'])


# ---LabelSpreading
from sklearn import semi_supervised
digits = myML.DataPre.load_datasets("digits")
######   混洗样本　########
rng = np.random.RandomState(0)
indices = np.arange(len(digits.data)) # 样本下标集合
rng.shuffle(indices) # 混洗样本下标集合
X = digits.data[indices]
y = digits.target[indices]
###### 生成未标记样本的下标集合 ####
n_labeled_points = int(len(y)/10) # 只有 10% 的样本有标记
unlabeled_indices = np.arange(len(y))[n_labeled_points:] # 后面 90% 的样本未标记

X,y,unlabeled_indices

# 测试 LabelSpreading 的用法
y_train=np.copy(y) # 必须拷贝，后面要用到 y
y_train[unlabeled_indices]=-1 # 未标记样本的标记设定为 -1
clf=semi_supervised.LabelSpreading(max_iter=100,kernel='rbf',gamma=0.1)
clf.fit(X,y_train)
### 获取预测准确率
predicted_labels = clf.transduction_[unlabeled_indices] # 预测标记
true_labels = y[unlabeled_indices] # 真实标记
myML.Semi.showModelTest(clf,X[unlabeled_indices],true_labels)

# 测试 LabelSpreading 的 rbf 核时，预测性能随 alpha 和 gamma 的变化
y_train=np.copy(y) # 必须拷贝，后面要用到 y
y_train[unlabeled_indices]=-1 # 未标记样本的标记设定为 -1
gammas=np.logspace(-2,2,num=10)
myML.plotML.PlotParam_Score(X,X[unlabeled_indices],y_train,y[unlabeled_indices],
                            "semi_supervised.LabelSpreading()",drawParam=1,logX=True,
                            gamma=gammas,max_iter=[100],kernel=['rbf'])

# 测试 LabelSpreading 的 knn 核时，预测性能随 alpha 和 n_neighbors 的变化
y_train=np.copy(y) # 必须拷贝，后面要用到 y
y_train[unlabeled_indices]=-1 # 未标记样本的标记设定为 -1
Ks=[1,2,3,4,5,8,10,15,20,25,30,35,40,50]
myML.plotML.PlotParam_Score(X,X[unlabeled_indices],y_train,y[unlabeled_indices],
                            "semi_supervised.LabelSpreading()",drawParam=1,logX=True,
                            n_neighbors=Ks,max_iter=[100],kernel=['knn'])

