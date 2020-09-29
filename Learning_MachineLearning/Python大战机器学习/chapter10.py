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

# ------集成学习
# ---AdaBoostClassifier
from sklearn import ensemble
digits = myML.DataPre.load_datasets("digits") # 使用 scikit-learn 自带的 digits 数据集
X_train,X_test,y_train,y_test = myML.DataPre.train_test_split(digits.data,digits.target,test_size=0.25,random_state=0,stratify=digits.target) # 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4

# 测试 AdaBoostClassifier 的用法，绘制 AdaBoostClassifier 的预测性能随基础分类器数量的影响
myML.Ensemble.plotparam_ensemble(X_train,X_test,y_train,y_test,"ensemble.AdaBoostClassifier()",bool_staged=True, learning_rate=[0.1])

# 测试  AdaBoostClassifier 的预测性能随基础分类器数量和基础分类器的类型的影响
myML.Ensemble.plotparam_ensemble(X_train,X_test,y_train,y_test,"ensemble.AdaBoostClassifier()",bool_staged=True, base_estimator=[None,"naive_bayes.GaussianNB()"],learning_rate=[0.1])

# 测试  AdaBoostClassifier 的预测性能随学习率的影响
learning_rates=np.linspace(0.1,0.5)
myML.Ensemble.plotparam_ensemble(X_train,X_test,y_train,y_test,"ensemble.AdaBoostClassifier()",bool_staged=False,drawParam=1,show=False,learning_rate=learning_rates,n_estimators=[10])
myML.Ensemble.plotparam_ensemble(X_train,X_train,y_train,y_train,"ensemble.AdaBoostClassifier()",bool_staged=False,drawParam=1,show=True,learning_rate=learning_rates,n_estimators=[10])

# 测试  AdaBoostClassifier 的预测性能随学习率和 algorithm 参数的影响
algorithms=['SAMME.R','SAMME']
learning_rates=[0.05,0.1,0.5,0.9]
myML.Ensemble.plotparam_ensemble(X_train,X_test,y_train,y_test,"ensemble.AdaBoostClassifier()",bool_staged=True,learning_rate=learning_rates,algorithm=algorithms)


# ---AdaBoostRegressor
digits = myML.DataPre.load_datasets("digits") # 使用 scikit-learn 自带的 digits 数据集
X_train,X_test,y_train,y_test = myML.DataPre.train_test_split(digits.data,digits.target,test_size=0.25,random_state=0,stratify=digits.target) # 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4

# 测试 AdaBoostRegressor 的用法，绘制 AdaBoostRegressor 的预测性能随基础回归器数量的影响
myML.Ensemble.plotparam_ensemble(X_train,X_test,y_train,y_test,"ensemble.AdaBoostRegressor()",bool_staged=True)

# 测试 AdaBoostRegressor 的预测性能随基础回归器数量的和基础回归器类型的影响
myML.Ensemble.plotparam_ensemble(X_train,X_test,y_train,y_test,"ensemble.AdaBoostRegressor()",bool_staged=True,base_estimator=[None,"svm.LinearSVR(epsilon=0.01,C=100)"])

# 测试 AdaBoostRegressor 的预测性能随学习率的影响
learning_rates=np.linspace(0.01,0.5)
myML.Ensemble.plotparam_ensemble(X_train,X_test,y_train,y_test,"ensemble.AdaBoostRegressor()",bool_staged=False,learning_rate=learning_rates,n_estimators=[20])

# 测试 AdaBoostRegressor 的预测性能随损失函数类型的影响
losses=['linear','square','exponential']
myML.Ensemble.plotparam_ensemble(X_train,X_test,y_train,y_test,"ensemble.AdaBoostRegressor()",bool_staged=True,loss=losses,n_estimators=[30])


# ---GradientBoostingClassifier
from sklearn import ensemble
digits = myML.DataPre.load_datasets("digits") # 使用 scikit-learn 自带的 digits 数据集
X_train,X_test,y_train,y_test = myML.DataPre.train_test_split(digits.data,digits.target,test_size=0.25,random_state=0,stratify=digits.target) # 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4

# 测试 GradientBoostingClassifier 的用法
clf=ensemble.GradientBoostingClassifier()
clf.fit(X_train,y_train)
print("Traing Score:%f"%clf.score(X_train,y_train))
print("Testing Score:%f"%clf.score(X_test,y_test))

# 测试 GradientBoostingClassifier 的预测性能随 n_estimators 参数的影响
nums=np.arange(1,10,step=2)
myML.Ensemble.plotparam_ensemble(X_train,X_test,y_train,y_test,"ensemble.GradientBoostingClassifier()",bool_staged=False,n_estimators=nums)

# 测试 GradientBoostingClassifier 的预测性能随 max_depth 参数的影响
maxdepths=np.arange(1,5)
myML.Ensemble.plotparam_ensemble(X_train,X_test,y_train,y_test,"ensemble.GradientBoostingClassifier()",bool_staged=False,max_depth=maxdepths,max_leaf_nodes=[None])

# 测试 GradientBoostingClassifier 的预测性能随学习率参数的影响
learnings=np.linspace(0.1, 0.4, 4)
myML.Ensemble.plotparam_ensemble(X_train,X_test,y_train,y_test,"ensemble.GradientBoostingClassifier()",bool_staged=False,learning_rate=learnings)

# 测试 GradientBoostingClassifier 的预测性能随 subsample 参数的影响
subsamples=np.linspace(0.01,1.0,4)
myML.Ensemble.plotparam_ensemble(X_train,X_test,y_train,y_test,"ensemble.GradientBoostingClassifier()",bool_staged=False,subsample=subsamples)

# 测试 GradientBoostingClassifier 的预测性能随 max_features 参数的影响
max_features=np.linspace(0.01,1.0,5)
myML.Ensemble.plotparam_ensemble(X_train,X_test,y_train,y_test,"ensemble.GradientBoostingClassifier()",bool_staged=False,max_features=max_features)





