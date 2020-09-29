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


'''# ------损失函数'''
from sklearn.metrics import zero_one_loss,log_loss

# ---测试 0-1 损失函数
y_true=[1,1,1,1,1,0,0,0,0,0]
y_pred=[0,0,0,1,1,1,1,1,0,0]
# normalize = True返回误分类样本比例 / False返回误分类样本数量
print("zero_one_loss<fraction>:",zero_one_loss(y_true,y_pred,normalize=True))
print("zero_one_loss<num>:",zero_one_loss(y_true,y_pred,normalize=False))

# ---测试对数损失函数
y_true=[1, 1, 1, 0, 0, 0]
# 数据依次表示[[ 预测为0的概率, 预测为1的概率 ]]
y_pred=[[0.1, 0.9],[0.2, 0.8],[0.3, 0.7], [0.7, 0.3],[0.8, 0.2],[0.9, 0.1]]
# normalize 如果为true，则返回每个样本的平均损失。 否则，返回每个样本损失的总和。
print("log_loss<average>:",log_loss(y_true,y_pred,normalize=True))
print("log_loss<total>:",log_loss(y_true,y_pred,normalize=False))


'''# ------数据集切分'''
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,LeaveOneOut,cross_val_score

# ---测试  train_test_split 的用法
X=[[1,2,3,4],[11,12,13,14],[21,22,23,24],[31,32,33,34],[41,42,43,44],[51,52,53,54],[61,62,63,64],[71,72,73,74]]
y=[1,1,0,0,1,1,0,0]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4, random_state=0) # 切分，测试集大小为原始数据集大小的 40%
print("X_train=",X_train)
print("X_test=",X_test)
print("y_train=",y_train)
print("y_test=",y_test)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4,random_state=0,stratify=y) # 分层采样切分，测试集大小为原始数据集大小的 40%
print("Stratify:X_train=",X_train)
print("Stratify:X_test=",X_test)
print("Stratify:y_train=",y_train)
print("Stratify:y_test=",y_test)

# ---测试  KFold 的用法
X=np.array([[1,2,3,4], [11,12,13,14], [21,22,23,24], [31,32,33,34], [41,42,43,44], [51,52,53,54], [61,62,63,64], [71,72,73,74], [81,82,83,84]])
y=np.array([1,1,0,0,1,1,0,0,1])
# 切分之前不混洗数据集
folder=KFold(n_splits=3,random_state=0,shuffle=False)
for train_index,test_index in folder.split(X,y):
      print("Train Index:",train_index)
      print("Test Index:",test_index)
      print("X_train:",X[train_index])
      print("X_test:",X[test_index])
      print("")
# 切分之前混洗数据集
shuffle_folder=KFold(n_splits=3,random_state=0,shuffle=True)
for train_index,test_index in shuffle_folder.split(X,y):
      print("Shuffled Train Index:",train_index)
      print("Shuffled Test Index:",test_index)
      print("Shuffled X_train:",X[train_index])
      print("Shuffled X_test:",X[test_index])
      print("")

# ---测试  StratifiedKFold 的用法
X=np.array([[1,2,3,4],[11,12,13,14],[21,22,23,24],[31,32,33,34],[41,42,43,44],[51,52,53,54],[61,62,63,64],[71,72,73,74]])
y=np.array([1,1,0,0,1,1,0,0])
folder=KFold(n_splits=4,random_state=0,shuffle=False)
stratified_folder=StratifiedKFold(n_splits=4,random_state=0,shuffle=False)
for train_index,test_index in folder.split(X,y):
      print("Train Index:",train_index)
      print("Test Index:",test_index)
      print("y_train:",y[train_index])
      print("y_test:",y[test_index])
      print("")
for train_index,test_index in stratified_folder.split(X,y):
      print("Stratified Train Index:",train_index)
      print("Stratified Test Index:",test_index)
      print("Stratified y_train:",y[train_index])
      print("Stratified y_test:",y[test_index])
      print("")


# ---测试  LeaveOneOut 的用法
X=np.array([[1,2,3,4], [11,12,13,14], [21,22,23,24], [31,32,33,34]])
y=np.array([1,1,0,0])
lo=LeaveOneOut()
for train_index,test_index in lo.split(X):
      print("Train Index:",train_index)
      print("Test Index:",test_index)
      print("X_train:",X[train_index])
      print("X_test:",X[test_index])
      print("")


# ---测试  cross_val_score 的用法
from sklearn.datasets import  load_digits
from sklearn.svm import  LinearSVC
digits=load_digits() # 加载用于分类问题的数据集
X=digits.data
y=digits.target
# 使用 LinearSVC 作为分类器
result=cross_val_score(LinearSVC(),X,y,cv=5)
print("Cross Val Score is:",result)


"""# ------分类问题性能度量"""
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,fbeta_score,classification_report,confusion_matrix,precision_recall_curve,roc_auc_score,roc_curve
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import  SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# ---测试 accuracy_score 的用法
y_true=[1,1,1,1,1,0,0,0,0,0]
y_pred=[0,0,1,1,0,0,1,1,0,0]
print('Accuracy Score(normalize=True):',accuracy_score(y_true,y_pred,normalize=True))
print('Accuracy Score(normalize=False):',accuracy_score(y_true,y_pred,normalize=False))

# ---测试 precision_score 的用法
y_true=[1,1,1,1,1,0,0,0,0,0]
y_pred=[0,0,1,1,0,0,0,0,0,0]
print('Accuracy Score:',accuracy_score(y_true,y_pred,normalize=True))
print('Precision Score:',precision_score(y_true,y_pred))

# ---测试 recall_score 的用法
y_true=[1,1,1,1,1,0,0,0,0,0]
y_pred=[0,0,1,1,0,0,0,0,0,0]
print('Accuracy Score:',accuracy_score(y_true,y_pred,normalize=True))
print('Precision Score:',precision_score(y_true,y_pred))
print('Recall Score:',recall_score(y_true,y_pred))

# ---测试 f1_score 的用法
y_true=[1,1,1,1,1,0,0,0,0,0]
y_pred=[0,0,1,1,0,0,0,0,0,0]
print('Accuracy Score:',accuracy_score(y_true,y_pred,normalize=True))
print('Precision Score:',precision_score(y_true,y_pred))
print('Recall Score:',recall_score(y_true,y_pred))
print('F1 Score:',f1_score(y_true,y_pred))

# ---测试 fbeta_score 的用法
y_true=[1,1,1,1,1,0,0,0,0,0]
y_pred=[0,0,1,1,0,0,0,0,0,0]
print('Accuracy Score:',accuracy_score(y_true,y_pred,normalize=True))
print('Precision Score:',precision_score(y_true,y_pred))
print('Recall Score:',recall_score(y_true,y_pred))
print('F1 Score:',f1_score(y_true,y_pred))
print('Fbeta Score(beta=0.001):',fbeta_score(y_true,y_pred,beta=0.001))
print('Fbeta Score(beta=1):',fbeta_score(y_true,y_pred,beta=1))
print('Fbeta Score(beta=10):',fbeta_score(y_true,y_pred,beta=10))
print('Fbeta Score(beta=10000):',fbeta_score(y_true,y_pred,beta=10000))

# ---测试 classification_report 的用法
y_true=[1,1,1,1,1,0,0,0,0,0]
y_pred=[0,0,1,1,0,0,0,0,0,0]
print('Classification Report:\n',classification_report(y_true,y_pred,target_names=["class_0","class_1"]))

# ---测试 confusion_matrix 的用法
y_true=[1,1,1,1,1,0,0,0,0,0]
y_pred=[0,0,1,1,0,0,0,0,0,0]
print('Confusion Matrix:\n',confusion_matrix(y_true,y_pred,labels=[0,1]))

# ---测试 precision_recall_curve 的用法，并绘制 P-R 曲线
### 加载数据
iris=load_iris()
X=iris.data
y=iris.target
# 二元化标记
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]
#### 添加噪音
np.random.seed(0)
n_samples, n_features = X.shape
X = np.c_[X, np.random.randn(n_samples, 200 * n_features)]
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.5,random_state=0)
### 训练模型
clf=OneVsRestClassifier(SVC(kernel='linear', probability=True,random_state=0))
clf.fit(X_train,y_train)
y_score = clf.fit(X_train, y_train).decision_function(X_test)
### 获取 P-R
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
precision = dict()
recall = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],y_score[:, i])
    ax.plot(recall[i],precision[i],label="target=%s"%i)
ax.set_xlabel("Recall Score")
ax.set_ylabel("Precision Score")
ax.set_title("P-R")
ax.legend(loc='best')
ax.set_xlim(0,1.1)
ax.set_ylim(0,1.1)
ax.grid()
plt.show()

# ---测试 roc_curve、roc_auc_score 的用法，并绘制 ROC 曲线
### 加载数据
iris=load_iris()
X=iris.data
y=iris.target
# 二元化标记
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]
#### 添加噪音
np.random.seed(0)
n_samples, n_features = X.shape
X = np.c_[X, np.random.randn(n_samples, 200 * n_features)]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=0)
### 训练模型
clf=OneVsRestClassifier(SVC(kernel='linear', probability=True,random_state=0))
clf.fit(X_train,y_train)
y_score = clf.fit(X_train, y_train).decision_function(X_test)
### 获取 ROC
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
fpr = dict()
tpr = dict()
roc_auc=dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i],y_score[:, i])
    roc_auc[i] = roc_auc_score(fpr[i], tpr[i])
    ax.plot(fpr[i],tpr[i],label="target=%s,auc=%s"%(i,roc_auc[i]))
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title("ROC")
ax.legend(loc="best")
ax.set_xlim(0,1.1)
ax.set_ylim(0,1.1)
ax.grid()
plt.show()


"""# ------回归问题性能度量"""
from sklearn.metrics import mean_absolute_error,mean_squared_error

# ---测试 mean_absolute_error 的用法
y_true=[1,1,1,1,1,2,2,2,0,0]
y_pred=[0,0,0,1,1,1,0,0,0,0]
print("Mean Absolute Error:",mean_absolute_error(y_true,y_pred))

# ---测试 mean_squared_error 的用法
y_true=[1,1,1,1,1,2,2,2,0,0]
y_pred=[0,0,0,1,1,1,0,0,0,0]
print("Mean Absolute Error:",mean_absolute_error(y_true,y_pred))
print("Mean Square Error:",mean_squared_error(y_true,y_pred))



"""# ------验证曲线"""
from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.model_selection import validation_curve

# ---测试 validation_curve 的用法 。验证对于 LinearSVC 分类器 ， C 参数对于预测准确率的影响
### 加载数据
digits = load_digits()
X,y=digits.data,digits.target
#### 获取验证曲线 ######
param_name="C"
param_range = np.logspace(-2, 2)
train_scores, test_scores = validation_curve(LinearSVC(), X, y, param_name=param_name,
         param_range=param_range,cv=10, scoring="accuracy")
###### 对每个 C ，获取 10 折交叉上的预测得分上的均值和方差 #####
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
####### 绘图 ######
fig=plt.figure()
ax=fig.add_subplot(1,1,1)

ax.semilogx(param_range, train_scores_mean, label="Training Accuracy", color="r")
ax.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
ax.semilogx(param_range, test_scores_mean, label="Testing Accuracy", color="g")
ax.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="g")

ax.set_title("Validation Curve with LinearSVC")
ax.set_xlabel("C")
ax.set_ylabel("Score")
ax.set_ylim(0,1.1)
ax.legend(loc='best')
plt.show()



