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

# ---二元化
from sklearn.preprocessing import Binarizer
X=[   [1,2,3,4,5],[5,4,3,2,1],[3,3,3,3,3,],[1,1,1,1,1]  ]
print("before transform:",X)
binarizer=Binarizer(threshold=2.5)
print("after transform:",binarizer.transform(X))


# ---独热码编码
from sklearn.preprocessing import OneHotEncoder
X=[   [1,2,3,4,5],[5,4,3,2,1],[3,3,3,3,3,],[1,1,1,1,1] ]
print("before transform:",X)
encoder=OneHotEncoder(sparse=False)
encoder.fit(X)
print("after transform:",encoder.transform( [[1,2,3,4,5]]))


# ---数据标准化
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler
# 测试 MinMaxScaler 的用法
X=[   [1,5,1,2,10],[2,6,3,2,7],[3,7,5,6,4,],[4,8,7,8,1] ]
print("before transform:",X)
scaler=MinMaxScaler(feature_range=(0,2))
scaler.fit(X)
print("min_ is :",scaler.min_)
print("scale_ is :",scaler.scale_)
print("data_max_ is :",scaler.data_max_)
print("data_min_ is :",scaler.data_min_)
print("data_range_ is :",scaler.data_range_)
print("after transform:",scaler.transform(X))

# 测试 MaxAbsScaler 的用法
X=[   [1,5,1,2,10],[2,6,3,2,7],[3,7,5,6,4,],[4,8,7,8,1] ]
print("before transform:",X)
scaler=MaxAbsScaler()
scaler.fit(X)
print("scale_ is :",scaler.scale_)
print("max_abs_ is :",scaler.max_abs_)
print("after transform:",scaler.transform(X))

# 测试 StandardScaler 的用法
X=[   [1,5,1,2,10],[2,6,3,2,7],[3,7,5,6,4,],[4,8,7,8,1] ]
print("before transform:",X)
scaler=StandardScaler()
scaler.fit(X)
print("scale_ is :",scaler.scale_)
print("mean_ is :",scaler.mean_)
print("var_ is :",scaler.var_)
print("after transform:",scaler.transform(X))


# ---数据正则化
from sklearn.preprocessing import Normalizer
X=[ [1,2,3,4,5],[5,4,3,2,1],[1,3,5,2,4,],[2,4,1,3,5] ]
print("before transform:",X)
normalizer=Normalizer(norm='l2')
print("after transform:",normalizer.transform(X))


# ---过滤式特征选择
from sklearn.feature_selection import  VarianceThreshold,SelectKBest,f_classif
# 测试 VarianceThreshold  的用法
X=[[100,1,2,3], [100,4,5,6],[100,7,8,9],[101,11,12,13]]
selector=VarianceThreshold(1)
selector.fit(X)
print("Variances is %s"%selector.variances_)
print("After transform is %s"%selector.transform(X))
print("The surport is %s"%selector.get_support(True))
print("After reverse transform is %s"% selector.inverse_transform(selector.transform(X)))
# 测试 SelectKBest  的用法，其中考察的特征指标是 f_classif
X=[   [1,2,3,4,5], [5,4,3,2,1], [3,3,3,3,3,],[1,1,1,1,1] ]
y=[0,1,0,1]
print("before transform:",X)
selector=SelectKBest(score_func=f_classif,k=3)
selector.fit(X,y)
print("scores_:",selector.scores_)
print("pvalues_:",selector.pvalues_)
print("selected index:",selector.get_support(True))
print("after transform:",selector.transform(X))


# ---包裹式特征选择
from sklearn.feature_selection import  RFE,RFECV
from sklearn.svm import LinearSVC
from sklearn.datasets import  load_iris
from  sklearn import  model_selection

# 测试 RFE 的用法，其中目标特征数量为 2
iris=load_iris()
X=iris.data
y=iris.target
estimator=LinearSVC()
selector=RFE(estimator=estimator,n_features_to_select=2)
selector.fit(X,y)
print("N_features %s"%selector.n_features_)
print("Support is %s"%selector.support_)
print("Ranking %s"%selector.ranking_)

# 测试 RFECV 的用法
iris=load_iris()
X=iris.data
y=iris.target
estimator=LinearSVC()
selector=RFECV(estimator=estimator,cv=3)
selector.fit(X,y)
print("N_features %s"%selector.n_features_)
print("Support is %s"%selector.support_)
print("Ranking %s"%selector.ranking_)
print("Grid Scores %s"%selector.grid_scores_)

# 比较经过特征选择和未经特征选择的数据集，对 LinearSVC 的预测性能的区别
iris=load_iris()
X,y=iris.data,iris.target
### 特征提取
estimator=LinearSVC()
selector=RFE(estimator=estimator,n_features_to_select=2)
X_t=selector.fit_transform(X,y)
#### 切分测试集与验证集
X_train,X_test,y_train,y_test=model_selection.train_test_split(X, y, test_size=0.25,random_state=0,stratify=y)
X_train_t,X_test_t,y_train_t,y_test_t=model_selection.train_test_split(X_t, y, test_size=0.25,random_state=0,stratify=y)
### 测试与验证
clf=LinearSVC()
clf_t=LinearSVC()
clf.fit(X_train,y_train)
clf_t.fit(X_train_t,y_train_t)
print("Original DataSet: test score=%s"%(clf.score(X_test,y_test)))
print("Selected DataSet: test score=%s"%(clf_t.score(X_test_t,y_test_t)))


# ---嵌入式特征选择
from sklearn.feature_selection import  SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.datasets import  load_digits,load_diabetes
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# 测试 SelectFromModel 的用法。
digits=load_digits()
X=digits.data
y=digits.target
estimator=LinearSVC(penalty='l1',dual=False)
selector=SelectFromModel(estimator=estimator,threshold='mean')
selector.fit(X,y)
selector.transform(X)
print("Threshold %s"%selector.threshold_)
print("Support is %s"%selector.get_support(indices=True))

# 测试 alpha 与稀疏性的关系
alphas=np.logspace(-2,2)
zeros=[]
for alpha in alphas:
  regr=Lasso(alpha=alpha)
  regr.fit(X,y)
  ### 计算零的个数 ###
  num=0
  for ele in regr.coef_:
      if abs(ele) < 1e-5:num+=1
  zeros.append(num)
##### 绘图
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(alphas,zeros)
ax.set_xlabel(r"$\alpha$")
ax.set_xscale("log")
ax.set_ylim(0,X.shape[1]+1)
ax.set_ylabel("zeros in coef")
ax.set_title("Sparsity In Lasso")
plt.show()

# 测试 C  与 稀疏性的关系
Cs=np.logspace(-2,2)
zeros=[]
for C in Cs:
  clf=LinearSVC(C=C,penalty='l1',dual=False)
  clf.fit(X,y)
   ### 计算零的个数 ###
  num=0
  for row in clf.coef_:
      for ele in row:
          if abs(ele) < 1e-5:num+=1
  zeros.append(num)
##### 绘图
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(Cs,zeros)
ax.set_xlabel("C")
ax.set_xscale("log")
ax.set_ylabel("zeros in coef")
ax.set_title("Sparsity In SVM")
plt.show()


# ---流水线
from sklearn.svm import  LinearSVC
from sklearn.datasets import  load_digits
from sklearn import  model_selection
from sklearn.linear_model import LogisticRegression
from  sklearn.pipeline import Pipeline
# 测试 Pipeline 的用法
data=load_digits() # 生成用于分类问题的数据集
X_train,X_test,y_train,y_test=model_selection.train_test_split(data.data, data.target,test_size=0.25,random_state=0,stratify=data.target)
steps=[("Linear_SVM",LinearSVC(C=1,penalty='l1',dual=False)), ("LogisticRegression",LogisticRegression(C=1))]
pipeline=Pipeline(steps)
pipeline.fit(X_train,y_train)
print("Named steps:",pipeline.named_steps)
print("Pipeline Score:",pipeline.score(X_test,y_test))


# ---字典学习
from sklearn.decomposition import DictionaryLearning
X=[[1,2,3,4,5],[6,7,8,9,10],[10,9,8,7,6,],[5,4,3,2,1] ]
print("before transform:",X)
dct=DictionaryLearning(n_components=3)
dct.fit(X)
print("components is :",dct.components_)
print("after transform:",dct.transform(X))



