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
__mypath__ = MyPath.MyClass_Path("\\Python机器学习基础教程")  # 路径类
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
# mySMT5 = MyMql.MyClass_SocketMT5() # 以Socket方式建立python与MT5通信类
# myBaidu = MyWebCrawler.MyClass_BaiduPan() # Baidu网盘交互类
# myImage = MyImage.MyClass_ImageProcess()  # 图片处理类
# myDefault = MyDefault.MyClass_Default_Matplotlib() # matplotlib默认设置
myWebQD = MyWebCrawler.MyClass_WebQuotesDownload()  # 金融行情下载类
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector()  # 向量型回测类
myML = MyMachineLearning.MyClass_MachineLearning()  # 机器学习综合类
#------------------------------------------------------------
# %%
from MyPackage.bookcode.preamble import *

# %%
# 生成(小斑点)聚类算法的测试数据；
X, y = myML.DataPre.make_datasets("blobs",centers=2, random_state=4, n_samples=30)
# plot dataset
myML.plotML.plot_discrete_scatter(X[:, 0], X[:, 1], hue=y)
print("X.shape:", X.shape)

# %%
# 生成自定义 随机波动数据
X, y = myML.DataPre.make_datasets("wave",n_samples=40)
plt.plot(X, y, 'o'); plt.ylim(-3, 3); plt.xlabel("Feature"); plt.ylabel("Target");

# %%
cancer = myML.DataPre.load_datasets("breast_cancer")
print("cancer.keys():\n", cancer.keys())
print("Shape of cancer data:", cancer.data.shape)
print("Sample counts per class:\n",{n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})
print("Feature names:\n", cancer.feature_names)

# %%
boston = myML.DataPre.load_datasets("boston")
print("Data shape:", boston.data.shape)

# %%
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
boston = myML.DataPre.load_datasets("boston")
X = boston.data
X = MinMaxScaler().fit_transform(boston.data) # 缩放到0-1
X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X) # 生成深度2的多项式 [1, a, b, a^2, ab, b^2]
y = boston.target
print("X.shape:", X.shape)

# %%
# k-Nearest Neighbors
# 生成(小斑点)聚类算法的测试数据；
X, Y = myML.DataPre.make_datasets("blobs",centers=2, random_state=4, n_samples=30)
X_test = np.array([[8.2, 3.66214339], [9.9, 3.2], [11.2, .5]])
myML.KNN.plot_knn_classification(X,Y,X_test,n_neighbors=5,show=True)

# %%
from MyPackage.bookcode.preamble import *
X,y = myML.DataPre.make_datasets("blobs",forge=True, centers=2, random_state=4, n_samples=30)
X_train, X_test, y_train, y_test = myML.DataPre.train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print("Test set predictions:", clf.predict(X_test))
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

# %% md
##### Analyzing KNeighborsClassifier
n_neighbors = [1, 3, 9]
myML.plotML.plotparam_classifier_boundaries(X,y,"neighbors.KNeighborsClassifier()",n_neighbors=n_neighbors)


# %%
cancer = myML.DataPre.load_datasets("breast_cancer")
X_train, X_test, y_train, y_test = myML.DataPre.train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)
myML.plotML.PlotParam_Score(X_train, X_test, y_train, y_test,"neighbors.KNeighborsClassifier()",drawParam=1,n_neighbors=neighbors_settings)


# %% md
##### k-neighbors regression
from MyPackage.bookcode.preamble import *

# %%
X, y = myML.DataPre.make_datasets("wave",n_samples=40)
X_test = np.array([[-1.5], [0.9], [1.5]])
myML.KNN.plot_knn_regression(X, y,X_test,n_neighbors=3)

# %%
from sklearn.neighbors import KNeighborsRegressor
X, y = myML.DataPre.make_datasets("wave",n_samples=40)
X_train, X_test, y_train, y_test = myML.DataPre.train_test_split(X, y, random_state=0)
reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)

# %%
print("Test set predictions:\n", reg.predict(X_test))
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))

# %% md
#### Analyzing KNeighborsRegressor
# %%
n_neighbors = [1,3,9]
myML.plotML.plotparam_regression_predict(X_train, X_test, y_train, y_test,"neighbors.KNeighborsRegressor()", n_neighbors=n_neighbors)

# %% md

##### Strengths, weaknesses, and parameters

# %% md
#### Linear Models
##### Linear models for regression
from MyPackage.bookcode.preamble import *

# %% md

#### Linear regression aka ordinary least squares

# %%
from sklearn.linear_model import LinearRegression
X, y = myML.DataPre.make_datasets("wave",n_samples=60)
X_train, X_test, y_train, y_test = myML.DataPre.train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)

# %%
print("lr.coef_:", lr.coef_)
print("lr.intercept_:", lr.intercept_)

# %%
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# %%
boston = myML.DataPre.load_datasets("boston")
X = boston.data
y = boston.target
X_train, X_test, y_train, y_test = myML.DataPre.train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

# %%
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# %% md

##### Ridge regression

# %%

from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

# %%

ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))

# %%

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))

# %%
plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-25, 25)
plt.legend()

# %%
X,y = myML.DataPre.load_datasets("boston",forge=True)

from sklearn.linear_model import Ridge, LinearRegression
myML.plotML.plotparam_learning_curve(Ridge(alpha=1), X, y,show=False)
myML.plotML.plotparam_learning_curve(LinearRegression(), X, y,show=True)

# %% md
##### Lasso

# %%

from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used:", np.sum(lasso.coef_ != 0))

# %%

# we increase the default setting of "max_iter",
# otherwise the model would warn us that we should increase max_iter.
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used:", np.sum(lasso001.coef_ != 0))

# %%

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used:", np.sum(lasso00001.coef_ != 0))

# %%
plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")

# %% md
##### Linear models for classification
from MyPackage.bookcode.preamble import *

# %%
X,y = myML.DataPre.make_datasets("blobs",forge=True, centers=2, random_state=4, n_samples=30)
myML.plotML.plotparam_classifier_boundaries(X,y,"linear_model.LogisticRegression()")
myML.plotML.plotparam_classifier_boundaries(X,y,"svm.LinearSVC()")


# %%
Cs = [1e-2, 10, 1e3]
myML.plotML.plotparam_classifier_boundaries(X,y,"svm.LinearSVC()",C=Cs, tol=[0.00001], dual=[False])

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = myML.DataPre.train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg = LogisticRegression().fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

# %%

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))

# %%

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))

# %%

plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()

# %%

for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(C=C, solver='liblinear', penalty="l1").fit(X_train, y_train)
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
        C, lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
        C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))

plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")

plt.ylim(-5, 5)
plt.legend(loc=3)

# %% md

##### Linear models for multiclass classification

# %%
from MyPackage.bookcode.preamble import *
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
X, y = make_blobs(random_state=42)
myML.plotML.plot_discrete_scatter(X[:, 0], X[:, 1], y)

# %%
myML.plotML.plotparam_classifier_boundaries(X,y,"svm.LinearSVC()")

# %% md

#### Strengths, weaknesses and parameters

# %%

# instantiate model and fit it in one line
logreg = LogisticRegression().fit(X_train, y_train)

# %%

logreg = LogisticRegression()
y_pred = logreg.fit(X_train, y_train).predict(X_test)

# %%
y_pred = LogisticRegression().fit(X_train, y_train).predict(X_test)

# %% md
### Naive Bayes Classifiers

# %%
X = np.array([[0, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

# %%
counts = {}
for label in np.unique(y):
    # iterate over each class
    # count (sum) entries of 1 per feature
    counts[label] = X[y == label].sum(axis=0)
print("Feature counts:\n", counts)

# %% md
#### Strengths, weaknesses and parameters
from MyPackage.bookcode.preamble import *

# %% md
### Decision trees
mglearn.plots.plot_animal_tree()

# %% md
##### Building decision trees

# %%
X, y = myML.DataPre.make_datasets("moons", n_samples=100, noise=0.25, random_state=3)
myML.plotML.plot_discrete_scatter(X[:, 0], X[:, 1], y)
myML.TreeModel.plottree_partition_tree(X,y,max_depth = 5)



# %% md

##### Controlling complexity of decision trees

# %%
from MyPackage.bookcode.preamble import *
from sklearn.tree import DecisionTreeClassifier
cancer = myML.DataPre.load_datasets("breast_cancer")
X_train, X_test, y_train, y_test = myML.DataPre.train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
# myML.TreeModel.PlotTree_Tree(tree)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# %% md

#### Analyzing Decision Trees

# %%
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)

import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))

# %% md

#### Feature Importance in trees

# %%
print("Feature importances:")
print(tree.feature_importances_)

# %%
myML.TreeModel.plot_feature_importances(tree,cancer.feature_names)

# %%

tree = mglearn.plots.plot_tree_not_monotone()
display(tree)

# %%

import os

ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))

plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel("Year")
plt.ylabel("Price in $/Mbyte")

# %%

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
# use historical data to forecast prices after the year 2000
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

# predict prices based on date
X_train = data_train.date[:, np.newaxis]
# we use a log-transform to get a simpler relationship of data to target
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

# predict on all data
X_all = ram_prices.date[:, np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

# undo log-transform
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

# %%

plt.semilogy(data_train.date, data_train.price, label="Training data")
plt.semilogy(data_test.date, data_test.price, label="Test data")
plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
plt.legend()

# %% md

#### Strengths, weaknesses and parameters

# %% md

#### Ensembles of Decision Trees
##### Random forests
###### Building random forests
###### Analyzing random forests

# %%
from MyPackage.bookcode.preamble import *
from sklearn.ensemble import RandomForestClassifier

X, y = myML.DataPre.make_datasets("moons", n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = myML.DataPre.train_test_split(X, y, stratify=y, random_state=42)

forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)

# %%
myML.TreeModel.plotforest_boundary(forest, X_train,y_train,ax=None)

# %%

X_train, X_test, y_train, y_test = myML.DataPre.train_test_split(cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

# %%
myML.TreeModel.plot_feature_importances(forest)


# %% md

###### Strengths, weaknesses, and parameters

# %% md
#### Gradient Boosted Regression Trees (Gradient Boosting Machines)

# %%
from sklearn.ensemble import GradientBoostingClassifier
cancer = myML.DataPre.load_datasets("breast_cancer")
X_train, X_test, y_train, y_test = myML.DataPre.train_test_split(cancer.data, cancer.target, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

# %%
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

# %%

gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

# %%

gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
myML.TreeModel.plot_feature_importances(gbrt)


# %% md

##### Strengths, weaknesses and parameters

# %% md
#### Kernelized Support Vector Machines
#### Linear Models and Non-linear Features
from MyPackage.bookcode.preamble import *
from mpl_toolkits.mplot3d import Axes3D
# %%

X, y =myML.DataPre.make_datasets("blobs",centers=4, random_state=8 )
y = y % 2
# myML.plotML.plot_discrete_scatter(X[:, 0], X[:, 1], y)

# %%
from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X, y)
# myML.plotML.plotparam_classifier_boundaries(X,y,"svm.LinearSVC()")

# %%
# add the squared first feature
X_new = np.hstack([X, X[:, 1:] ** 2])
# myfig.__init__(AddFigure=True)
# myfig.set_axes_3d2d()
# ax = myfig.plot3D_scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2],show=False)

# %%
Cs=[0.0001,1.0,10000]
myML.plotML.plotparam_classifier_boundaries(X_new, y,"svm.LinearSVC()",C=Cs)

# %%
linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
# show linear decision boundary
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
XX, YY = np.meshgrid(xx, yy)
ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
             alpha=0.5)
myML.plotML.plot_discrete_scatter(X[:, 0], X[:, 1], y,show=False)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# %% md
#### The Kernel Trick
#### Understanding SVMs

# %%
from MyPackage.bookcode.preamble import *
from sklearn.svm import SVC
X, y =myML.DataPre.make_datasets("blobs",forge=True,centers=2, random_state=4, n_samples=30)

svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plot support vectors
sv = svm.support_vectors_
# class labels of support vectors are given by the sign of the dual coefficients
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()


# %% md
#### Tuning SVM parameters
C=[0.1,1,1000]
gamma = [0.1,1,10]
myML.plotML.plotparam_classifier_boundaries(X,y,"svm.SVC()",kernel=['rbf'], C=C, gamma=gamma)

# %%
cancer = myML.DataPre.load_datasets("breast_cancer")
X_train, X_test, y_train, y_test = myML.DataPre.train_test_split(cancer.data, cancer.target, random_state=0)

svc = SVC()
svc.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))

# %%

plt.boxplot(X_train)
plt.yscale("symlog")
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")

# %% md

##### Preprocessing data for SVMs

# %%

# Compute the minimum value per feature on the training set
min_on_training = X_train.min(axis=0)
# Compute the range of each feature (max - min) on the training set
range_on_training = (X_train - min_on_training).max(axis=0)

# subtract the min, divide by range
# afterward, min=0 and max=1 for each feature
X_train_scaled = (X_train - min_on_training) / range_on_training
print("Minimum for each feature\n", X_train_scaled.min(axis=0))
print("Maximum for each feature\n", X_train_scaled.max(axis=0))

# %%

# use THE SAME transformation on the test set,
# using min and range of the training set. See Chapter 3 (unsupervised learning) for details.
X_test_scaled = (X_test - min_on_training) / range_on_training

# %%

svc = SVC()
svc.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
    svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))

# %%

svc = SVC(C=1000)
svc.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
    svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))

# %% md
#### Strengths, weaknesses and parameters
from MyPackage.bookcode.preamble import *

# %% md
### Neural Networks (Deep Learning)
#### The Neural Network Model

# %%
mglearn.plots.plot_logistic_regression_graph().view()
mglearn.plots.plot_single_hidden_layer_graph().view()

# %%

line = np.linspace(-3, 3, 100)
plt.plot(line, np.tanh(line), label="tanh")
plt.plot(line, np.maximum(line, 0), label="relu")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("relu(x), tanh(x)")

# %%

mglearn.plots.plot_two_hidden_layer_graph().view()

# %% md

#### Tuning Neural Networks

# %%
from sklearn.neural_network import MLPClassifier
X, y = myML.DataPre.make_datasets("moons",n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = myML.DataPre.train_test_split(X, y, stratify=y,random_state=42)

mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)

myML.plotML.plotparam_classifier_boundaries(X_train,y_train,"neural_network.MLPClassifier()",solver=['lbfgs'], random_state=[0])


# %%

mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# %%

# using two hidden layers, with 10 units each
mlp = MLPClassifier(solver='lbfgs', random_state=0,
                    hidden_layer_sizes=[10, 10])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# %%

# using two hidden layers, with 10 units each, now with tanh nonlinearity.
mlp = MLPClassifier(solver='lbfgs', activation='tanh',
                    random_state=0, hidden_layer_sizes=[10, 10])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# %%
hidden_layer_sizes=[10, 100]
alpha=[0.0001, 0.01, 0.1, 1]
myML.plotML.plotparam_classifier_boundaries(X_train,y_train,"neural_network.MLPClassifier()",hidden_layer_sizes=hidden_layer_sizes,alpha=alpha,solver=['lbfgs'], random_state=[0])

# %%

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for i, ax in enumerate(axes.ravel()):
    mlp = MLPClassifier(solver='lbfgs', random_state=i,
                        hidden_layer_sizes=[100, 100])
    mlp.fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)

# %%

print("Cancer data per-feature maxima:\n{}".format(cancer.data.max(axis=0)))

# %%

X_train, X_test, y_train, y_test = myML.DataPre.train_test_split(
    cancer.data, cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))

# %%

# compute the mean value per feature on the training set
mean_on_train = X_train.mean(axis=0)
# compute the standard deviation of each feature on the training set
std_on_train = X_train.std(axis=0)

# subtract the mean, and scale by inverse standard deviation
# afterward, mean=0 and std=1
X_train_scaled = (X_train - mean_on_train) / std_on_train
# use THE SAME transformation (using training mean and std) on the test set
X_test_scaled = (X_test - mean_on_train) / std_on_train

mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
    mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# %%

mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
    mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# %%

mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
    mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# %%

plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()

# %% md

#### Strengths, weaknesses and parameters
##### Estimating complexity in neural networks

# %% md
### Uncertainty estimates from classifiers
from MyPackage.bookcode.preamble import *

# %%
from sklearn.ensemble import GradientBoostingClassifier

X, y = myML.DataPre.make_datasets("circles",noise=0.25, factor=0.5, random_state=1)

# we rename the classes "blue" and "red" for illustration purposes:
y_named = np.array(["blue", "red"])[y]

# we can call train_test_split with arbitrarily many arrays;
# all will be split in a consistent manner
X_train, X_test, y_train_named, y_test_named, y_train, y_test = myML.DataPre.train_test_split(X, y_named, y, random_state=0)

# build the gradient boosting model
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)

# %% md

#### The Decision Function

# %%

print("X_test.shape:", X_test.shape)
print("Decision function shape:",gbrt.decision_function(X_test).shape)

# %%

# show the first few entries of decision_function
print("Decision function:", gbrt.decision_function(X_test)[:6])

# %%

print("Thresholded decision function:\n", gbrt.decision_function(X_test) > 0)
print("Predictions:\n", gbrt.predict(X_test))

# %%

# make the boolean True/False into 0 and 1
greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)
# use 0 and 1 as indices into classes_
pred = gbrt.classes_[greater_zero]
# pred is the same as the output of gbrt.predict
print("pred is equal to predictions:", np.all(pred == gbrt.predict(X_test)))

# %%

decision_function = gbrt.decision_function(X_test)
print("Decision function minimum: {:.2f} maximum: {:.2f}".format(np.min(decision_function), np.max(decision_function)))

# %%

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4,
                                fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1],
                                            alpha=.4, cm=mglearn.ReBl)

for ax in axes:
    # plot training and test points
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test,
                             markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,
                             markers='o', ax=ax)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
cbar = plt.colorbar(scores_image, ax=axes.tolist())
cbar.set_alpha(1)
cbar.draw_all()
axes[0].legend(["Test class 0", "Test class 1", "Train class 0",
                "Train class 1"], ncol=4, loc=(.1, 1.1))

# %% md

#### Predicting Probabilities

# %%

print("Shape of probabilities:", gbrt.predict_proba(X_test).shape)

# %%

# show the first few entries of predict_proba
print("Predicted probabilities:")
print(gbrt.predict_proba(X_test[:6]))

# %%

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

mglearn.tools.plot_2d_separator(
    gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(
    gbrt, X, ax=axes[1], alpha=.5, cm=mglearn.ReBl, function='predict_proba')

for ax in axes:
    # plot training and test points
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test,
                             markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,
                             markers='o', ax=ax)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
# don't want a transparent colorbar
cbar = plt.colorbar(scores_image, ax=axes.tolist())
cbar.set_alpha(1)
cbar.draw_all()
axes[0].legend(["Test class 0", "Test class 1", "Train class 0",
                "Train class 1"], ncol=4, loc=(.1, 1.1))

# %% md
#### Uncertainty in multiclass classification

iris = myML.DataPre.load_datasets("iris")

X_train, X_test, y_train, y_test = myML.DataPre.train_test_split(iris.data, iris.target, random_state=42)

gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(X_train, y_train)

# %%

print("Decision function shape:", gbrt.decision_function(X_test).shape)
# plot the first few entries of the decision function
print("Decision function:")
print(gbrt.decision_function(X_test)[:6, :])

# %%

print("Argmax of decision function:")
print(np.argmax(gbrt.decision_function(X_test), axis=1))
print("Predictions:")
print(gbrt.predict(X_test))

# %%

# show the first few entries of predict_proba
print("Predicted probabilities:")
print(gbrt.predict_proba(X_test)[:6])
# show that sums across rows are one
print("Sums:", gbrt.predict_proba(X_test)[:6].sum(axis=1))

# %%

print("Argmax of predicted probabilities:")
print(np.argmax(gbrt.predict_proba(X_test), axis=1))
print("Predictions:")
print(gbrt.predict(X_test))

# %%

logreg = LogisticRegression()

# represent each target by its class name in the iris dataset
named_target = iris.target_names[y_train]
logreg.fit(X_train, named_target)
print("unique classes in training data:", logreg.classes_)
print("predictions:", logreg.predict(X_test)[:10])
argmax_dec_func = np.argmax(logreg.decision_function(X_test), axis=1)
print("argmax of decision function:", argmax_dec_func[:10])
print("argmax combined with classes_:",
      logreg.classes_[argmax_dec_func][:10])

# %% md

### Summary and Outlook


