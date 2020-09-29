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

# ---主成分分析 PCA
from sklearn import decomposition
iris = myML.DataPre.load_datasets("iris") # 使用 scikit-learn 自带的 iris 数据集
X,y = iris.data,iris.target

# 测试 PCA 的用法 (注意：此PCA基于scipy.linalg来实现SVD分解，因此不能应用于实数矩阵，并且无法适用于超大规模数据。)
pca=decomposition.PCA(n_components=None) # 使用默认的 n_components
pca.fit(X)
print('explained variance ratio : %s'% str(pca.explained_variance_ratio_))

# 绘制经过 PCA 降维到二维之后的样本点
myML.DimReduce.plotparam_decomposition(X,y,"decomposition.PCA()",n_components=[2])

# 超大规模数据降维 IncrementalPCA
pca=decomposition.IncrementalPCA(n_components=None) # 使用默认的 n_components
pca.fit(X)
print('explained variance ratio : %s'% str(pca.explained_variance_ratio_))


# ---核化线性降维 KernelPCA
from sklearn import decomposition
iris = myML.DataPre.load_datasets("iris") # 使用 scikit-learn 自带的 iris 数据集
X,y = iris.data,iris.target

# 测试 KernelPCA 的用法
kernels=['linear','poly','rbf']
for kernel in kernels:
    kpca=decomposition.KernelPCA(n_components=None,kernel=kernel).fit(X) # 依次测试四种核函数
    print('kernel=%s --> lambdas: %s'% (kernel, kpca.lambdas_) )

# 绘制经过 KernelPCA 降维到二维之后的样本点
kernels=["linear","poly","rbf","sigmoid"]
myML.DimReduce.plotparam_decomposition(X,y,"decomposition.KernelPCA()",n_components=[2],kernel=kernels)

# 绘制经过 使用 poly 核的KernelPCA 降维到二维之后的样本点
p=[3,10]; gamma = [1,10] ; r = [1,10]
myML.DimReduce.plotparam_decomposition(X,y,"decomposition.KernelPCA()",gamma=gamma,degree=p,coef0=r)

# 绘制经过 使用 rbf 核的KernelPCA 降维到二维之后的样本点
Gammas=[0.5,1,4,10]# rbf 核的参数组成的列表。每个参数就是 gamma值
myML.DimReduce.plotparam_decomposition(X,y,"decomposition.KernelPCA()",n_components=[2],kernel=['rbf'],gamma=Gammas)

# 绘制经过 使用 sigmoid 核的KernelPCA 降维到二维之后的样本点
gamma = [0.01,0.1,0.2];
r =[0.1,0.2]
myML.DimReduce.plotparam_decomposition(X,y,"decomposition.KernelPCA()",n_components=[2],kernel=['sigmoid'],gamma=gamma,coef0=r)


# ---MDS
from sklearn import  manifold
iris = myML.DataPre.load_datasets("iris") # 使用 scikit-learn 自带的 iris 数据集
X,y = iris.data,iris.target

# 测试 MDS 的用法
for n in [4,3,2,1]: # 依次考察降维目标为 4维、3维、2维、1维
    mds=manifold.MDS(n_components=n)
    mds.fit(X)
    print('stress(n_components=%d) : %s'% (n, str(mds.stress_)))

# 绘制经过 使用 MDS 降维到二维之后的样本点
myML.DimReduce.plotparam_decomposition(X,y,"manifold.MDS()",n_components=[2])


# ---Isomap
from sklearn import  manifold
iris = myML.DataPre.load_datasets("iris") # 使用 scikit-learn 自带的 iris 数据集
X,y = iris.data,iris.target

# 测试 Isomap 的用法
for n in [4,3,2,1]: # 依次考察降维目标为 4维、3维、2维、1维
    isomap=manifold.Isomap(n_components=n)
    isomap.fit(X)
    print('reconstruction_error(n_components=%d) : %s'% (n, isomap.reconstruction_error()) )

# 测试 Isomap 中 n_neighbors 参数的影响，其中降维至 2维
Ks=[1,5,25,y.size-1] # n_neighbors参数的候选值的集合
myML.DimReduce.plotparam_decomposition(X,y,"manifold.Isomap()",n_components=[2],n_neighbors=Ks)

# 测试 Isomap 中 n_neighbors 参数的影响，其中降维至 1维
Ks=[1,5,25,y.size-1]# n_neighbors参数的候选值的集合
myML.DimReduce.plotparam_decomposition(X,y,"manifold.Isomap()",n_components=[1],n_neighbors=Ks)


# ---LLE
from sklearn import  manifold
iris = myML.DataPre.load_datasets("iris") # 使用 scikit-learn 自带的 iris 数据集
X,y = iris.data,iris.target

# 测试 LocallyLinearEmbedding 的用法
for n in [4,3,2,1]:# 依次考察降维目标为 4维、3维、2维、1维
    lle=manifold.LocallyLinearEmbedding(n_components=n)
    lle.fit(X)
    print('reconstruction_error(n_components=%d) : %s'% (n, lle.reconstruction_error_) )

# 测试 LocallyLinearEmbedding 中 n_neighbors 参数的影响，其中降维至 2维
Ks=[1,5,25,y.size-1]# n_neighbors参数的候选值的集合
myML.DimReduce.plotparam_decomposition(X,y,"manifold.LocallyLinearEmbedding()",n_components=[2],n_neighbors=Ks)

# 测试 LocallyLinearEmbedding 中 n_neighbors 参数的影响，其中降维至 1维
Ks=[1,5,25,y.size-1]# n_neighbors参数的候选值的集合
myML.DimReduce.plotparam_decomposition(X,y,"manifold.LocallyLinearEmbedding()",n_components=[1],n_neighbors=Ks)



