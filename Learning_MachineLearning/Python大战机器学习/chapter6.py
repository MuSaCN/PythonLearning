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

# ------聚类和EM算法
centers=[[1,1],[2,2],[1,2],[10,20]] # 用于产生聚类的中心点
X, labels_true = myML.DataPre.make_datasets("blobs", n_samples=1000, centers=centers, cluster_std=0.5 )
# 绘制用于聚类的数据集
myML.Cluster.plot_discrete_scatter(X[:,0],X[:,1],labels_true)

# ---KMeans
from sklearn import  cluster

# 测试 KMeans 的用法
clst=cluster.KMeans().fit(X)
myML.Cluster.showModelTest(clst,X,labels_true)

# 测试 KMeans 的聚类结果随 n_clusters 参数的影响
nclu = list(range(1,10))
ninit = [10,20,30,40,50]
myML.Cluster.plotparam_cluster(X,labels_true,"cluster.KMeans()",drawParam=1,n_clusters=nclu)
myML.Cluster.plotparam_cluster(X,labels_true,"cluster.KMeans()",drawParam=2,plot3D=True,n_clusters=nclu,max_iter = ninit)

# 测试 KMeans 的聚类结果随 n_init 和 init  参数的影响
nums=range(1,10)
init = ["k-means++","random"]
myML.Cluster.plotparam_cluster(X,labels_true,"cluster.KMeans()",drawParam=2,n_init=nums,init = init)


# ---DBSCAN
centers=[[1,1],[2,2],[1,2],[10,20]] # 用于产生聚类的中心点
X, labels_true = myML.DataPre.make_datasets("blobs", n_samples=1000, centers=centers, cluster_std=0.5 )
from sklearn import  cluster

# 测试 DBSCAN 的用法
clst=cluster.DBSCAN()
predicted_labels=clst.fit_predict(X)
myML.Cluster.showModelTest(clst,X,labels_true)

# 测试 DBSCAN 的聚类结果随  eps 参数的影响
epsilons=np.logspace(-1,1.5)
myML.Cluster.plotparam_cluster(X,labels_true,"cluster.DBSCAN()",drawParam=1,logX=True,eps=epsilons)

# 测试 DBSCAN 的聚类结果随  min_samples 参数的影响
min_samples=range(1,100)
myML.Cluster.plotparam_cluster(X,labels_true,"cluster.DBSCAN()",drawParam=1,logX=False,min_samples=min_samples)


# ---AgglomerativeClustering
centers=[[1,1],[2,2],[1,2],[10,20]] # 用于产生聚类的中心点
X, labels_true = myML.DataPre.make_datasets("blobs", n_samples=1000, centers=centers, cluster_std=0.5 )
from sklearn import  cluster
from sklearn.metrics import adjusted_rand_score
# 测试 AgglomerativeClustering 的用法
clst=cluster.AgglomerativeClustering()
predicted_labels=clst.fit_predict(X)
print("ARI:%s"% adjusted_rand_score(labels_true,predicted_labels))

# 测试 AgglomerativeClustering 的聚类结果随 n_clusters 参数的影响
nums=range(1,50)
myML.plotML.plotparam_cluster(X,labels_true,"cluster.AgglomerativeClustering()",drawParam=1,n_clusters=nums)

# 测试 AgglomerativeClustering 的聚类结果随链接方式的影响
nums=range(1,50)
linkages=['ward','complete','average']
myML.plotML.plotparam_cluster(X,labels_true,"cluster.AgglomerativeClustering()",drawParam=2,n_clusters=nums,linkage=linkages)


# ---GMM
centers=[[1,1],[2,2],[1,2],[10,20]] # 用于产生聚类的中心点
X, labels_true = myML.DataPre.make_datasets("blobs", n_samples=1000, centers=centers, cluster_std=0.5 )
from sklearn import mixture
from sklearn.metrics import adjusted_rand_score

# 测试 GMM 的用法
clst=mixture.GaussianMixture()
clst.fit(X)
predicted_labels=clst.predict(X)
print("ARI:%s"% adjusted_rand_score(labels_true,predicted_labels))

# 测试 GMM 的聚类结果随 n_components 参数的影响
nums=range(1,20)
myML.plotML.plotparam_cluster(X,labels_true,"mixture.GaussianMixture()",n_components=nums)

# 测试 GMM 的聚类结果随协方差类型的影响
nums=range(1,20)
cov_types=['spherical','tied','diag','full']
myML.plotML.plotparam_cluster(X,labels_true,"mixture.GaussianMixture()",drawParam=2,n_components=nums,covariance_type=cov_types)




