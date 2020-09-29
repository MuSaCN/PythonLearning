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

# ---感知机模型
# data
train_data0 = myML.Neur.creat_testdata(n=20,no_linear=False,value0=10,value1=20,plot=False)
train_data1 = myML.Neur.creat_testdata(n=20,no_linear=True,value0=10,value1=20,plot=False)

# ---对线性可分数据集执行感知机的原始算法并绘制分离超平面
data=train_data0 #产生线性可分数据集
w_0= np.ones((3,1),dtype=float) # 初始化 权重
w,b,num = myML.Neur.perceptron_algorithm(data,w_0,eta=0.1,b_0=1) # 执行感知机的原始形式
### 绘图
myplt.set_backend()
myML.Neur.plot_samples(data,w=w,b=b)


# ---对线性可分数据集执行感知机的原始算法和对偶形式算法，并绘制分离超平面
data=train_data0
## 执行原始形式的算法
w_1,b_1,num_1=myML.Neur.perceptron_algorithm(data,w_0=np.ones((3,1),dtype=float),eta=0.1,b_0=1)
myML.Neur.Plot_Samples(data,w=w_1,b=b_1)
## 执行对偶形式的算法
import time
print(time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())))
w_2,b_2,num_2,alpha=myML.Neur.perceptron_algorithm_dual(train_data=data,alpha_0=np.zeros((data.shape[0],1)),eta=0.1,b_0=0)
myML.Neur.Plot_Samples(data,w=w_2,b=b_2)
print(time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())))
#
print("w_1,b_1",w_1,b_1)
print("w_2,b_2",w_2,b_2)

# 测试学习率对于感知机两种形式算法的收敛速度的影响
data=train_data0 # 线性可分数据集
etas=np.linspace(0.01,1,num=25,endpoint=False)
w_0,b_0,alpha_0=np.ones((3,1)),0,np.zeros((data.shape[0],1))
etas=np.linspace(0.01,1,num=25,endpoint=False)
nums1=[]
for eta in etas:
    _,_,num_1=myML.Neur.perceptron_algorithm(data,w_0=w_0,eta=eta,b_0=b_0) # 获取原始形式算法的迭代次数
    nums1.append(num_1)
fig=plt.figure()
fig.suptitle("perceptron")
ax=fig.add_subplot(1,1,1)
ax.set_xlabel(r'$\eta$')
ax.plot(etas,np.array(nums1),label='orignal iteraton times')
ax.legend(loc="best",framealpha=0.5)
plt.show()



# ------多层神经网络
from sklearn import neural_network
train_data=myML.Neur.CreatTestData(500,no_linear=True,value0=10,value1=20,datadim=2,plot=True)

# ---使用 MLPClassifier绘制预测结果
train_x=train_data[:,:-1]
train_y=train_data[:,-1]
clf=neural_network.MLPClassifier(activation='logistic',max_iter=1000)# 构造分类器实例
clf.fit(train_x,train_y) # 训练分类器
print(clf.score(train_x,train_y)) # 查看在训练集上的评价预测精度

## 用训练好的训练集预测平面上每一点的输出##
myML.Neur.Plot_Samples(train_data,2,instance=clf)

# ------神经网络模型：用于 iris 模型
## 加载数据集
iris = myML.DataPre.load_datasets("iris")# 使用 scikit-learn  自带的 iris 数据集
X=iris.data[:,0:2] # 使用前两个特征，方便绘图
Y=iris.target # 标记值
data=np.hstack((X,Y.reshape(Y.size,1)))
np.random.seed(0)
np.random.shuffle(data) # 混洗数据。因为默认的iris 数据集：前50个数据是类别0，中间50个数据是类别1，末尾50个数据是类别2.混洗将打乱这个顺序
X=data[:,:-1]
Y=data[:,-1]
train_x=X[:-30]
test_x=X[-30:] # 最后30个样本作为测试集
train_y=Y[:-30]
test_y=Y[-30:]

# 测试score曲线
a= {"a":[(10,),(30,),(100,),(5,5),(10,10),(30,30)]}
myML.plotML.PlotParam_Score(train_x,test_x,train_y,test_y,"neural_network.MLPClassifier()", drawParam=2,max_iter=range(10000,10010),hidden_layer_sizes=a["a"])
myML.plotML.PlotParam_Score(train_x,train_x,train_y,train_y,"neural_network.MLPClassifier()", drawParam=2,max_iter=range(10000,10010),hidden_layer_sizes=a["a"])


# 画结果
myML.Neur.plotparam_MLP_classifier(train_x,test_x,train_y,test_y,"neural_network.MLPClassifier()",activation=['logistic'],max_iter=[10000])

# 使用 MLPClassifier 预测调整后的 iris 数据集。考察不同的 hidden_layer_sizes 的影响
hidden_layer_sizes = [(10,), (30,), (100,), (5, 5), (10, 10), (30, 30)]  # 候选的 hidden_layer_sizes 参数值组成的数组
myML.Neur.plotparam_MLP_classifier(train_x,test_x,train_y,test_y,"neural_network.MLPClassifier()",hidden_layer_sizes=hidden_layer_sizes,activation=['logistic'],max_iter=[10000])

# 使用 MLPClassifier 预测调整后的 iris 数据集。考察不同的 activation 的影响
ativations = ["logistic", "tanh", "relu"]  # 候选的激活函数字符串组成的列表
myML.Neur.plotparam_MLP_classifier(train_x,test_x,train_y,test_y,"neural_network.MLPClassifier()",activation=ativations,max_iter=[10000], hidden_layer_sizes=[(30,)])

# 使用 MLPClassifier 预测调整后的 iris 数据集。考察不同的 algorithm 的影响
algorithms=["lbfgs","sgd","adam"] # 候选的算法字符串组成的列表
myML.Neur.plotparam_MLP_classifier(train_x,test_x,train_y,test_y,"neural_network.MLPClassifier()",solver=algorithms,activation=["tanh"],max_iter=[10000],hidden_layer_sizes=[(30,)])

# 使用 MLPClassifier 预测调整后的 iris 数据集。考察不同的学习率的影响
etas = [0.1, 0.01, 0.001, 0.0001]  # 候选的学习率值组成的列表
myML.Neur.plotparam_MLP_classifier(train_x,test_x,train_y,test_y,"neural_network.MLPClassifier()",learning_rate_init=etas,activation=["tanh"], max_iter=[1000000],hidden_layer_sizes=[(30,)], solver=['sgd'])

