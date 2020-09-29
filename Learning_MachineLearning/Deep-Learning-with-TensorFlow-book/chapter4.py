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
__mypath__ = MyPath.MyClass_Path("\\Deep-Learning-with-TensorFlow-book")  # 路径类
myfile = MyFile.MyClass_File()  # 文件操作类
myword = MyFile.MyClass_Word()  # word生成类
myexcel = MyFile.MyClass_Excel()  # excel生成类
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
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector()  # 向量型回测类
myML = MyMachineLearning.MyClass_MachineLearning()  # 机器学习综合类
mySQL = MyDataBase.MyClass_MySQL(connect=False)  # MySQL类
mySQLAPP = MyDataBase.MyClass_SQL_APPIntegration()  # 数据库应用整合
myWebQD = MyWebCrawler.MyClass_QuotesDownload(tushare=False)  # 金融行情下载类
myWebR = MyWebCrawler.MyClass_Requests()  # Requests爬虫类
myWebS = MyWebCrawler.MyClass_Selenium(openChrome=False)  # Selenium模拟浏览器类
myWebAPP = MyWebCrawler.MyClass_Web_APPIntegration()  # 爬虫整合应用类
myEmail = MyWebCrawler.MyClass_Email()  # 邮箱交互类
myReportA = MyQuant.MyClass_ReportAnalysis()  # 研报分析类
myFactorD = MyQuant.MyClass_Factor_Detection()  # 因子检测类
myKeras = MyDeepLearning.MyClass_Keras()  # Keras综合类
myTensor = MyDeepLearning.MyClass_TensorFlow()  # Tensorflow综合类
#------------------------------------------------------------

# 需从书上摘取代码
import tensorflow as tf
a = 1.2 # python 语言方式创建标量
aa = tf.constant(1.2) # TF 方式创建标量
type(a), type(aa), tf.is_tensor(aa)
x = tf.constant([1, 2., 3.3])
a = tf.constant([1.2]) # 创建一个元素的向量
a, a.shape
a = tf.constant([1,2, 3.]) # 创建 3 个元素的向量
a, a.shape


a = tf.constant('Hello, Deep Learning.') # 创建字符串
# 在 tf.strings 模块中，提供了常见的字符串类型的工具函数，如小写化 lower()、拼接join()、长度 length()、切分 split()等。
tf.strings.lower(a)

a = tf.constant(True) # 创建布尔类型标量
a = tf.constant([True, False]) # 创建布尔类型向量
a = tf.constant(True) # 创建 TF 布尔张量
a is True # TF 布尔类型张量与 python 布尔类型比较
a == True # 仅数值比较

tf.constant(123456789, dtype=tf.int16)
tf.constant(123456789, dtype=tf.int32)
import numpy as np
np.pi # 从 numpy 中导入 pi 常量
tf.constant(np.pi, dtype=tf.float32) # 32 位
tf.constant(np.pi, dtype=tf.float64) # 64 位
print('before:',a.dtype) # 读取原有张量的数值精度
if a.dtype != tf.float32: # 如果精度不符合要求，则进行转换
    a = tf.cast(a,tf.float32) # tf.cast 函数可以完成精度转换
print('after :',a.dtype) # 打印转换后的精度

a = tf.constant(np.pi, dtype=tf.float16) # 创建 tf.float16 低精度张量
tf.cast(a, tf.double) # 转换为高精度张量

# TensorFlow 增加了一种专门的数据类型来支持梯度信息的记录：tf.Variable。tf.Variable 类型在普通的张量类型基础上添加了 name，trainable 等属性来支持计算图的构建。
a = tf.constant([-1, 0, 1, 2]) # 创建 TF 张量
aa = tf.Variable(a) # 转换为 Variable 类型
aa.name, aa.trainable # Variable 类型张量的属性
a = tf.Variable([[1,2],[3,4]]) # 直接创建 Variable 张量

tf.zeros([]),tf.ones([]) # 创建全 0，全 1 的标量
tf.ones([3, 2])
tf.zeros_like(a) # 创建一个与 a 形状相同，但是全 0 的新矩阵
tf.ones_like(a) # 创建一个与 a 形状相同，但是全 1 的新矩阵
# tf.*_like 是一系列的便捷函数，可以通过 tf.zeros(a.shape)等方式实现。

# 通过 tf.fill(shape, value)可以创建全为自定义数值 value 的张量
tf.fill([], -1) # 创建-1 的标量
tf.fill([1], -1) # 创建-1 的向量
tf.fill([2,2], 99) # 创建 2 行 2 列，元素全为 99 的矩阵

# 通过 tf.random.normal(shape, mean=0.0, stddev=1.0)可以创建形状为 shape，均值为mean，标准差为 stddev 的正态分布𝒩(mean,stddev 2 )。
tf.random.normal([2, 2], mean=1, stddev=2)
# 通过 tf.random.uniform(shape, minval=0, maxval=None, dtype=tf.float32)可以创建采样自[minval,maxval)区间的均匀分布的张量。
tf.random.uniform([2,2],maxval=10) # 创建采样自[0,10)均匀分布的矩阵

# tf.range(limit, delta=1)可以创建[0,limit)之间，步长为 delta 的整型序列，不包含 limit 本身。
tf.range(10) # 0~10，不包含 10
tf.range(1, 10, delta=2)  # 1~10


out = tf.random.uniform([4,10]) #随机模拟网络输出
y = tf.constant([2,3,2,0]) # 随机构造样本真实标签
y = tf.one_hot(y, depth=10) # one-hot 编码
loss = tf.keras.losses.mse(y, out) # 计算每个样本的 MSE
loss = tf.reduce_mean(loss) # 平均 MSE,loss 应是标量
print(loss)


x = tf.random.uniform([28,28],maxval=10,dtype=tf.int32)
x.shape
x = tf.expand_dims(x,axis=2) # axis=2 表示宽维度后面的一个维度
x.shape
x = tf.squeeze(x, axis=2) # 删除shape中为1的维度
x = tf.random.normal([2,32,32,3])
tf.transpose(x,perm=[0,3,1,2]) # 交换维度


b = tf.constant([1,2]) # 创建向量 b
b = tf.expand_dims(b, axis=0) # 插入新维度，变成矩阵
b = tf.tile(b, multiples=[2,1]) # 通过平铺一个给定的张量来构造一个张量。

