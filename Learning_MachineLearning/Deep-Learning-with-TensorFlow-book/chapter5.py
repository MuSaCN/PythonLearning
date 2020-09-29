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
__mypath__ = MyPath.MyClass_Path("\\Hands-On Machine Learning")  # 路径类
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
#------------------------------------------------------------


#%%
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets
import  os


#%%
a = tf.random.normal([4,35,8]) # 模拟成绩册A
b = tf.random.normal([6,35,8]) # 模拟成绩册B
a.shape
b.shape
c = tf.concat([a,b],axis=0) # 合并成绩册
c.shape

a = tf.random.normal([35,8])
b = tf.random.normal([35,8])
c = tf.stack([a,b],axis=0) # 堆叠合并为 2 个班级，班级维度插入在最前
c.shape


#%%
x = tf.random.normal([2,784])
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
o1 = tf.matmul(x,w1) + b1  #
o1 = tf.nn.relu(o1)

#%%
x = tf.ones([2,2])
tf.norm(x,ord=1) # 计算 L1 范数
tf.norm(x,ord=2) # 计算 L2 范数

x = tf.random.normal([4,10]) # 模型生成概率
tf.reduce_max(x,axis=1) # 统计概率维度上的最大值


out = tf.random.normal([2,10])
out = tf.nn.softmax(out, axis=1) # 通过 softmax 函数转换为概率值


#%%
x = tf.random.uniform([4,35,8],maxval=100,dtype=tf.int32) # 成绩册张量
tf.gather(x,[0,1],axis=0) # 在班级维度收集第 1~2 号班级成绩册
tf.gather(x,[0,3,8,11,12,26],axis=1)

#%%
a = tf.ones([3,3]) # 构造 a 为全 1 矩阵
b = tf.zeros([3,3]) # 构造 b 为全 0 矩阵
cond =tf.constant([[True,False,False],[False,True,False],[True,True,False]])
tf.where(cond)


#####################################################

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers
print(tf.__version__, keras.__version__)

def preprocess(x, y):
    # [b, 28, 28], [b]
    print(x.shape, y.shape)
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


# %%
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)
# %%
batchsz = 512
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000)
train_db = train_db.batch(batchsz)
train_db = train_db.map(preprocess)
train_db = train_db.repeat(20)

# %%
x_test.shape, y_test.shape
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(1000).batch(batchsz).map(preprocess)
x, y = next(iter(train_db))
x, y = next(iter(test_db))
print('train sample:', x.shape, y.shape)
print(x[0], y[0])


# %%
def main():
    # learning rate
    lr = 1e-2
    accs, losses = [], []
    # 784 => 512
    w1, b1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # 512 => 256
    w2, b2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1)), tf.Variable(tf.zeros([128]))
    # 256 => 10
    w3, b3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))
    for step, (x, y) in enumerate(train_db):
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 784))
        with tf.GradientTape() as tape:
            # layer1.
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            # layer2
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # output
            out = h2 @ w3 + b3
            # out = tf.nn.relu(out)
            # compute loss
            # [b, 10] - [b, 10]
            loss = tf.square(y - out)
            # [b, 10] => scalar
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        for p, g in zip([w1, b1, w2, b2, w3, b3], grads):
            p.assign_sub(lr * g)
        # print
        if step % 80 == 0:
            print(step, 'loss:', float(loss))
            losses.append(float(loss))
        if step % 80 == 0:
            # evaluate/test
            total, total_correct = 0., 0
            for x, y in test_db:
                # layer1.
                h1 = x @ w1 + b1
                h1 = tf.nn.relu(h1)
                # layer2
                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)
                # output
                out = h2 @ w3 + b3
                # [b, 10] => [b]
                pred = tf.argmax(out, axis=1)
                # convert one_hot y to number y
                y = tf.argmax(y, axis=1)
                # bool type
                correct = tf.equal(pred, y)
                # bool tensor => int tensor => numpy
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                total += x.shape[0]
            print(step, 'Evaluate Acc:', total_correct / total)
            accs.append(total_correct / total)
    plt.figure()
    x = [i * 80 for i in range(len(losses))]
    plt.plot(x, losses, color='C0', marker='s', label='训练')
    plt.ylabel('MSE')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('train.svg')

    plt.figure()
    plt.plot(x, accs, color='C1', marker='s', label='测试')
    plt.ylabel('准确率')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('test.svg')


if __name__ == '__main__':
    main()

