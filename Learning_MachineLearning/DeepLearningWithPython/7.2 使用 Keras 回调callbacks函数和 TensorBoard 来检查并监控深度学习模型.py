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
__mypath__ = MyPath.MyClass_Path("\\DeepLearningWithPython")  # 路径类
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



# 7.2　使用 Keras 回调callbacks函数和 TensorBoard 来检查并监控深度学习模型

# 1. ModelCheckpoint 与 EarlyStopping 回调函数
# 如果监控的目标指标在设定的轮数内不再改善，可以用 EarlyStopping 回调函数来中断训练。比如，这个回调函数可以在刚开始过拟合的时候就中断训练，从而避免用更少的轮次重新训练模型。这个回调函数通常与 ModelCheckpoint 结合使用，后者可以在训练过程中持续不断地保存模型（你也可以选择只保存目前的最佳模型，即一轮结束后具有最佳性能的模型）。
import keras
callbacks_list = [
    # 如果不再改善，就中断训练
    # monitor 监控模型的验证精度；patience 如果精度在多于一轮的时间（即两轮）内不再改善，中断训练
    keras.callbacks.EarlyStopping(monitor='acc',patience=1,),
    # 在每轮过后保存当前权重
    # filepath 目标模型文件的保存路径
    # monitor, save_best_only 这两个参数的含义是，如果 val_loss 没有改善，那么不需要覆盖模型文件。这就可以始终保存在训练过程中见到的最佳模型
    keras.callbacks.ModelCheckpoint(filepath='my_model.h5',monitor='val_loss',save_best_only=True,)
]

# metrics=['acc'] 你监控精度，所以它应该是模型指标的一部分
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
# 注意，由于回调函数要监控验证损失和验证精度，所以在调用 fit 时需要传入 validation_data （验证数据）
model.fit(x, y,epochs=10,batch_size=32,callbacks=callbacks_list, validation_data=(x_val, y_val))


# 2. ReduceLROnPlateau 回调函数
# 如果验证损失不再改善，你可以使用这个回调函数来降低学习率。在训练过程中如果出现了损失平台（loss plateau），那么增大或减小学习率都是跳出局部最小值的有效策略。
# monitor 监控模型的验证损失；factor 触发时将学习率除以 10；patience 如果验证损失在 10 轮内都没有改善，那么就触发这个回调函数
callbacks_list = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,)]
# 注意，因为回调函数要监控验证损失，所以你需要在调用 fit 时传入 validation_data （验证数据）
model.fit(x, y,epochs=10,batch_size=32,callbacks=callbacks_list,validation_data=(x_val, y_val))


# 3. 编写你自己的回调函数
import keras
import numpy as np
class ActivationLogger(keras.callbacks.Callback):
    def set_model(self, model):
        self.model = model
        layer_outputs = [layer.output for layer in model.layers]
        self.activations_model = keras.models.Model(model.input,
        layer_outputs)
    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data.')
        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)
        f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w')
        np.savez(f, activations)
        f.close()


# -------------使用了 TensorBoard 的文本分类模型
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
max_features = 2000
max_len = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
model = tensorflow.keras.models.Sequential()
model.add(layers.Embedding(max_features, 128,input_length=max_len,name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

# 为 TensorBoard 日志文件创建一个目录
import os
os.mkdir("C:\\Users\\i2011\\.keras\\my_log_dir")

# 使用一个 TensorBoard 回调函数来训练模型
# log_dir日志文件将被写入这个位置; histogram_freq每一轮之后记录激活直方图; embeddings_freq每一轮之后记录嵌入数据
callbacks = [
    tensorflow.keras.callbacks.TensorBoard(log_dir="C:\\Users\\i2011\\.keras\\my_log_dir",
                                           histogram_freq=1,embeddings_freq=1,)
]
history = model.fit(x_train, y_train,epochs=20,batch_size=128, validation_split=0.2,callbacks=callbacks)

# tensorboard --logdir="C:\\Users\\i2011\\.keras\\my_log_dir"
#  http://localhost:6006




