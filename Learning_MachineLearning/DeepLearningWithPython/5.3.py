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


#%%
from tensorflow.keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

#%%
conv_base.summary()

#%%
import os
import numpy as np

original_dataset_dir = os.path.expandvars('%USERPROFILE%')+'\\.kaggle\\dogs-vs-cats'
base_dir = original_dataset_dir+'\\cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# 使用已知模型快速特征提取
train_features, train_labels = myKeras.extract_features_from_directory(conv_base,train_dir,2000,batch_size=20)
validation_features, validation_labels = myKeras.extract_features_from_directory(conv_base,validation_dir,1000,batch_size=20)
test_features, test_labels = myKeras.extract_features_from_directory(conv_base,test_dir,1000,batch_size=20)

#%%
reshapecount = np.array(train_features.shape[1:]).cumprod()[-1]
train_features = np.reshape(train_features, (2000, reshapecount))
validation_features = np.reshape(validation_features, (1000, reshapecount))
test_features = np.reshape(test_features, (1000, reshapecount))

#%%
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5)) #（注意要使用 dropout 正则化）
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))
myKeras.plot_acc_loss(history)


#%%
myKeras.clear_session()
from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#%%
model.summary()

#%%
# 冻结conv_base网络
print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))
conv_base.trainable = False
print('总共有 4 个权重张量，每层2个（主权重矩阵和偏置向量）。', len(model.trainable_weights))

#%%
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])
model,history = myKeras.cnn2D_fit_from_directory(model,train_dir,validation_dir,augmentation=True,flow_batch_size=20,epochs=30,plot=True)

#%%
myKeras.plot_acc_loss(history)
model.save(base_dir+'\\cats_and_dogs_small_3.h5')

#%%
conv_base.summary()

#%%
conv_base = myKeras.fine_tune_model(conv_base,'block5_conv1')

#%%
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])
model,history = myKeras.cnn2D_fit_from_directory(model,train_dir,validation_dir,augmentation=True,flow_batch_size=20,epochs=30,plot=True)

#%%
model.save(base_dir+'\\cats_and_dogs_small_4.h5')

#%%
myKeras.cnn2D_evaluate_from_directory(model,test_dir,flow_batch_size=20,steps=50)

