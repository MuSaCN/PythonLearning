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
# 函数式 API 实现
# 简单的例子
from keras import Input
input_tensor = Input(shape=(64,))

from keras import layers
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

# Model 类将输入张量和输出张量转换为一个模型
from keras.models import  Model
model = Model(input_tensor, output_tensor)

model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# 生成用于训练的虚构 Numpy 数据
import numpy as np
x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))
model.fit(x_train, y_train, epochs=10, batch_size=128)
score = model.evaluate(x_train, y_train)


#%%
# 用函数式 API 实现双输入问答模型
from keras import Input
from keras import layers

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

# ---输入1，文本输入是一个长度可变的整数序列。注意，你可以选择对输入进行命名。
text_input = Input(shape=(None,), dtype='int32', name='text')
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

# ---输入2，对问题进行相同的处理（使用不同的层实例）
question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

# 将编码后的问题和文本连接起来
concatenated = layers.concatenate([encoded_text, encoded_question],axis=-1)
# 在上面添加一个softmax 分类器
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

# 在模型实例化时，指定两个输入和输出
from keras.models import Model
model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# 将数据输入到多输入模型中
import numpy as np
num_samples = 1000
max_length = 100
# 生成虚构的 Numpy 数据
text = np.random.randint(1, text_vocabulary_size,size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size,size=(num_samples, max_length))
answers = np.random.randint(answer_vocabulary_size, size=(num_samples))
import keras
# 回答是 one-hot 编码的，不是整数
answers = keras.utils.to_categorical(answers, answer_vocabulary_size)
# 使用输入组成的列表来拟合
model.fit([text, question], answers, epochs=10, batch_size=128)
# 使用输入组成的字典来拟合（只有对输入进行命名之后才能用这种方法）
model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)


#%%
# 用函数式 API 实现一个三输出模型
from keras import Input
from keras import layers

vocabulary_size = 50000
num_income_groups = 10
posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)
# 注意，输出层都具有名称
age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups,activation='softmax',name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

from keras.models import Model
model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])
model.compile(optimizer='rmsprop',loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])
# 与上述写法等效（只有输出层具有名称时才能采用这种写法）
model.compile(optimizer='rmsprop',loss={'age': 'mse','income': 'categorical_crossentropy','gender': 'binary_crossentropy'})
# ---多输出模型的编译选项：损失加权
model.compile(optimizer='rmsprop',loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],loss_weights=[0.25, 1., 10.])
# 与上述写法等效（只有输出层具有名称时才能采用这种写法）
model.compile(optimizer='rmsprop',loss={'age': 'mse','income': 'categorical_crossentropy','gender': 'binary_crossentropy'},loss_weights={'age': 0.25,'income': 1.,'gender': 10.})
# ---将数据输入到多输出模型中
# 假设 age_targets 、 income_targets 和 gender_targets 都是 Numpy 数组
model.fit(posts, [age_targets, income_targets, gender_targets],epochs=10, batch_size=64)
# 与上述写法等效（只有输出层具有名称时才能采用这种写法）
model.fit(posts, {'age': age_targets,'income': income_targets,'gender': gender_targets},epochs=10, batch_size=64)


# ---使用 Keras 函数式 API 中的层共享（层重复使用）可以实现这样的模型
from keras import layers
from keras import Input

# 将一个 LSTM 层实例化一次
lstm = layers.LSTM(32)
# 构建模型的左分支：输入是长度 128 的向量组成的变长序列
left_input = Input(shape=(None, 128))
left_output = lstm(left_input)
# 构建模型的右分支：如果调用已有的层实例，那么就会重复使用它的权重
right_input = Input(shape=(None, 128))
right_output = lstm(right_input)
# 在上面构建一个分类器
merged = layers.concatenate([left_output, right_output], axis=-1)
predictions = layers.Dense(1, activation='sigmoid')(merged)

from keras.models import Model
model = Model([left_input, right_input], predictions)
model.fit([X_left, X_right], y)


# ---将模型作为层
from keras import layers
from keras import applications
from keras import Input
xception_base = applications.Xception(weights=None,include_top=False)
# 输入是 250×250 的 RGB 图像
left_input = Input(shape=(250, 250, 3))
right_input = Input(shape=(250, 250, 3))
# 对相同的视觉模型调用两次
left_features = xception_base(left_input)
right_input = xception_base(right_input)
# 合并后的特征包含来自左右两个视觉输入中的信息
merged_features = layers.concatenate([left_features, right_input], axis=-1)


