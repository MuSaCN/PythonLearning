# Author:Zhang Yuan
from MyPackage import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# ------------------------------------------------------------
__mypath__ = MyPath.MyClass_Path("")  # 路径类
mylogging = MyDefault.MyClass_Default_Logging(activate=False)  # 日志记录类，需要放在上面才行
myfile = MyFile.MyClass_File()  # 文件操作类
myword = MyFile.MyClass_Word()  # word生成类
myexcel = MyFile.MyClass_Excel()  # excel生成类
myini = MyFile.MyClass_INI()  # ini文件操作类
mytime = MyTime.MyClass_Time()  # 时间类
myparallel = MyTools.MyClass_ParallelCal()  # 并行运算类
myplt = MyPlot.MyClass_Plot()  # 直接绘图类(单个图窗)
mypltpro = MyPlot.MyClass_PlotPro()  # Plot高级图系列
myfig = MyPlot.MyClass_Figure(AddFigure=False)  # 对象式绘图类(可多个图窗)
myfigpro = MyPlot.MyClass_FigurePro(AddFigure=False)  # Figure高级图系列
mynp = MyArray.MyClass_NumPy()  # 多维数组类(整合Numpy)
mypd = MyArray.MyClass_Pandas()  # 矩阵数组类(整合Pandas)
mypdpro = MyArray.MyClass_PandasPro()  # 高级矩阵数组类
myDA = MyDataAnalysis.MyClass_DataAnalysis()  # 数据分析类
myDefault = MyDefault.MyClass_Default_Matplotlib()  # 画图恢复默认设置类
# myMql = MyMql.MyClass_MqlBackups() # Mql备份类
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
myKeras = MyDeepLearning.MyClass_tfKeras()  # tfKeras综合类
myTensor = MyDeepLearning.MyClass_TensorFlow()  # Tensorflow综合类
myMT5 = MyMql.MyClass_ConnectMT5(connect=False)  # Python链接MetaTrader5客户端类
myMT5Pro = MyMql.MyClass_ConnectMT5Pro(connect=False)  # Python链接MT5高级类
myMT5Indi = MyMql.MyClass_MT5Indicator()  # MT5指标Python版
myMT5Report = MyMT5Report.MyClass_StratTestReport(AddFigure=False)  # MT5策略报告类
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=False)  # 固定杠杆仓位类
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=False)  # 浮动杠杆仓位类
myMT5run = MyMql.MyClass_RunningMT5()  # Python运行MT5
myMT5code = MyMql.MyClass_CodeMql5()  # Python生成MT5代码
myMoneyM = MyTrade.MyClass_MoneyManage()  # 资金管理类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------

'''
梯度提升（CATBOOST）在交易系统开发中的应用. 初级的方法
https://www.mql5.com/zh/articles/8642
'''
myMT5Pro.__init__(connect=True)


#%% 准备数据
# 导入所需的 Python 模块：
import MetaTrader5 as mt5

from datetime import datetime
import random
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

mt5.initialize()

# check for gpu devices is availible
from catboost.utils import get_gpu_device_count
print('%i GPU devices' % get_gpu_device_count())

# 然后初始化所有全局变量：
LOOK_BACK = 250 # look_back — 分析历史的深度
MA_PERIOD = 15 # ma_period  — 用于计算价格增量的移动平均周期数
SYMBOL = 'EURUSD' # symbol — 应当在 MetaTrader 5 终端中载入的交易品种报价
MARKUP = 0.0001 # markup  — 用于自定义测试器的点差大小
TIMEFRAME = mt5.TIMEFRAME_H1 # timeframe  — 应当载入数据的时间框架
START = datetime(2020, 5, 1) # start, stop  — 数据范围
STOP = datetime(2021, 1, 1)

# 让我们编写一个函数，直接接收原始数据并创建一个包含训练所需列的数据帧：
def get_prices(look_back = 15):
    prices = pd.DataFrame(mt5.copy_rates_range(SYMBOL, TIMEFRAME, START, STOP), columns=['time', 'close']).set_index('time')
    # set df index as datetime
    prices.index = pd.to_datetime(prices.index, unit='s')
    prices = prices.dropna()
    ratesM = prices.rolling(MA_PERIOD).mean()
    ratesD = prices - ratesM
    for i in range(look_back):
        prices[str(i)] = ratesD.shift(i)
    return prices.dropna()
pr = get_prices(look_back=LOOK_BACK)


#%% 创建训练标签（随机抽样）
# 让我们考虑二元分类，其中模型将预测将训练示例确定为类0或1的概率。0和1可用于交易方向：买入或卖出。
# add_labels 函数随机（在最小、最大范围内）设置每笔交易的持续时间（以柱形为单位）。通过更改最大和最小持续时间，您可以更改交易采样频率。因此，如果当前价格大于下一个“rand”柱向前的价格，这就是卖出标签（1）。在相反的情况下，标签是0。让我们看看应用上述函数后数据集的外观：
def add_labels(dataset, min, max):
    labels = []
    for i in range(dataset.shape[0]-max): # i=0
        rand = random.randint(min, max)
        if dataset['close'][i] >= (dataset['close'][i + rand]): # 价格减少了
            labels.append(1.0) # sell
        elif dataset['close'][i] <= (dataset['close'][i + rand]): # 价格增长了
            labels.append(0.0) # buy
        else:
            labels.append(0.0)
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels'] = labels
    dataset = dataset.dropna()
    return dataset

pr = add_labels(dataset=pr, min=10, max=25)



#%% 开发自定义测试器
# 因为我们正在创建一个交易系统，所以最好有一个策略测试器来进行及时的模型测试。下面是此类测试器的示例：
# tester 函数接受一个数据集和一个“标记”（可选）并检查整个数据集，类似于在 MetaTrader 5 测试器中的操作。在每一个新柱都会检查一个信号（标签），当标签改变时，交易就会反转。因此，卖出信号作为结束买入头寸和打开卖出头寸的信号。
# 注意：思路可以，但是未来函数。
def tester(dataset, markup = 0.0):
    last_deal = int(2)
    last_price = 0.0
    report = [0.0]
    for i in range(dataset.shape[0]): # i=0
        pred = dataset['labels'][i]
        if last_deal == 2: # 第一次
            last_price = dataset['close'][i]
            last_deal = 0 if pred <=0.5 else 1 # 0信号记录买
            continue
        if last_deal == 0 and pred > 0.5: # 上一次买，当前卖
            last_deal = 1
            report.append(report[-1] - markup + (dataset['close'][i] - last_price))
            last_price = dataset['close'][i]
            continue
        if last_deal == 1 and pred <=0.5: # 上一次卖，当前买
            last_deal = 0
            report.append(report[-1] - markup + (last_price - dataset['close'][i]))
            last_price = dataset['close'][i]
    return report

pr = get_prices(look_back=LOOK_BACK)
pr = add_labels(pr, 10, 25)
rep = tester(dataset=pr, markup=MARKUP)
rep = tester(dataset=pr, markup=0.0003)
plt.plot(rep)
plt.show()


#%% 训练 CatBoost 模型

#splitting on train and validation subsets
X = pr[pr.columns[1:-1]]
y = pr[pr.columns[-1]]
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size = 0.5, test_size = 0.5, shuffle=True) # False True

#learning with train and validation subsets
# 以下是模型参数的简要说明：
# iterations — 模型中树的最大数目。模型在每次迭代后都会增加弱模型（树）的数量，因此请确保设置足够大的值。根据我的实践，对于这个特定的例子，1000次迭代通常已经足够了。
# depth  — 每棵树的深度。深度越小，模型越粗糙-输出的交易越少。深度在6到10之间似乎是最佳的。
# learning_rate  — 梯度步长值；这与神经网络中使用的原理相同。合理的参数范围为0.01～0.1。值越低，模型训练的时间就越长。但在这种情况下，它可以找到更好的结果。
# custom_loss, eval_metric  — 用于评估模型的度量。分类的经典标准是“准确度”
# use_best_model  — 在每一步中，模型都会评估“准确性”，这可能会随着时间的推移而改变。此标志允许以最小的误差保存模型，否则最后一次迭代得到的模型将被保存。
# task_type  — 允许在GPU上训练模型（默认情况下使用CPU）。这只适用于非常大的数据；在其他情况下，在GPU内核上执行训练的速度比在处理器上执行训练的速度慢。
# early_stopping_rounds  — 该模型有一个内置的过拟合检测器，其工作原理简单。如果度量在指定的迭代次数内停止减少/增加（对于“精确度”，它停止增加），则训练停止。
model = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.01, custom_loss=['Accuracy'],
                           eval_metric='Accuracy', verbose=True, use_best_model=True, task_type='CPU')
model.fit(train_X, train_y, eval_set = (test_X, test_y), early_stopping_rounds=50, plot=False)

#test the learned model
p = model.predict_proba(X)
p2 = [x[0]<0.5 for x in p]
pr2 = pr.iloc[:len(p2)].copy()
pr2['labels'] = p2
rep = tester(pr2, MARKUP)
plt.plot(rep)
plt.show()


#%% 将模型移植到 MetaTrader 5
def export_model_to_MQL_code(model):
    model.save_model('catmodel.h',
           format="cpp",
           export_parameters=None,
           pool=None)
    # 创建一个字符串，并使用标准 Python 函数将 C++ 代码解析为MQL5：
    code = 'double catboost_model' + '(const double &features[]) { \n'
    code += '    '
    with open('catmodel.h', 'r') as file:
        data = file.read()
        code += data[data.find("unsigned int TreeDepth"):data.find("double Scale = 1;")]
    code +='\n\n'
    code+= 'return ' + 'ApplyCatboostModel(features, TreeDepth, TreeSplits , BorderCounts, Borders, LeafValues); } \n\n'

    code += 'double ApplyCatboostModel(const double &features[],uint &TreeDepth_[],uint &TreeSplits_[],uint &BorderCounts_[],float &Borders_[],double &LeafValues_[]) {\n\
    uint FloatFeatureCount=ArrayRange(BorderCounts_,0);\n\
    uint BinaryFeatureCount=ArrayRange(Borders_,0);\n\
    uint TreeCount=ArrayRange(TreeDepth_,0);\n\
    bool     binaryFeatures[];\n\
    ArrayResize(binaryFeatures,BinaryFeatureCount);\n\
    uint binFeatureIndex=0;\n\
    for(uint i=0; i<FloatFeatureCount; i++) {\n\
       for(uint j=0; j<BorderCounts_[i]; j++) {\n\
          binaryFeatures[binFeatureIndex]=features[i]>Borders_[binFeatureIndex];\n\
          binFeatureIndex++;\n\
       }\n\
    }\n\
    double result=0.0;\n\
    uint treeSplitsPtr=0;\n\
    uint leafValuesForCurrentTreePtr=0;\n\
    for(uint treeId=0; treeId<TreeCount; treeId++) {\n\
       uint currentTreeDepth=TreeDepth_[treeId];\n\
       uint index=0;\n\
       for(uint depth=0; depth<currentTreeDepth; depth++) {\n\
          index|=(binaryFeatures[TreeSplits_[treeSplitsPtr+depth]]<<depth);\n\
       }\n\
       result+=LeafValues_[leafValuesForCurrentTreePtr+index];\n\
       treeSplitsPtr+=currentTreeDepth;\n\
       leafValuesForCurrentTreePtr+=(1<<currentTreeDepth);\n\
    }\n\
    return 1.0/(1.0+MathPow(M_E,-result));\n\
    }'

    file = open('cat_model' + '.mqh', "w")
    file.write(code)
    file.close()
    print('The file ' + 'cat_model' + '.mqh ' + 'has been written to disc')

# 经过训练的模型对象被输入到函数中，然后以C++格式保存对象：
model.save_model('catmodel.h', format="cpp", export_parameters=None, pool=None)

export_model_to_MQL_code(model)


