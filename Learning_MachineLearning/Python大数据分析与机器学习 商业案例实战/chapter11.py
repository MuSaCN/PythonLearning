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
__mypath__ = MyPath.MyClass_Path("\\Python大数据分析与机器学习 商业案例实战")  # 路径类
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
myKeras = MyDeepLearning.MyClass_tfKeras()  # tfKeras综合类
myTensor = MyDeepLearning.MyClass_TensorFlow()  # Tensorflow综合类
myMT5 = MyMql.MyClass_ConnectMT5(connect=False)  # Python链接MetaTrader5客户端类
myMT5Pro = MyMql.MyClass_ConnectMT5Pro(connect = False) # Python链接MT5高级类
#------------------------------------------------------------

# # 第十一章 特征工程之数据预处理（11.5）
# # 11.5 特征筛选：WOE值与IV值
# 11.5.3 WOE值与IV值的代码实现
# 构造数据
data = pd.DataFrame([[22,1],[25,1],[20,0],[35,0],[32,1],[38,0],[50,0],[46,1]], columns=['年龄', '是否违约'])
feature = data['年龄']
y = data['是否违约']
myDA.woe_iv(feature,y,cutbins=3, showprint=True)


# # 11.6 多重共线性的分析与处理
import pandas as pd
filepath = r"C:\Users\i2011\PycharmProjects\PythonLearning\Python大数据分析与机器学习 商业案例实战\第11章 特征工程之数据预处理\源代码汇总_PyCharm格式\数据.xlsx"
df = pd.read_excel(filepath)
df.head()
X = df.drop(columns='Y')
y = df['Y']
myDA.multicollinearity(X)


# # 11.7 过采样和欠采样
# 11.7.1 过采样
import pandas as pd
filepath = r"C:\Users\i2011\PycharmProjects\PythonLearning\Python大数据分析与机器学习 商业案例实战\第11章 特征工程之数据预处理\源代码汇总_PyCharm格式\信用卡数据.xlsx"
data = pd.read_excel(filepath)
data.head()
X = data.drop(columns='分类')
y = data['分类']
myDA.Counter(y)

X_oversampled, y_oversampled = myDA.RandomOverSampler(X,y,random_state=0)
print(myDA.Counter(y_oversampled))
print(X_oversampled.shape)

# SMOTE法 过采样，它是一种针对随机过采样容易导致过拟合问题的改进方案。它随机选取少数类中一个样本点，然后找到离该样本最近的n=4个样本点。在选中的样本点和最近的4个样本点分别连成的4条线段上随机选取4点生成新样本点。
X_smotesampled, y_smotesampled = myDA.SMOTE(X,y)
print(myDA.Counter(y_smotesampled))

# 欠采样
X_undersampled, y_undersampled = myDA.RandomUnderSampler(X,y)
print(myDA.Counter(y_undersampled))
print(X_undersampled.shape)








