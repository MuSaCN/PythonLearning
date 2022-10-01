# Author:Zhang Yuan
import warnings
warnings.filterwarnings('ignore')

from MyPackage import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

#------------------------------------------------------------
__mypath__ = MyPath.MyClass_Path("")  # 路径类
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
myMT5Pro = MyMql.MyClass_ConnectMT5Pro(connect = False) # Python链接MT5高级类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------

'''
# 1.根据前面输出的优化结果，自动寻找最佳参数点。由于品种较多，再算上极值点判断方法，耗时较长，故采用多核运算。
# 2.自动寻找的思路为：对 过滤0次、过滤1次、过滤2次 的数据寻找极值点。会输出图片和表格。注意过滤后的数据判断完极值后，会根据其位置索引到源数据，再组成表格的内容。注意图片中的过滤部分极值，并没有更改为源数据，仅表格更改了。
# 3.并行运算必须处理好图片释放内存的问题，且并行逻辑与目录逻辑不一样要一样。此处是以品种作为并行方案。
# 4.根据输出的图片看过滤几次较好，以及判断极值每一边用有多少点进行比较较好。
# 5.为下一步批量自动回测做准备。
'''

#%%
from MyPackage.MyProjects.向量化策略测试.Strategy_Param_Opt import Auto_Choose_StratOptParam
choose_opt = Auto_Choose_StratOptParam()
myDefault.set_backend_default("agg") # 这句必须放到类下面

#%% ************ 需要修改的部分 ************
choose_opt.total_folder = "F:\\工作---策略研究\\简单的动量反转\\_反转研究"
choose_opt.filename_prefix = "反转"
choose_opt.symbol_list = myMT5Pro.get_mainusd_symbol_name_list()
choose_opt.para_fixed_list = [{"k":None, "holding":1, "lag_trade":1}] # 仅检测 holding=1

#%%
choose_opt.y_name = ["sharpe"] # 过滤的y轴，不能太多。仅根据夏普选择就可以了.
choose_opt.core_num = -1 # -1表示留1个进程不执行运算。
# ---多进程必须要在这里执行
if __name__ == '__main__':
    # ---
    choose_opt.main_func()









