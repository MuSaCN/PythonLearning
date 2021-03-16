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
# 研究思路，类似 Permutation Test 置换检验：
分析 Indicator 与 Return 是否有关，不能直接用简单的相关系数，需要个基准。
0.今天的 Indicator 要与明天的 Return 匹配。
1.用 spearman 相关系数。
2.一次实验：随机生成一组白噪声(可以为正态分布、t分布)，计算 noise 与 Indicator 的相关系数。
3.重复许多次实验，得到相关系数序列。这个序列可以形成一个分布。能计算出 0.05 位置 和 0.95 位置的值。
4.计算某一参数 Indicator 和 Return 的相关系数，并且能得出其在上述分布中的概率。
5.不同参数下 Indicator 都有一个概率，这就会形成一个序列。通过序列能判断哪些参数区间有效。
注意：白噪声用于模拟收益率，所以这两者只能与 Indicator 比较。
'''


#%% ###################################
import warnings
warnings.filterwarnings('ignore')

# ---获取数据
eurusd = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2015,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True, col_capitalize=True)
# 研究指标与价格波动的关系，不需要区分训练集和测试集
# 并行运算不需要先大致判断下 收益率与指标 相关性范围。
# 向后移动，我们希望今天的特征能与明天的波动匹配，所以shift(-1)，不能是shift(1)
rate = eurusd.Rate.shift(-1)


#%%
# 获取非共线性的技术指标，输入指标名称及其泛化参数
indi_name="rsi"
indi_params = [("Close",i) for i in range(5,12+1)]


#%%
# 指标1个参数范围合理性测试，仅适合1个参数变化时分析。
totalstep = 10000
volatility = rate
total_data = eurusd
# 生成白噪声
np.random.seed(420)
noise_df = pd.DataFrame(np.random.randn(len(total_data), totalstep), index=total_data.index)

# 用于多核，计算概率，para 传递指标参数 indi_params 中元素
def cal_prob(para):
    prob = myDA.indi.indicator_param1D_prob(volatility, noise_df, indi_name, total_data[para[0]], para[1])
    print("\r", "{}/{} finished !".format(para[1] - indi_params[0][1], len(indi_params)), end="", flush=True)
    return prob

#%%
if __name__ == '__main__':
    # 并行运算
    multi_para = indi_params
    import timeit
    t0 = timeit.default_timer()
    prob_list = myBTV.muiltcore.multi_processing(cal_prob, multi_para, core_num=0)
    t1 = timeit.default_timer()
    print("\n", 'indicator_param1D_range 耗时为：', t1 - t0)

    # 画图，这里的index
    prob_series = pd.Series(prob_list, index=[i[1] for i in indi_params])
    prob_series.plot(title="不同参数下 %s 指标的概率分数" % indi_name)
    # 保存图片
    folder = __mypath__.get_desktop_path() + "\\__指标参数范围分析__"
    savefig = folder + "\\%s.png" % indi_name
    myplt.savefig(fname=None)
    plt.show()






