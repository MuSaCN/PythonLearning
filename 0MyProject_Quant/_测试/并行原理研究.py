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
多进程原理：0 调动 1-8；main之外的代码会多次运行；main之内的非多进程代码，只有0会运行。
# 1.主程序，我们称为 0部分，在运行到 if __name__ == '__main__' 里多进程语句时，操作系统会分配多个部分，比如 1-8部分。
# 2.1-8部分会先分别 重新运行下 if __name__ == '__main__' 之外的代码。然后直接跳到对应的多进程代码。
# 3.所以main之外的代码会多次运行。
# 4.0部分 会接受 1-8部分 的运行结果。然后继续执行main之内的非多进行代码。
# 5.所以main之内的非多进程代码，只有0会运行。
# 6.如果 0部分 再次遇到多进程代码，操作系统会再次分配。同时，再次多次运行main之外的代码。之后，直接跳到对应的多进程代码。
# 7.所以多次执行多进程时，要小心，要注意原理。要定义 多个func函数 来实现不同的目的。
'''

# if __name__ == '__main__' 之内的其他语句，会只执行一次。


import warnings
warnings.filterwarnings('ignore')
# ---获取数据
eurusd = myMT5Pro.getsymboldata("EURUSD","TIMEFRAME_D1",[2010,1,1,0,0,0],[2020,1,1,0,0,0],index_time=True)

# ---计算信号，仅分析做多信号
price = eurusd.close   # 设定价格为考虑收盘价
# 外部参数
k_end = 100
holding_end = 10

# 必须把总结果写成函数，且只能有一个参数，所以参数以列表或元组形式传递
temp = 0 # 用来显示进度
def func(para):
    k = para[0]
    holding = para[1]
    # 打印进度
    global temp
    temp += 1
    print("\r", "{}/{}".format(temp*8, k_end * holding_end), end="", flush=True)
    # 退出条件
    if holding > k: return None
    # 获取信号数据
    signaldata = myBTV.stra.momentum(price, k=k, stra_mode="Continue")
    # 信号分析
    outStrat, outSignal = myBTV.signal_quality(signaldata["buysignal"], price_DataFrame=eurusd, holding=holding, lag_trade=1, plotRet=False, plotStrat=False)
    # 设置信号统计
    out = outStrat["BuyOnly"]
    winRate = out["winRate"]
    cumRet = out["cumRet"]
    sharpe = out["sharpe"]
    maxDD = out["maxDD"]
    count = out["TradeCount"]
    marketRet = outSignal["市场收益率"]
    out["k"] = k
    out["holding"] = holding
    # ---
    result = pd.DataFrame()  # 要放到里面
    if cumRet > marketRet and cumRet > 0 and sharpe > 0:
        result = result.append(out, ignore_index=True)
    return result

# 设定参数
para = [(k, holding) for k in range(1, k_end + 1) for holding in range(1, holding_end + 1)]

# 会多次执行，8核会执行9次(主1次，分配核8次) * 并行次数
print("开始并行研究！！！！！！！！！！！！！！")


# 多进程必须要在这里写
if __name__ == '__main__':
    import timeit
    t0 = timeit.default_timer()  # 主程序执行，只执行1次。
    print("第一次并行！！！！！！！！！！！！！！") # 主程序执行，只执行1次。
    print("计算次数为 = ",k_end * holding_end) # 主程序执行，只执行1次。结果为1000
    # 必须要写在里面
    out = myBTV.muiltcore.multi_processing(func , para) # 并行执行，所以总次数为1000
    # 由于out结果为list，需要分开添加
    result = []
    for i in out:
        result.append(i)
    result = pd.concat(result, ignore_index=True)  # 可以自动过滤None
    t1 = timeit.default_timer()
    print("\n",'第一次并行 耗时为：', t1 - t0)  # 耗时为：20
    print("第二次并行！！！！！！！！！！！！！！")  # 主程序执行，只执行1次。

    t0 = timeit.default_timer()  # 主程序执行，只执行1次。
    k_end = 10                   # 主程序执行，并行不执行
    holding_end = 10             # 主程序执行，并行不执行
    out = myBTV.muiltcore.multi_processing(func, para)  # 并行执行，由于先运行main外部代码，然后直接跳到这句。所以总次数依然为1000，而不是100。
    print("计算次数为 = ",k_end * holding_end) # 主程序执行，所以结果为100
    # 由于out结果为list，需要分开添加
    result = []
    for i in out:
        result.append(i)
    result = pd.concat(result, ignore_index=True)  # 可以自动过滤None
    t1 = timeit.default_timer()
    print("\n", '第二次并行 耗时为：', t1 - t0)  # 耗时为：20







