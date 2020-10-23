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
myPjMT5 = MyProject.MT5_MLLearning()  # MT5机器学习项目类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------

'''
# 0.这里的回测是建立在前面已经对策略的参数做了选择。
# 1.根据前面整理的自动选择的最佳参数表格文档，读取参数，再做原始的策略测试。
# 2.策略结果保存到 "策略参数自动选择\品种\auto_para_1D_{order}\原始策略回测_filter1" 文件夹下面。
# 3.策略测试所用的区间要增大。
# 4.回测结果较多，构成策略库供后续选择研究。
# 5.并行运算注意内存释放，并且不要一次性都算完，这样容易爆内存。分组进行并行。
'''

#%% 根据 非策略参数 定位文件 ###########################
# 策略内参数(非策略参数 symbol、timeframe、direct 会自动解析)
para_name = ["k", "holding", "lag_trade"]

#%%
# 自动策略测试 symbol = para[0]； order = para[1]； filter_level = para[2]；
def run_auto_stratgy_test(para):
    # para = ("EURUSD", 30, "filter1")
    symbol = para[0]
    order = para[1]
    filter_level = para[2]  # 选择哪个过滤表格"filter0, filter1, filter2".

    # ---文档定位 ******修改这里******
    folder_para1D = __mypath__.get_desktop_path() + "\\_动量研究\\策略参数自动选择\\{}\\auto_para_1D_{}".format(symbol, order)
    filepath_para1D = folder_para1D + "\\%s.%s.xlsx" % (symbol, filter_level)

    filecontent = pd.read_excel(filepath_para1D)

    # ---解析，显然没有内容则直接跳过
    for i in range(len(filecontent)):  # i=0
        # ---获取各参数和策略评价
        symbol = filecontent.iloc[i]["symbol"]
        timeframe = filecontent.iloc[i]["timeframe"]
        direct = filecontent.iloc[i]["direct"]

        # 策略参数 ******修改这里******
        k = filecontent.iloc[i][para_name[0]]
        holding = filecontent.iloc[i][para_name[1]]
        lag_trade = filecontent.iloc[i][para_name[2]]

        # 解析参数生成字符串变量，用于 添加策略图的标注 和 输出图片命名。
        para_str = myBTV.string_strat_para(para_name, [k, holding, lag_trade])

        # ---加载测试数据，由于不需要训练集、测试集数据，只需要对应时间即可。
        date_from, date_to = myPjMT5.get_date_range(timeframe, to_Timestamp=True)
        train_x0 = date_from
        train_x1 = myPjMT5.get_train_test(data=None, t0=date_from, t1=date_to, train_scale=0.8)
        # 把训练集的时间进行左右扩展
        bound_left, bound_right = myPjMT5.extend_train_time(train_t0=train_x0, train_t1=train_x1, extend_scale=0)
        # 再次重新加载下全部的数据
        data_total = myPjMT5.getsymboldata(symbol, timeframe, bound_left, bound_right, index_time=True, col_capitalize=True)

        # ---获取信号数据 ******修改这里******
        signaldata = myBTV.stra.momentum(data_total.Close, k=k, holding=holding, sig_mode=direct, stra_mode="Continue")
        signaldata_input = signaldata[direct]

        # ---保存输出图片
        savefig = folder_para1D + "\\原始策略回测_{}\\{}.{}.{}.png".format(filter_level, timeframe, direct, para_str)
        myBTV.plot_signal_no_filter_and_quality(signaldata_input, train_x0=train_x0, train_x1=train_x1, price_DataFrame=data_total, price_Series=data_total.Close, holding=holding, lag_trade=lag_trade, noRepeatHold=True, para_str=para_str, train_evalute=filecontent.iloc[i], savefig=savefig, batch=True)
        # 手动释放内存 与 下一次循环被覆盖 在内存中是不一样的。
        del data_total, signaldata, signaldata_input

        # ---显示进度
    print("自动原始策略回测 finished:", order, symbol, filter_level)


#%%
################# 多进程执行函数 ########################################
cpu_core = -1 # -1表示留1个进程不执行运算。
# ---多进程必须要在这里执行
if __name__ == '__main__':
    symbol_list = myPjMT5.get_all_symbol_name().tolist()
    order_list = [30, 40, 50]  # [30,40,50]
    filter_level_list = ["filter1"] # 仅回测过滤1次的数据就可以了
    # ---设置多步，以更好的控制进度，更好的释放内存。
    for order in order_list:
        para_muilt = [(symbol, order, filter_level) for symbol in symbol_list for filter_level in filter_level_list]
        import timeit
        # ---开始多核执行
        t0 = timeit.default_timer()
        myBTV.multi_processing(run_auto_stratgy_test, para_muilt, core_num=cpu_core)
        t1 = timeit.default_timer()
        print("\n", 'para_muilt_%s 耗时为：' % order, t1 - t0)







