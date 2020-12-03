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
mylogging = MyDefault.MyClass_Default_Logging(activate=True, filename=__mypath__.get_desktop_path()+"\\范围过滤策略回测.log") # 日志记录类，需要放在上面才行

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
# 说明
# 这里的策略回测是建立在前面已经对指标的范围过滤做了参数选择。
# 前面对每个具体策略都通过指标过滤方式，算出了各个指标过滤效果的极值。我们根据极值对应的指标值做回测。
# 画的图中，min-max表示 "max最大的以max之前的min最小" 或 "min最小的以min之后的max最大"，start-end表示上涨额度最大的区间。
# 根据训练集获取过滤区间，然后作用到整个样本。
# 并行以品种来并行，以时间框来分组。
# 由于指标较多，并行运算时间长，防止出错输出日志。
'''

myplt.set_backend("agg")  # agg 后台输出图片，不占pycharm内存


#%% ************ 需要修改的部分 ************
# ******修改这里******
symbol_list = myMT5Pro.get_main_symbol_name_list()
total_folder = "F:\\工作---策略研究\\简单的动量反转\\_动量研究"
# ******修改这个函数****** sig_mode方向、stra_mode策略模式(默认值重要，不明写)、para_list策略参数
def stratgy_signal(price, sig_mode, stra_mode="Continue", para_list=list or tuple):
    return myBTV.stra.momentum(price=price, k=para_list[0], holding=para_list[1], sig_mode=sig_mode, stra_mode=stra_mode)


#%%
# ---多核函数，自动过滤策略回测，结果输出图片。
def run_auto_filter_stratgy_test(para):
    # 显示进度
    # para = ("EURUSD","TIMEFRAME_D1")
    print("\r", "当前执行参数为：", para, end="", flush=True)
    # 定位目录
    symbol = para[0]
    timeframe = para[1]
    in_folder0 = total_folder + "\\范围指标参数自动选择\\{}.{}".format(symbol, timeframe)
    # 判断是否存在，不存在则返回
    if __mypath__.path_exists(in_folder0) == False:
        return

    # ---以 特定参数的策略 作为研究对象
    folder_dir = __mypath__.listdir(in_folder0)
    for foldname in folder_dir:  # foldname = folder_dir[0]
        # 如果是文件，不是文件夹，则跳过
        if __mypath__.is_folder_or_file(in_folder0+"\\"+foldname, check_folder=False):
            continue
        # 解析下名称
        direct, suffix = foldname.split(".")[0:2]
        # 输入路径要重新设置下
        in_folder1 = in_folder0 + "\\" + foldname
        in_file = in_folder1 + "\\{}.{}.xlsx".format(suffix,"filter1") # 只分析 filter1
        filecontent = pd.read_excel(in_file)

        # ---解析，显然没有内容则直接跳过
        for i in range(len(filecontent)):  # i=1
            # ---获取各参数

            # 解析策略参数，-1索引固定为 lag_trade，-2索引固定为 holding.
            strat_para_list = myBTV.string_strat_para(strat_para=suffix)
            # [k, holding, lag_trade] = myBTV.string_strat_para(strat_para=suffix)

            # 解析下指标信息
            indi_name = filecontent.iloc[i]["indi_name"]
            indi_message=filecontent.iloc[i]["direct":"indi_name"][1:-1] # 要斩头去尾
            indi_para = [value for value in indi_message.dropna()] # 必须要丢弃nan

            # ---前面自动选择的指标参数排除在指定范围内，不一定要排除。******修改这里******
            # if indi_para[1] in [5,6,7]:
            #     continue

            # ---获取数据
            date_from, date_to = myMT5Pro.get_date_range(timeframe)
            data_total = myMT5Pro.getsymboldata(symbol, timeframe, date_from, date_to, index_time=True, col_capitalize=True)
            # 由于信号利润过滤是利用训练集的，所以要区分训练集和测试集
            data_train, data_test = myMT5Pro.get_train_test(data_total, train_scale=0.8)
            # 测试不需要把数据集区分训练集、测试集，仅画区间就可以了
            train_x0 = data_train.index[0]
            train_x1 = data_train.index[-1]
            # 把训练集的时间进行左右扩展
            bound_left, bound_right = myMT5Pro.extend_train_time(train_t0=train_x0, train_t1=train_x1, extend_scale=0)
            # 再次重新加载下全部的数据
            data_total = myMT5Pro.getsymboldata(symbol, timeframe, bound_left, bound_right, index_time=True, col_capitalize=True)

            # ---获取训练集和整个样本的信号
            # 获取训练集的信号
            signaldata_train = stratgy_signal(data_train.Close,sig_mode=direct,para_list=strat_para_list)
            signal_train = signaldata_train[direct]
            # 计算整个样本的信号
            signaldata_all = stratgy_signal(data_total.Close,sig_mode=direct,para_list=strat_para_list)
            signal_all = signaldata_all[direct]

            # ---(核心，在库中添加)获取指标
            indicator = myBTV.indiMT5.get_indicator_firstbuffer(data_total, indi_name, *indi_para)

            # ---信号利润过滤及测试
            # 输出图片的目录
            out_folder = in_folder1 + "\\指标过滤策略回测_filter1"
            # 指标参数字符串
            indi_para_name = []
            for index in range(len(indi_para)):
                indi_para_name.append("indi_para%s"%index)
            indi_suffix = myBTV.string_strat_para(indi_para_name, indi_para)
            savefig = out_folder + "\\{}.{}.png".format(indi_name,indi_suffix)
            # 过滤及测试后，输出图片
            myBTV.rfilter.plot_signal_range_filter_and_quality(signal_train=signal_train, signal_all=signal_all, indicator=indicator, train_x0=train_x0, train_x1=train_x1, price_DataFrame=data_total, price_Series=data_total.Close, holding=strat_para_list[-2], lag_trade=strat_para_list[-1], noRepeatHold=True, indi_name="%s%s" % (indi_name,indi_suffix), savefig=savefig, batch=True)
            del data_total, data_train, data_test, indicator, signaldata_train, signaldata_all, signal_train, signal_all
    # 打印下进度
    print(symbol, timeframe, "过滤策略回测完成！")


#%%
core_num = -1 # 注意，M1时间框数据量较大时，并行太多会爆内存。
if __name__ == '__main__':
    # ---
    def main_func(core_num):
        timeframe_list = ["TIMEFRAME_D1", "TIMEFRAME_H12", "TIMEFRAME_H8", "TIMEFRAME_H6",
                          "TIMEFRAME_H4", "TIMEFRAME_H3", "TIMEFRAME_H2", "TIMEFRAME_H1",
                          "TIMEFRAME_M30", "TIMEFRAME_M20", "TIMEFRAME_M15", "TIMEFRAME_M12",
                          "TIMEFRAME_M10", "TIMEFRAME_M6", "TIMEFRAME_M5", "TIMEFRAME_M4",
                          "TIMEFRAME_M3", "TIMEFRAME_M2", "TIMEFRAME_M1"]
        # 以时间框来分组
        finish_timeframe = []
        for timeframe in timeframe_list:
            # --- 1分钟时间框内存容易爆
            if timeframe == "TIMEFRAME_M1":
                core_num = 4
            # ---
            multi_params = [(symbol,timeframe) for symbol in symbol_list]
            import timeit
            t0 = timeit.default_timer()
            myBTV.muiltcore.multi_processing(run_auto_filter_stratgy_test, multi_params, core_num=core_num)
            t1 = timeit.default_timer()
            print("\n", '{} 耗时为：'.format(timeframe), t1 - t0)
            # ---记录指标完成
            finish_timeframe.append(timeframe)
            mylogging.warning("symbol finished: {}".format(finish_timeframe))
    # ---
    main_func(core_num)












