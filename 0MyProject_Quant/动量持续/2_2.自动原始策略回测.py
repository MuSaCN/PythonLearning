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
myPjMT5 = MyProject.MT5_MLLearning()  # MT5机器学习项目类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------

'''
# 1.根据前面整理的最佳参数表格文档，读取参数，再做原始的策略测试。
# 2.策略结果保存到“自动参数选择1D\品种\原始策略测试”文件夹下面
'''

#%% 根据 非策略参数 定位文件 ###########################
import warnings
warnings.filterwarnings('ignore')

symbol_list = myPjMT5.get_all_symbol_name().tolist()
# 策略内参数(非策略参数 symbol、timeframe、direct 会自动解析)
para_name = ["k", "holding", "lag_trade"]
# 仅根据夏普选择就可以了.
evaluate = "sharpe"

#%%
finish_symbol = []
for symbol in symbol_list:
    folder_para1D = __mypath__.get_desktop_path() + "\\_动量研究\\自动参数选择1D\\%s"%symbol
    filepath_para1D = folder_para1D + "\\%s_aotu_para_1D.xlsx"%symbol
    filecontent = pd.read_excel(filepath_para1D)
    # ---解析
    for i in range(len(filecontent)):
        # ---获取各参数和策略评价
        timeframe = filecontent.iloc[i]["timeframe"]
        direct = filecontent.iloc[i]["direct"]
        k = filecontent.iloc[i][para_name[0]]
        holding = filecontent.iloc[i][para_name[1]]
        lag_trade = filecontent.iloc[i][para_name[2]]
        eva_train = filecontent.iloc[i][evaluate] # 训练集策略评价
        # 解析参数生成字符串变量，用于 添加策略图的标注 和 输出图片命名。
        para_str = ""
        for name in para_name:
            para_str = para_str + name + "=%s" % filecontent.iloc[i][name] + ";"
        # ---加载测试数据
        date_from, date_to = myPjMT5.get_date_range(timeframe)
        data_total = myPjMT5.getsymboldata(symbol, timeframe, date_from, date_to, index_time=True, col_capitalize=True)
        data_train, data_test = myPjMT5.get_train_test(data_total, train_scale=0.8)
        # 单独测试对全数据进行测试，训练集、测试集仅画区间就可以了
        train_x0 = data_train.index[0]
        train_x1 = data_train.index[-1]
        # ---获取信号数据
        signaldata = myBTV.stra.momentum(data_total.Close, k=k, holding=holding, sig_mode=direct, stra_mode="Continue")
        if direct == "BuyOnly":
            signaldata_input = signaldata["buysignal"]
        elif direct == "SellOnly":
            signaldata_input = signaldata["sellsignal"]
        # ---信号分析，不重复持仓
        outStrat, outSignal = myBTV.signal_quality_NoRepeatHold(signaldata_input, price_DataFrame=data_total, holding=holding, lag_trade=lag_trade, plotStrat=True, train_x0=train_x0, train_x1=train_x1, savefig=None, show=False) # show必须设为False
        # ---在策略图上标注 训练集和全集的策略评价 和 参数字符串para_str
        eva_all = outStrat[direct][evaluate] # 全集策略评价
        ax = plt.gca() # 获取策略图的ax
        y1 = (outStrat[direct]["cumRet"]/2 + 1)
        ax.annotate(s="%s train=%.4f,all=%.4f"%(evaluate, eva_train, eva_all), xy=[train_x0, y1], xytext=[train_x0, y1])
        ax.annotate(s="%s" % para_str, xy=[train_x0, 1], xytext=[train_x0, 1])
        # ---保存输出图片
        savefig = folder_para1D + "\\原始策略回测\\{}.{}({}).png".format(timeframe,direct,para_str)
        import os
        os.makedirs(os.path.dirname(savefig), exist_ok=True)
        fig = ax.get_figure()
        fig.savefig(savefig)
        # 关闭图片，在批量操作时，释放内存
        plt.close(fig)
        plt.show()
        print(symbol,timeframe,direct,para_str,"完成！")
    # ---显示进度
    finish_symbol.append(symbol)
    print("自动原始策略回测 finished:", finish_symbol)



