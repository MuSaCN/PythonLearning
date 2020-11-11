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
mylogging = MyDefault.MyClass_Default_Logging(activate=False)  # 日志记录类，需要放在上面才行
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
# 由于一个品种 30、40、50 的极值选择会有重复的。所以我们汇总到一起，删除重复的。
# 保存到 ...\_**研究\策略参数自动选择\symbol\symbol.total.filter*.xlsx
# 汇总目的在于为后续分析提供便利。
'''

# %%
strat_para_name = ["k", "holding", "lag_trade"]
order_list = [30, 40, 50]
flevel_list = ["filter0", "filter1", "filter2"]


# %%
def run_flevel_concat(para):
    symbol = para[0]
    # 各过滤等级分别输出文档
    for flevel in flevel_list:
        total_df = pd.DataFrame()
        # ---目录定位 ******修改这里******
        total_folder = __mypath__.get_desktop_path() + "\\_反转研究\\策略参数自动选择\\%s" % symbol
        for order in order_list:
            in_folder = total_folder + "\\auto_para_1D_{}".format(order)
            in_file = in_folder + "\\" + "{}.{}.xlsx".format(symbol, flevel)
            filecontent = pd.read_excel(in_file, index_col="Unnamed: 0")
            total_df = pd.concat((total_df, filecontent), ignore_index=True)
        # ---
        total_df = total_df.sort_values(by=["symbol", "timeframe", "direct"] + strat_para_name, ignore_index=True)
        total_df = total_df.drop_duplicates(ignore_index=True)
        total_df.to_excel(total_folder + "\\{}.total.{}.xlsx".format(symbol, flevel))
    print(symbol, "文档合并完成！")


# %%
core_num = -1
if __name__ == '__main__':
    symbol_list = myMT5Pro.get_all_symbol_name().tolist()
    para_muilt = [(symbol,) for symbol in symbol_list]
    import timeit
    # ---开始多核执行
    t0 = timeit.default_timer()
    myBTV.muiltcore.multi_processing(run_flevel_concat, para_muilt, core_num=core_num)
    t1 = timeit.default_timer()
    print("\n", 'run_level_concat 耗时为：', t1 - t0)

