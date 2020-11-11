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


#%% 根据 非策略参数 定位文件 ###########################
direct_para = ["BuyOnly", "SellOnly"]  # direct_para = ["BuyOnly", "SellOnly", "All"]
timeframe_list = ["TIMEFRAME_D1","TIMEFRAME_H12","TIMEFRAME_H8","TIMEFRAME_H6",
                  "TIMEFRAME_H4","TIMEFRAME_H3","TIMEFRAME_H2","TIMEFRAME_H1",
                  "TIMEFRAME_M30","TIMEFRAME_M20","TIMEFRAME_M15","TIMEFRAME_M12",
                  "TIMEFRAME_M10","TIMEFRAME_M6","TIMEFRAME_M5","TIMEFRAME_M4",
                  "TIMEFRAME_M3","TIMEFRAME_M2","TIMEFRAME_M1"]


#%%
myDefault.set_backend_default("agg")
# 仅检测 holding=1 就可以了
para_fixed_list = [{"k":None, "holding":1, "lag_trade":1}]
# 仅根据夏普选择就可以了.
y_name = ["sharpe"] # 过滤的y轴，不能太多


#%%
# ---并行算法参数：0---order极值每一边用有多少点进行比较 ；1---symbol品种；
def run_auto_strat_opt(para):
    symbol = para[0]
    order = para[1]
    # 批量运算，最后合并且输出表格
    total_df0 = pd.DataFrame([])
    total_df1 = pd.DataFrame([])
    total_df2 = pd.DataFrame([])
    # 把所有的timeframe和direct都整理到一个文档中
    for timeframe in timeframe_list:
        # ---输入目录和输出目录 ***修改这里***
        in_folder = __mypath__.get_desktop_path() + "\\_动量研究\\{}.{}".format(symbol, timeframe)
        out_folder = __mypath__.dirname(in_folder,uplevel=0) + "\\策略参数自动选择\\{}\\auto_para_1D_{}".format(symbol, order)
        # ---
        for direct in direct_para:
            # ---路径 ***修改这里***
            filepath = in_folder + "\\动量_{}.xlsx".format(direct)  # 选择训练集文件
            filecontent = pd.read_excel(filepath)
            for para_fixed in para_fixed_list:
                # 过滤0，输出图片
                out_df0 = myBTV.auto_strat_para_1D(filepath=filepath, filecontent=filecontent, para_fixed=para_fixed, y_name=y_name, order=order, filterlevel=0, plot=True, savefolder=out_folder, batch=True)
                total_df0 = pd.concat([total_df0,out_df0 ],axis=0, ignore_index=True)
                # 过滤1，不输出图片
                out_df1 = myBTV.auto_strat_para_1D(filepath=filepath, filecontent=filecontent, para_fixed=para_fixed, y_name=y_name, order=order, filterlevel=1, plot=False)
                total_df1 = pd.concat([total_df1, out_df1], axis=0, ignore_index=True)
                # 过滤2，不输出图片
                out_df2 = myBTV.auto_strat_para_1D(filepath=filepath, filecontent=filecontent, para_fixed=para_fixed, y_name=y_name, order=order, filterlevel=2, plot=False)
                total_df2 = pd.concat([total_df2, out_df2], axis=0, ignore_index=True)
        print("\r", symbol, timeframe, "OK", end="", flush=True)
    # 输出表格
    total_df0.to_excel(out_folder + "\\%s.filter0.xlsx" % symbol)
    total_df1.to_excel(out_folder + "\\%s.filter1.xlsx" % symbol)
    total_df2.to_excel(out_folder + "\\%s.filter2.xlsx" % symbol)
    # 显示进度
    print(symbol, "auto_para_1D_%s finished:"%order)


#%%
################# 多进程执行函数 ########################################
cpu_core = -1 # -1表示留1个进程不执行运算。
# ---多进程必须要在这里执行
if __name__ == '__main__':
    symbol_list = myMT5Pro.get_all_symbol_name().tolist()
    order_list = [30, 40, 50]  # [30,40,50]
    # ---多步并行，以更好的控制进度
    for order in order_list:
        para_muilt = [(symbol, order) for symbol in symbol_list]
        import timeit
        # ---开始多核执行
        t0 = timeit.default_timer()
        myBTV.muiltcore.multi_processing(run_auto_strat_opt, para_muilt, core_num=cpu_core)
        t1 = timeit.default_timer()
        print("\n", 'para_muilt_%s 耗时为：'%order, t1 - t0)










