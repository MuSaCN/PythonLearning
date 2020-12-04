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
# 1.根据前面 信号利润过滤测试 输出的文档，解析文档名称，读取参数，选择极值。
# 2.一个特定的策略参数作为一个目录，存放该下面所有指标的结果。
# 3.不同名称的指标会自动判断极值，且输出图片。最后会输出表格文档，整理这些极值。
# 4.由于不是大型计算，并行是一次性所有并行。
# 5.并行运算注意内存释放，并且不要一次性都算完，这样容易爆内存。分组进行并行。
'''
myDefault.set_backend_default("agg")

#%% 根据 非策略参数 定位文件 ###########################
symbol_list = myMT5Pro.get_main_symbol_name_list()
y_name = ["sharpe"] # 过滤的y轴，不能太多。仅根据夏普选择就可以了.
# 指标名称
indi_name_list = myBTV.indiMT5.indi_name_directfilter()
# 指标参数固定和1D浮动设定，返回字典。
indi_para_fixed_list = myBTV.indiMT5.indi_params_setting1D(indi_name_list)
total_folder = "F:\\工作---策略研究\\简单的动量反转\\_动量研究"


#%%

# 指标参数自动判定
def run_auto_indi_direct_opt(para):
    print("\r", "当前执行参数为：", para, end="", flush=True)
    # para = ("EURUSD", "TIMEFRAME_D1")
    symbol = para[0]
    timeframe = para[1]
    order = 30 # 由于指标参数结果较为稳定，选择30就可以了。
    # 目录定位
    in_folder = total_folder + "\\方向指标参数自动选择\\{}.{}".format(symbol,timeframe)
    # 判断是否存在，不存在则返回
    if __mypath__.path_exists(in_folder) == False:
        return
    # ---以特定参数的策略 作为研究对象
    file_dir = __mypath__.listdir(in_folder)
    for filename in file_dir: # filename = file_dir[0]
        # 如果不是 xlsx格式文件则跳过
        if ".xlsx" not in filename:
            continue
        # 解析下文件名称
        direct, suffix = filename.split(".")[0:2]
        # 输入路径要重新设置下
        in_file = in_folder + "\\" + filename
        filecontent = pd.read_excel(in_file)
        # 批量运算，最后合并且输出表格，存放 所有指标 的过滤结果，过滤有3个等级。
        total_df0 = pd.DataFrame([])
        total_df1 = pd.DataFrame([])
        total_df2 = pd.DataFrame([])
        # ---分别处理不同指标
        for indi_name in indi_name_list: # indi_name = indi_name_list[0]
            # 加载指标固定浮动参数
            indi_para_fixed = indi_para_fixed_list[indi_name]
            # 过滤0，输出图片
            out_df0 = myBTV.dfilter.auto_indi_para_direct_filter_1D(filepath=in_file, filecontent=filecontent, indi_name=indi_name, indi_para_fixed=indi_para_fixed, y_name=y_name, order=order, filterlevel=0, plot=True, savefolder="default", batch=True)
            total_df0 = pd.concat([total_df0, out_df0], axis=0, ignore_index=True)
            # 过滤1，不输出图片
            out_df1 = myBTV.dfilter.auto_indi_para_direct_filter_1D(filepath=in_file, filecontent=filecontent, indi_name=indi_name, indi_para_fixed=indi_para_fixed, y_name=y_name, order=order, filterlevel=1, plot=False, savefolder="default", batch=True)
            total_df1 = pd.concat([total_df1, out_df1], axis=0, ignore_index=True)
            # 过滤2，不输出图片
            out_df2 = myBTV.dfilter.auto_indi_para_direct_filter_1D(filepath=in_file, filecontent=filecontent, indi_name=indi_name,indi_para_fixed=indi_para_fixed, y_name=y_name, order=order, filterlevel=2, plot=False, savefolder="default", batch=True)
            total_df2 = pd.concat([total_df2, out_df2], axis=0, ignore_index=True)
        # ---输出表格文档文件0、1、2，存放 所有指标 的过滤结果
        # 输出文件夹
        out_folder = in_folder + "\\{}.{}".format(direct, suffix)
        total_df0.to_excel(out_folder + "\\{}.filter0.xlsx".format(suffix)) # 输出文件0
        total_df1.to_excel(out_folder + "\\{}.filter1.xlsx".format(suffix)) # 输出文件1
        total_df2.to_excel(out_folder + "\\{}.filter2.xlsx".format(suffix)) # 输出文件2
        # 显示进度
        print(symbol,timeframe,direct,suffix,"finished!")
    # 总进度
    print("指标参数自动选择 finished:", symbol, timeframe,)


#%%
core_num = -1
# ---多进程必须要在这里执行
if __name__ == '__main__':

    def main_func():
        timeframe_list = ["TIMEFRAME_D1", "TIMEFRAME_H12", "TIMEFRAME_H8", "TIMEFRAME_H6",
                          "TIMEFRAME_H4", "TIMEFRAME_H3", "TIMEFRAME_H2", "TIMEFRAME_H1",
                          "TIMEFRAME_M30", "TIMEFRAME_M20", "TIMEFRAME_M15", "TIMEFRAME_M12",
                          "TIMEFRAME_M10", "TIMEFRAME_M6", "TIMEFRAME_M5", "TIMEFRAME_M4",
                          "TIMEFRAME_M3", "TIMEFRAME_M2", "TIMEFRAME_M1"]
        para_muilt = [(symbol,timeframe) for symbol in symbol_list for timeframe in timeframe_list]
        import timeit
        # ---开始多核执行，内容较少，不用分组。
        t0 = timeit.default_timer()
        myBTV.muiltcore.multi_processing(run_auto_indi_direct_opt, para_muilt, core_num=core_num)
        t1 = timeit.default_timer()
        print("\n", 'run_auto_indi_opt 耗时为：', t1 - t0)
    # ---
    main_func()









