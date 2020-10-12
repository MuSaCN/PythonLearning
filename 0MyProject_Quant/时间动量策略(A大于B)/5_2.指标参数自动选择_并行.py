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
myPjMT5 = MyProject.MT5_MLLearning()  # MT5机器学习项目类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
#------------------------------------------------------------


'''
# 1.根据前面 信号利润过滤测试 输出的文档，读取参数，选择极值，再做原始的策略测试，选择合适的指标参数。
# 2.策略结果保存到“...\指标过滤\品种.时间框\指标名称\自动指标参数1D_*\”文件夹下面。
# 3.策略测试所用的区间要增大。
# 4.回测结果较多，构成策略库供后续选择研究。
# 5.并行运算注意内存释放，并且不要一次性都算完，这样容易爆内存。分组进行并行。
# 6.并行是针对一个品种、一个时间框、一个方向下，不同指标进行并行
'''

myDefault.set_backend_default("agg")

#%% 根据 非策略参数 定位文件 ###########################
strategy_para_name = ["k", "holding", "lag_trade"]
strategy_para_direct = [[101,1,1], [101,1,1]] # 索引对应 BuyOnly、SellOnly

indi_name_list=["rsi"] # 参数设置在 para 的 -4 的位置
direct_para = ["BuyOnly","SellOnly"] # 保存在 para 的 -3 位置
timeframe_list = ["TIMEFRAME_D1"] # 保存在 para 的 -2 位置
symbol_list = ["EURUSD"] # 保存在 para 的 -1 位置
y_name = ["sharpe"] # 过滤的y轴，不能太多。仅根据夏普选择就可以了.
indi_para_fixed_list = [{"indi_para0":"Close", "indi_para1":None}]  # 指标参数固定和浮动设定

#%%
order = 30
symbol = symbol_list[0]
timeframe = timeframe_list[0]
direct = direct_para[0]
indi_name = indi_name_list[0]
y = y_name[0]
indi_para_fixed = indi_para_fixed_list[0]


# 生成策略参数字符串，用于定位文档
suffix = myBTV.string_strat_para(strategy_para_name, strategy_para_direct[direct_para.index(direct)])
# 输入路径
in_folder = __mypath__.get_desktop_path()+"\\_动量研究\\指标过滤\\{}.{}\\{}".format(symbol,timeframe,indi_name)
# 输入文件
in_file = in_folder + "\\{}{}.xlsx".format(direct,suffix)
# 输出路径
out_folder = in_folder + "\\自动指标参数选择1D_%s" % order
# 输出文件0、1、2
out_file0 = out_folder + "\\{}_auto_{}_1D_filter0.xlsx.xlsx".format(direct,indi_name) # 输出文件0
out_file1 = out_folder + "\\{}_auto_{}_1D_filter1.xlsx.xlsx".format(direct,indi_name) # 输出文件1
out_file2 = out_folder + "\\{}_auto_{}_1D_filter2.xlsx.xlsx".format(direct,indi_name) # 输出文件2


#%%
# 批量运算，最后合并且输出表格
total_df0 = pd.DataFrame([])
total_df1 = pd.DataFrame([])
total_df2 = pd.DataFrame([])


filecontent = pd.read_excel(in_file)

#%%
filterlevel = 1
out_df0 = myBTV.auto_indi_para_1D(filepath=in_file,filecontent=filecontent,indi_name=indi_name,indi_para_fixed=indi_para_fixed,y_name=y_name,order=order,filterlevel=filterlevel,plot=True,savefolder="default",batch=True)
total_df0 = pd.concat([total_df0,out_df0 ],axis=0, ignore_index=True)

##################################################################



#%%

# 过滤0，输出图片
out_df0 = myBTV.auto_para_1D(filepath=in_file, filecontent=filecontent, para_fixed=indi_para_fixed, y_name=y_name, order=order, filterlevel=0, plot=True, savefolder=out_folder, batch=False)
total_df0 = pd.concat([total_df0,out_df0 ],axis=0, ignore_index=True)

# 过滤1，不输出图片
out_df1 = myBTV.auto_para_1D(filepath=filepath, filecontent=filecontent, para_fixed=para_fixed, y_name=y_name, order=order, filterlevel=1, plot=False)
total_df1 = pd.concat([total_df1, out_df1], axis=0, ignore_index=True)
# 过滤2，不输出图片
out_df2 = myBTV.auto_para_1D(filepath=filepath, filecontent=filecontent, para_fixed=para_fixed, y_name=y_name, order=order, filterlevel=2, plot=False)
total_df2 = pd.concat([total_df2, out_df2], axis=0, ignore_index=True)
print("\r", symbol, timeframe, "OK", end="", flush=True)
# 输出表格
total_df0.to_excel(out_folder + "\\%s_aotu_para_1D_filter0.xlsx" % symbol)
total_df1.to_excel(out_folder + "\\%s_aotu_para_1D_filter1.xlsx" % symbol)
total_df2.to_excel(out_folder + "\\%s_aotu_para_1D_filter2.xlsx" % symbol)
# 显示进度
print("自动选择最佳参数1D_%s finished:"%order, symbol)


#%%
################# 多进程执行函数 ########################################
cpu_core = -1 # -1表示留1个进程不执行运算。
# ---多进程必须要在这里执行
if __name__ == '__main__':
    order_list = [30, 40, 50]  # [30,40,50]
    pass







