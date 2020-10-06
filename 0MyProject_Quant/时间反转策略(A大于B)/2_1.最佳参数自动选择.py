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
# 1.根据前面输出的优化结果，自动寻找最佳参数点。
# 2.运算后的输出内容都放在主目录下的“自动参数选择1D”文件夹，然后在里面分别建立品种目录存档结果。
# 3.人工浏览下自动寻找的最佳参数点，排除一些潜在不合理的地方。
# 4.修改表格内容，为下一步批量自动回测做准备。
'''


#%% 根据 非策略参数 定位文件 ###########################
import warnings
warnings.filterwarnings('ignore')

direct_para = ["BuyOnly", "SellOnly"]  # direct_para = ["BuyOnly", "SellOnly", "All"]
symbol_list = myPjMT5.get_all_symbol_name().tolist()
timeframe_list = ["TIMEFRAME_D1","TIMEFRAME_H12","TIMEFRAME_H8","TIMEFRAME_H6",
                  "TIMEFRAME_H4","TIMEFRAME_H3","TIMEFRAME_H2","TIMEFRAME_H1",
                  "TIMEFRAME_M30","TIMEFRAME_M20","TIMEFRAME_M15","TIMEFRAME_M12",
                  "TIMEFRAME_M10","TIMEFRAME_M6","TIMEFRAME_M5","TIMEFRAME_M4",
                  "TIMEFRAME_M3","TIMEFRAME_M2","TIMEFRAME_M1"]


#%%
myDefault.set_backend_default("agg")
# 仅检测 holding=1 就可以了
para_fixed_list = [{"k":None, "holding":1, "lag_trade":1}]
# 仅根据夏普选择就可以了. ["sharpe", "calmar_ratio", "cumRet", "maxDD"]
y_name = ["sharpe"]

#%%
order = 30 # 极值每一边用有多少点进行比较。
finish_symbol = []
for symbol in symbol_list:
    # 批量运算，最后合并且输出表格
    total_df = pd.DataFrame([])
    for timeframe in timeframe_list:
        # ---输入目录和输出目录 ***(修改这句)***
        in_folder = __mypath__.get_desktop_path() + "\\_反转研究\\{}.{}".format(symbol, timeframe)
        out_folder = __mypath__.dirname(in_folder) + "\\自动参数选择1D_%s\\" % order + symbol
        # ---
        for direct in direct_para:
            # ---文件位置 ***(修改这句)***
            filepath = in_folder + "\\反转_{}.xlsx".format(direct)  # 选择训练集文件
            filecontent = pd.read_excel(filepath)
            for para_fixed in para_fixed_list:
                out_df = myBTV.auto_para_1D(filepath=filepath, filecontent=filecontent, para_fixed=para_fixed, y_name=y_name, order=order, filterlevel=0, plot=True, savefolder=out_folder, batch=True)
                total_df = pd.concat([total_df,out_df ],axis=0, ignore_index=True)
        print(symbol, timeframe, "OK")
    # 输出表格
    total_df.to_excel(out_folder + "\\%s_aotu_para_1D.xlsx"%symbol)
    # 显示进度
    finish_symbol.append(symbol)
    print("自动选择最佳参数1D finished:", finish_symbol)







