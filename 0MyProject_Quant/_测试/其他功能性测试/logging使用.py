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
mylogging = MyDefault.MyClass_Default_Logging(activate=False) # 日志记录类，需要放在上面才行
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

#%%
# 单log写入
mylogging = MyDefault.MyClass_Default_Logging(activate=True, filename=__mypath__.get_desktop_path()+"\\record.log") # 日志记录类，需要放在上面才行

mylogging.debug("This is a debug log.")
mylogging.info("This is a info log.")
mylogging.warning("This is a warning log.")
mylogging.error("This is a error log.")
mylogging.critical("This is a critical log.")

#%% 多log写入，若单log也启动，则单log会写入所有。
filename1 = __mypath__.get_desktop_path()+"\\record1.log"
filename2 = __mypath__.get_desktop_path()+"\\record2.log"

log1 = mylogging.getLogger(filename1)
log2 = mylogging.getLogger(filename2)

mylogging.warning("abc.",log1)
mylogging.error("def.",log1)
mylogging.critical("ghi.",log1)
mylogging.warning("ABC.",log2)
mylogging.error("DEF.",log2)
mylogging.critical("GHI.",log2)


#%% 结论：多核只能输出到单log；多log无法兼容多核，多log时折中办法是在里面重建。
# 并行多log写入测试，默认情况下Python中的logging无法在多进程环境下打印日志。
def multi_process_logging(para):
    name = para[0]
    logger = para[-1]
    print(0,logger) # 有对象
    print(0,logger.name) # 有name
    print(0,logger.handlers) # 传递的logger的handlers = []，所以无法写入。
    # 下面重建内存虽可以写入，但是并行时不完全。
    mylogging.handlers_clear(logger)
    print(1,logger)
    print(1,logger.name)
    print(1,logger.handlers)
    logger = mylogging.getLogger(logger.name)
    print(2,logger.handlers)
    print(2,logger.name)
    mylogging.warning(name, logger=logger)



#%%
# 多进程必须要在这里写
if __name__ == '__main__':
    # ---
    filename3 = __mypath__.get_desktop_path() + "\\record3.log"
    log3 = mylogging.getLogger(filename3)
    name_list = ["A","B","C","D","E"]
    para = [(i,log3) for i in name_list]



    # ---这个可以写入
    for onepara in para:
        multi_process_logging(onepara)
    # log3.handlers.clear()
    # mylogging.removeHandler(log3)
    # log3 = mylogging.getLogger(log3.name)

    # ---多进行无法写入
    out = myBTV.muiltcore.multi_processing(multi_process_logging , para, 5)
    print(out, "finished")
