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

# ------------------------------------------------------------
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
myMT5Pro = MyMql.MyClass_ConnectMT5Pro(connect=False)  # Python链接MT5高级类
myMT5Indi = MyMql.MyClass_MT5Indicator()  # MT5指标Python版
myMT5Report = MyMT5Report.MyClass_StratTestReport(AddFigure=False)  # MT5策略报告类
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=False)  # 固定杠杆仓位类
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=False)  # 浮动杠杆仓位类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------
# myDefault.set_backend_default("agg") # 这句必须放到类下面

#%%
"1_1.参数优化部分："
# (***需修改***)策略说明：
'''
# DailyRange交叉动量策略，运用 "当天日线bar的open" +- "前一日线bar的range*n" 得到上下通道。
# close 向上交叉上轨，触发做空信号；向下交叉下轨，触发做多信号。
# 由于指标为直线类轨道。固该策略排除了下面的情况：金叉的触发是因为指标轨道在日线切换时下跳；死叉的触发是因为指标轨道在日线切换时上跳。本策略排除上下轨在日线切换时跳动触发交叉信号的情况。
# 只考虑入场，出场模式放在其他地方考虑。
# 信号触发且确认后，下一期进行交易。持有仓位周期为1根K线。
'''
# 参数优化说明：
'''
# 参数优化说明：
# 参数优化部分，需要专门设定训练集和测试集。由于参数较多，不可能都通过图示。所以，通过训练集来计算出各个参数下策略结果，安全起见保存结果到硬盘。
# 再根据训练集参数优化的结果，计算对应参数下测试集策略结果，把结果保存到硬盘。
# 整合两个结果到一张表格。
# 需要注意的是，由于 训练集和测试集 信号计算时依赖的数据集不同，所以要设定两个函数。
# 由于并行运算的原理，参数分为 策略参数 + 非策略参数
# 为了提高运算速度，可以只测试训练集，然后再通过后面的分析筛选。
# 由于并行运算时间长，防止出错输出日志。
'''

"1_2.策略参数自动选择："
'''
# 1.根据前面输出的优化结果，自动寻找最佳参数点。由于品种较多，再算上极值点判断方法，耗时较长，故采用多核运算。
# 2.自动寻找的思路为：对 过滤0次、过滤1次、过滤2次 的数据寻找极值点。会输出图片和表格。注意过滤后的数据判断完极值后，会根据其位置索引到源数据，再组成表格的内容。注意图片中的过滤部分极值，并没有更改为源数据，仅表格更改了。
# 3.并行运算必须处理好图片释放内存的问题，且并行逻辑与目录逻辑不一样要一样。此处是以品种作为并行方案。
# 4.根据输出的图片看过滤几次较好，以及判断极值每一边用有多少点进行比较较好。
# 5.为下一步批量自动回测做准备。
'''
"1_3.汇总过滤结果"
''' 汇总过滤结果：
# 由于一个品种 30、40、50 的极值选择会有重复的。所以我们汇总到一起，删除重复的。
# 保存到 ...\_**研究\策略参数自动选择\symbol\symbol.total.filter*.xlsx
# 汇总目的在于为后续分析提供便利。
'''
"2_1, 2_2.订单可管理性分析及回测"
'''
# 订单可管理性：如果一个策略在未来1期持仓表现不错，同时在未来多期持仓也表现不错。这就表明，这个策略的交易订单在时间伸展上能够被管理，我们称作为订单具备可管理性。
# 对训练集进行多holding回测，展示结果的夏普比曲线和胜率曲线。
# 采用无重复持仓模式和重复持仓模式。
# 如果前3个夏普都是递增的，则选择之。输出测试图片。否则不认为具有可管理性，则弃之。
# 并行运算以品种来并行
'''
'''
# 0.这里的回测是建立在前面已经对策略的参数做了选择。
# 1.根据前面整理的自动选择的最佳参数表格文档，读取参数，再做原始的策略测试。
# 2.策略结果保存到 "策略参数自动选择\品种\auto_para_1D_{order}\原始策略回测_filter1" 文件夹下面。
# 3.策略测试所用的区间要增大。
# 4.回测结果较多，构成策略库供后续选择研究。
# 5.并行运算注意内存释放，并且不要一次性都算完，这样容易爆内存。分组进行并行。
'''
"2_3.同策同框同向不同参数比较筛选"
'''
# 1.同一个策略、同一个时间框、同一个方向下，不同的参数之间进行比较筛选。
# 2.筛选最佳的占优策略 或排除最差的策略。思路：模式1：先分析词缀sharpe和cumRet下是否有指定比率(比如80%)领先者，若有则领先者为最佳，否则进入模式2；模式2：先分析某个词缀(比如sharpe)下哪个策略的优势超过指定比率(比如80%)，该策略得1分。对所有词缀进行分析，若某个策略的得分最大且超过指定数量(词缀个数*2*80%)，则该策略认为是最佳的占优策略。
# 3.排除最差策略思想与上述相反。
# 4.反复筛选，直到剩余1个 或 找不到最佳最差 或 找到最佳。
'''
"3_1.指标范围过滤输出文档"
'''
# 说明
# 1.根据信号的利润，运用其他指标来过滤，从累计利润角度进行过滤。可以分析出 其他指标的值 的哪些区间对于累计利润是正的贡献、哪些区间是负的贡献。所用的思想为“求积分(累积和)来进行噪音过滤”。
# 2.根据训练集获取过滤区间，然后作用到训练集，不是整个样本。
# 3.一个策略参数有许多个指标，每个指标有许多指标参数，这些结果都放到一个表格中。
# 4.有许多个指标，所以通过并行运算。并行是对一个品种、一个时间框下、一个方向下，不同指标的不同参数进行并行。
# 5.表格文档存放到硬盘路径"_**研究\过滤指标参数自动选择\symbol.timeframe"，以便于下一步极值分析。
# 6.由于属于大型计算，并行运算时间长，防止出错要输出日志。
# 7.后期要通过动态读取文件来解析品种、时间框、方向、策略参数名、策略参数值等
'''
"3_2.范围过滤参数自动选择及策略回测"
'''
# 1.根据前面 信号利润过滤测试 输出的文档，解析文档名称，读取参数，选择极值。
# 2.一个特定的策略参数作为一个目录，存放该下面所有指标的结果。
# 3.不同名称的指标会自动判断极值，且输出图片。最后会输出表格文档，整理这些极值。
# 4.由于不是大型计算，并行是一次性所有并行。
# 5.并行运算注意内存释放，并且不要一次性都算完，这样容易爆内存。分组进行并行。
'''
'''
# 说明
# 这里的策略回测是建立在前面已经对指标的范围过滤做了参数选择。
# 前面对每个具体策略都通过指标过滤方式，算出了各个指标过滤效果的极值。我们根据极值对应的指标值做回测。
# 画的图中，min-max表示 "max最大的以max之前的min最小" 或 "min最小的以min之后的max最大"，start-end表示上涨额度最大的区间。
# 根据训练集获取过滤区间，然后作用到整个样本。
# 并行以品种来并行，以时间框来分组。
# 由于指标较多，并行运算时间长，防止出错输出日志。
'''
"4_1.指标方向过滤输出文档"
'''
# 说明：
# 1.根据趋势性指标进行策略方向性过滤。价格在指标上方，只做多、不做空；价格在指标下方，只做空，不做多。
# 2.根据训练集获取过滤区间，然后作用到训练集，不是整个样本。
# 3.一个策略参数有许多个指标，每个指标有许多指标参数，这些结果都放到一个表格中。
# 4.有许多个指标，所以通过并行运算。并行是对一个品种、一个时间框下、一个方向下，不同指标的不同参数进行并行。
# 5.表格文档存放到硬盘路径"_**研究\过滤指标参数自动选择\symbol.timeframe"，以便于下一步极值分析。
# 6.由于属于大型计算，并行运算时间长，防止出错要输出日志。
# 7.后期要通过动态读取文件来解析品种、时间框、方向、策略参数名、策略参数值等
'''
"4_2.方向过滤参数自动选择及策略回测"
'''
# 1.根据前面 信号利润过滤测试 输出的文档，解析文档名称，读取参数，选择极值。
# 2.一个特定的策略参数作为一个目录，存放该下面所有指标的结果。
# 3.不同名称的指标会自动判断极值，且输出图片。最后会输出表格文档，整理这些极值。
# 4.由于不是大型计算，并行是一次性所有并行。
# 5.并行运算注意内存释放，并且不要一次性都算完，这样容易爆内存。分组进行并行。
'''
'''
# 说明
# 这里的策略回测是建立在前面已经对指标的范围过滤做了参数选择。
# 前面对每个具体策略都通过指标过滤方式，算出了各个指标过滤效果的极值。我们根据极值对应的指标值做回测。
# 画的图中，分别展示 过滤前训练集价格和指标、过滤前训练集策略、过滤后全集价格和指标、过滤后全集策略以及训练集策略。
# 方向过滤作用到整个样本。
# 并行以品种来并行，以时间框来分组。
# 由于指标较多，并行运算时间长，防止出错输出日志。
'''
"5.策略池整合"
'''
# 说明：
# 我们的思想是，不同组的策略参数可以看成不同的策略进行叠加。但是过滤的指标参数只能选择一个。
# 这一步把这些结果整合到一起，形成策略池。
# 前面已经对一个品种、一个时间框、一个方向、一组参数进行了指标范围过滤和指标方向过滤。
# 某个品种某个时间框某个参数组有许多个过滤情况，我们可以通过“策略参数自动选择”输出的极值图片来排除哪些策略参数组不好。
# 过滤后的结果选择 filter1 中的 sharpe_filter 最大值，即选择思想为过滤后的最大值。
# 由于前面对某些品种可能设置了条件，整合时注意要先判断对应的参数目录是否存在。
# 复制图片，必须夏普比率有所提高才复制。
# 并行运算以品种为并行参数。
'''

#%% ##################### (***需修改***)通用设置 #####################
""
# ---非策略设置
core_num = -1
total_folder = "F:\\工作---Python策略研究\\4.DailyRange交叉策略\\_交叉反转研究"
filename_prefix = "DailyRange交叉反转"
symbol_list = myMT5Pro.get_mainusd_symbol_name_list()
direct_para = ["BuyOnly", "SellOnly"] # 方向词缀 ["BuyOnly", "SellOnly", "All"]

# ---策略类设置
# 策略信号：策略的当期信号(不用平移)：para_list策略参数，默认-1为lag_trade，-2为holding。
def stratgy_signal(dataframe, para_list=list or tuple):
    return myBTV.stra.dailyrange_cross_reverse(dataframe, n=para_list[0])
# 策略参数名称，顺序不能搞错了，要与信号函数中一致
strategy_para_names = ["n", "holding", "lag_trade"]
# 设置固定和浮动的策略参数，key词缀不能搞错了
para_fixed_list = [{"n":None, "holding":1, "lag_trade":1}]


#%% ##################### (***需修改***)1_1.策略参数优化 ###########################
from MyPackage.MyProjects.向量化策略测试.Strategy_Param_Opt import Strategy_Param_Opt_OutPut
opt = Strategy_Param_Opt_OutPut()

# ************ 需要修改的部分 ************
# (***需修改***)策略参数，设置范围的最大值，按顺序保存在 para 的前面
opt.strategy_para_names = strategy_para_names  # 顺序不能搞错了，要与信号函数中一致
opt.para1_end = 2.0         # 通道的倍数参数，步长0.01 和 范围0.1~2.0
opt.holding_end = 1         # 持有期参数，可以不同固定为1
opt.lag_trade_end = 1       # 信号出现滞后交易参数，参数不能大

# (***需修改***)获取策略参数范围(direct、timeframe、symbol参数必须设置在-3、-2、-1的位置)
def get_strat_para_scope(direct, timeframe, symbol):
    return [(n, holding, lag_trade, direct, timeframe, symbol)
            for n in [i / 100.0 for i in range(1, int(opt.para1_end) * 100 + 1, 1)]
            for holding in range(1, opt.holding_end + 1)
            for lag_trade in range(1, opt.lag_trade_end + 1)]
opt.get_strat_para_scope = get_strat_para_scope

# (***需修改***)策略退出条件，strat_para = (n, holding, lag_trade)。
def strat_break(strat_para):
    pass # 这里可以没有 # if strat_para[1] > strat_para[0]: return True
opt.strat_break = strat_break

# (***需修改***)单个策略过滤方式
def filter_strategy(*args): # 由于结果太少，不做过滤。
    return myBTV.no_filter_strategy(*args) # no_filter_strategy filter_strategy
opt.filter_strategy = filter_strategy


# 非策略参数
opt.direct_para = direct_para # direct_para = ["BuyOnly", "SellOnly", "All"]
opt.symbol_list = symbol_list
opt.total_folder = total_folder
opt.filename_prefix = filename_prefix

# 策略信号
opt.stratgy_signal = stratgy_signal

# 优化时用的核心数
opt.core_num = core_num # 要显示进度不能是-1，需具体指定

#%% ##################### 1_2.策略参数自动选择 #####################
from MyPackage.MyProjects.向量化策略测试.Strategy_Param_Opt import Auto_Choose_StratOptParam
choose_opt = Auto_Choose_StratOptParam()

choose_opt.total_folder = total_folder
choose_opt.filename_prefix = filename_prefix
choose_opt.symbol_list = symbol_list
choose_opt.para_fixed_list = para_fixed_list # key词缀不能搞错了
choose_opt.direct_para = direct_para

choose_opt.y_name = ["sharpe"] # 过滤的y轴，不能太多。仅根据夏普选择就可以了.
choose_opt.core_num = core_num # -1表示留1个进程不执行运算。


#%% ##################### 1_3.汇总品种不同过滤结果 #####################
from MyPackage.MyProjects.向量化策略测试.Strategy_Param_Opt import Sum_Auto_Choose
sum_choo = Sum_Auto_Choose()

sum_choo.strategy_para_names = strategy_para_names # list(choose_opt.para_fixed_list[0].keys())
sum_choo.all_folder = total_folder
sum_choo.symbol_list = symbol_list
sum_choo.outfile_suffix = ".original" # 输出的文档加后缀
sum_choo.core_num = core_num


#%% ##################### 2_1.订单可管理性分析 #####################
from MyPackage.MyProjects.向量化策略测试.More_Holding import Auto_More_Holding
more_h = Auto_More_Holding()

more_h.strategy_para_names = strategy_para_names
more_h.symbol_list = symbol_list
more_h.total_folder = total_folder
more_h.readfile_suffix = sum_choo.outfile_suffix # 输入的文档加后缀
more_h.outfile_suffix = ".holdingtest" # 输出的文档加后缀
more_h.core_num = core_num
more_h.holding_testcount = 3  # 测试到的holding数量
more_h.stratgy_signal = stratgy_signal # 策略信号


#%% ##################### 2_2.订单可管理性分析后进行回测 #####################
from MyPackage.MyProjects.向量化策略测试.More_Holding import Strategy_BackTest
strat_bt = Strategy_BackTest()

# 策略内参数(非策略参数 symbol、timeframe、direct 会自动解析)
strat_bt.strategy_para_names = strategy_para_names
strat_bt.symbol_list = symbol_list
strat_bt.total_folder = total_folder
strat_bt.readfile_suffix = more_h.outfile_suffix # ".holdingtest" # 输入的文档加后缀
strat_bt.core_num = core_num # -1表示留1个进程不执行运算。
strat_bt.stratgy_signal = stratgy_signal # 策略信号


#%% ##################### 2_3.同策同框同向不同参数比较筛选 #####################
from MyPackage.MyProjects.向量化策略测试.More_Holding import Strategy_Better
s_better = Strategy_Better()

s_better.strategy_para_names = strategy_para_names
s_better.symbol_list = symbol_list
s_better.total_folder = total_folder
s_better.readfile_suffix = more_h.outfile_suffix # 输入的文档加后缀 .holdingtest
s_better.outfile_suffix = ".better" # 输出的文档加后缀
s_better.core_num = core_num
s_better.stratgy_signal = stratgy_signal


#%% ##################### 3_1.指标范围过滤输出文档 #####################
from MyPackage.MyProjects.向量化策略测试.Range_Filter import Range_Filter_Output
rf_out = Range_Filter_Output()

rf_out.strategy_para_names = strategy_para_names
rf_out.symbol_list = symbol_list
rf_out.total_folder = total_folder
rf_out.readfile_suffix = s_better.outfile_suffix
rf_out.stratgy_signal = stratgy_signal
rf_out.core_num = core_num


#%% ##################### 3_2.范围过滤参数自动选择及策略回测 #####################
from MyPackage.MyProjects.向量化策略测试.Range_Filter import Auto_Choose_RFilter_Param
rf_choo_para = Auto_Choose_RFilter_Param()

rf_choo_para.symbol_list = symbol_list
rf_choo_para.total_folder = total_folder
rf_choo_para.core_num = core_num


from MyPackage.MyProjects.向量化策略测试.Range_Filter import Range_Filter_BackTest
rf_bt = Range_Filter_BackTest()

rf_bt.symbol_list = symbol_list
rf_bt.total_folder = total_folder
rf_bt.core_num = core_num  # 注意，M1, M2时间框数据量较大时，并行太多会爆内存。
rf_bt.stratgy_signal = stratgy_signal


#%% ##################### 4_1.指标方向过滤输出文档 #####################
from MyPackage.MyProjects.向量化策略测试.Direct_Filter import Direct_Filter_Output
df_out = Direct_Filter_Output()

# 策略参数名称，用于文档中解析参数
df_out.strategy_para_names = strategy_para_names
df_out.symbol_list = symbol_list
df_out.total_folder = total_folder
df_out.readfile_suffix = s_better.outfile_suffix
df_out.stratgy_signal = stratgy_signal
df_out.core_num = core_num


#%% ##################### 4_2.方向过滤参数自动选择及策略回测 #####################
from MyPackage.MyProjects.向量化策略测试.Direct_Filter import Auto_Choose_DFilter_Param
df_choo_para = Auto_Choose_DFilter_Param()

df_choo_para.symbol_list = symbol_list
df_choo_para.total_folder = total_folder
df_choo_para.core_num = core_num


from MyPackage.MyProjects.向量化策略测试.Direct_Filter import Direct_Filter_BackTest
df_bt = Direct_Filter_BackTest()

df_bt.symbol_list = symbol_list
df_bt.total_folder = total_folder
df_bt.core_num = core_num
df_bt.stratgy_signal = stratgy_signal



#%% ##################### 5.策略池整合 #####################
from MyPackage.MyProjects.向量化策略测试.Strategy_Param_Opt import Strat_Pool_Integration
strat_pool = Strat_Pool_Integration()

strat_pool.strategy_para_names = strategy_para_names
strat_pool.symbol_list = symbol_list
strat_pool.total_folder = total_folder
strat_pool.readfile_suffix = s_better.outfile_suffix
strat_pool.core_num = core_num

#%% 设置图片输出方式
myDefault.set_backend_default("agg") # 这句必须放到类下面
#%%
# ---多进程必须要在这里执行
if __name__ == '__main__':

    # # ---1_1.参数优化。写入log
    # print("1_1. 开始策略参数优化_并行")
    # logger = mylogging.getLogger(__mypath__.get_desktop_path() + "\\1_1.参数优化.log")
    # opt.main_func(run_testset=False, logger=logger)
    # # ---1_2.参数自动选择
    # print("1_2. 开始策略参数自动选择_并行")
    # choose_opt.main_func()
    # # ---1_3.汇总品种不同过滤结果
    # print("1_3. 开始汇总品种不同过滤结果_并行")
    # sum_choo.main_func()
    #
    # # ---2_1.开始订单可管理性分析。包括 2_1_1新的文档文件、2_1_2分析图片写入log
    # print("2_1. 开始订单可管理性分析： ")
    # logger = mylogging.getLogger(__mypath__.get_desktop_path() + "\\2_1.订单可管理性分析.log")
    # more_h.main_func(logger=logger) # 注意，多核不完全输出，此处用的是折中办法。
    # # ---2_2.订单可管理性分析后进行回测
    # print("2_2. 开始筛选后策略自动回测： ")
    # strat_bt.main_func()
    # # ---2_3.同策同框同向不同参数比较筛选
    # print("2_3. 开始同策同框同向不同参数比较筛选： ")
    # s_better.main_func()
    #
    # # ---3_1.指标范围过滤输出文档
    # print("3_1. 开始指标范围过滤输出文档")
    # logger = mylogging.getLogger(__mypath__.get_desktop_path() + "\\3_1.指标范围过滤输出文档.log")
    # rf_out.main_func(logger=logger)
    # # ---3_2.范围过滤参数自动选择及策略回测
    # # 3_2_1.范围过滤参数自动选择
    # print("3_2_1. 开始范围过滤参数自动选择：")
    # rf_choo_para.main_func()
    # # 3_2_2. 范围过滤策略回测
    # print("3_2_2. 开始范围过滤策略回测：")
    # logger = mylogging.getLogger(__mypath__.get_desktop_path() + "\\3_2.范围过滤策略回测.log")
    # rf_bt.main_func(logger=logger)

    # ---4_1.指标方向过滤输出文档
    # print("4_1. 开始指标方向过滤输出文档")
    # logger = mylogging.getLogger(__mypath__.get_desktop_path() + "\\4_1.指标方向过滤输出文档.log")
    # df_out.main_func(logger=logger)
    # # ---4_2.方向过滤参数自动选择及策略回测
    # # 4_2_1 方向过滤参数自动选择
    # print("4_2_1. 开始方向过滤参数自动选择：")
    # df_choo_para.main_func()
    # # 4_2_2 方向过滤策略回测
    # print("4_2_2. 开始方向过滤策略回测：")
    # logger = mylogging.getLogger(__mypath__.get_desktop_path() + "\\4_2.方向过滤策略回测.log")
    # df_bt.main_func(logger=logger)

    # ---5.策略池整合
    print("5. 策略池整合")
    strat_pool.main_func()

















