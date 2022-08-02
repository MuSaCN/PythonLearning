# Author:Zhang Yuan
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
myparallel = MyTools.MyClass_ParallelCal() # 并行运算类
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
myMoneyM = MyTrade.MyClass_MoneyManage()  # 资金管理类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------

# 方法说明：
'''
# join()方法：主线程A中，创建了子线程B，并且在主线程A中调用了B.join()，那么，主线程A会在调用的地方等待，直到子线程B完成操作后，才可以接着往下执行，那么在调用这个线程时可以使用被调用线程的join方法。
# 原型：join([timeout])
# 里面的参数时可选的，代表线程运行的最大时间，即如果超过这个时间，不管这个此线程有没有执行完毕都会被回收，然后主线程或函数都会接着执行的。

# setDaemon() 方法(注意在 ipython与run 中结果不一样，ipython会无视setDaemon方法)。主线程A中，创建了子线程B，并且在主线程A中调用了B.setDaemon(),这个的意思是，把主线程A设置为守护线程，这时候，要是主线程A执行结束了，就不管子线程B是否完成,一并和主线程A退出.这就是setDaemon方法的含义，这基本和join是相反的。此外，还有个要特别注意的：必须在start() 方法调用之前设置，如果不设置为守护线程，程序会被无限挂起。

# lock() 方法，本质是阻断其他线程。不同线程间的数据共享
# 一个进程所含的不同线程间共享内存，这就意味着任何一个变量都可以被任何一个线程修改，因此线程之间共享数据最大的危险在于多个线程同时改一个变量，把内容给改乱了。如果不同线程间有共享的变量，其中一个方法就是在修改前给其上一把锁lock，确保一次只有一个线程能修改它。threading.lock()方法可以轻易实现对一个共享变量的锁定，修改完后release供其它线程使用。比如下例中账户余额balance是一个共享变量，使用lock可以使其不被改乱。
'''

# param_list = [[],[],[]] 表示多组参数，里面元素为 func 的参数，要以[...]来写.

#%% 简单测试，是否结构返回的结果
def worker(n):
    import time
    print("worker")
    print(n)
    time.sleep(1)
    return n
myparallel.multi_threading(worker,([1],[2],[3],[4],[5]))
myparallel.multi_threading(worker,([1],[2],[3],[4],[5]))
myparallel.multi_threading(worker,([[1,2]],[(3,4)],[5]))

myparallel.multi_threading_list(worker,([1],[2],[3],[4],[5]))
myparallel.multi_threading_list(worker,([[1,2]],[3],[4],[5]))

myparallel.multi_threading_para_df(worker,([[1],[2],[3],[4],[5]]))
myparallel.multi_threading_para_df(worker,([[1,2]],[3],[(4,)],[5]))

myparallel.multi_threading_dict(worker,([[1,2]],((3,4),),[5])) # [5, 3, 2, 4, 1]
myparallel.multi_threading_dict(worker,[[1],[2],[3],[4],[5]]) # [5, 3, 2, 4, 1]


#%%
def worker(*args):
    import time
    print("args=",args)
    print("*args=",*args)
    time.sleep(1)
    return 1
myparallel.multi_threading(worker,[[[1],[2]],[3],[4],[5]])
myparallel.multi_threading(worker,[[1],[2],[3],[4],[5]])
myparallel.multi_threading(worker, [[1,2],[(3,4)],[([3,4]),]] )

myparallel.multi_threading_list(worker,[[[1],[2]],[3],[4],[5]])

myparallel.multi_threading_para_df(worker,([[1,2]],[3],[(4,)],[5]) )
myparallel.multi_threading_para_df(worker,([[1,2],5],[3,[1]],[(4,),3],[5,6]) )

myparallel.multi_threading_dict(worker,([[1,2]],[3],[(4,)],[5]))
myparallel.multi_threading_dict(worker,([[1,2],5],[3,[1]],[(4,),3],[5,6]) )
myparallel.multi_threading_dict(worker,([[1,2],5],[3,[1]],[(4,),3],[5,6]) )

# myparallel.multi_threading_concat_df(worker,([[1,2]],[3],[(4,)],[5]))
# myparallel.multi_threading_concat_df(worker,([[1,2],5],[3,[1]],[(4,),3],[5,6]) )
# myparallel.multi_threading_concat_df(worker,([[1,2],5],[3,[1]],[(4,),3],[5,6]) )

#%% 测试2，多参数
num = 0
def add(a,b,c):
    global num
    print(a,b,c)
    return [num,a,b,c]
param_list = [["A","B","C"],[1,2,3],["A",1,3],[4,1,2],[8,"DE",0]]
myparallel.multi_threading(add,param_list=param_list)
myparallel.multi_threading_list(add,param_list=param_list)
myparallel.multi_threading_para_df(add, param_list=param_list)
myparallel.multi_threading_dict(add, param_list=param_list)
myparallel.multi_threading_concat_df(add, param_list=param_list)


#%% 测试3，返回df
def add(a,b,c):
    print(a, b, c)
    return pd.DataFrame([(a,b,c),(a,b,c)],columns=["A","B","C"])
param_list = [["A","B","C"],[1,2,3],["A",1,3],[4,1,2],[8,"DE",0]]
myparallel.multi_threading(add,param_list=param_list)
myparallel.multi_threading_list(add,param_list=param_list)
myparallel.multi_threading_para_df(add, param_list=param_list)
myparallel.multi_threading_dict(add, param_list=param_list)
myparallel.multi_threading_concat_df(add, param_list=param_list)





#%% lock阻断测试
lock = myparallel.lock()
lock1 = myparallel.lock()
# myparallel.lock().release() # 会失败
# lock1.acquire() # lock1锁对象锁住
# lock1.acquire() # 会阻断相同的对象lock1的这句语句，非多线程会卡住。

import random
tickt_count = 10
def run():
    # lock.acquire()
    global tickt_count
    while tickt_count > 0:
        print('notice:There has %d tickts remain ' %(tickt_count))
        if tickt_count > 2:
            number = random.randint(1,2)
        else:
            number = 1
        tickt_count -= number
        print('have buy %d tickt,the remain tickt\'t count is %d .Already buy \n' % (number, tickt_count))
    # lock.release()

for i in range(100):
    tickt_count = 10
    myparallel.multi_threading_para_df(run,param_list=[[],[],[]])
    print('tickt count ',tickt_count) # 不lock，会出现-1
    if tickt_count == -1:
        raise ValueError("结果出现-1")








#%% threading.activeCount()的使用，此方法返回当前进程中线程的个数。返回的个数中包含主线程。
import threading
import time

def worker():
    print("test")
    time.sleep(1)

for i in range(5):
    t = threading.Thread(target=worker)
    t.start()

print("current has %d threads" % (threading.activeCount() - 1))

#%% threading.enumerate()的使用。此方法返回当前运行中的Thread对象列表。
import threading
import time

def worker():
    print("test")
    time.sleep(2)

threads = []
for i in range(5):
    t = threading.Thread(target=worker)
    threads.append(t)
    t.start()

for item in threading.enumerate():
    print(item)

for item in threads:
    print(item)


#%% Lock() # 多运行几次，可以发现一个现象，那就是结果不正确。
# 默认情况，当一个 Lock 是 locked 状态时调用 acquire()，会阻塞线程本身。
# 但我们可以设置不阻塞，或者是阻塞指定时间。
# lock.acquire(False)
#阻塞指定时间，如 3 秒钟,当然 python3 的版本才有这个功能
# lock.acquire(timeout=3)

# lock() 方法，本质是阻断其他线程。不同线程间的数据共享
# 一个进程所含的不同线程间共享内存，这就意味着任何一个变量都可以被任何一个线程修改，因此线程之间共享数据最大的危险在于多个线程同时改一个变量，把内容给改乱了。如果不同线程间有共享的变量，其中一个方法就是在修改前给其上一把锁lock，确保一次只有一个线程能修改它。threading.lock()方法可以轻易实现对一个共享变量的锁定，修改完后release供其它线程使用。比如下例中账户余额balance是一个共享变量，使用lock可以使其不被改乱。
import threading
import time
lock = threading.Lock()

num=0
def run(n):
    lock.acquire() # 会慢速打印
    print(n)
    # lock.acquire() # 会告诉打印
    time.sleep(0.05)
    global num
    num +=1
    lock.release()

num = 0
t_objs = [] #存线程实例
for i in range(50):
    t = threading.Thread(target=run,args=[i])
    t.start()
    t_objs.append(t) #为了不阻塞后面线程的启动，不在这里join，先放到一个列表里

for t in t_objs: #循环线程实例列表，等待所有线程执行完毕
    t.join()

print("----------all threads has finished...",threading.current_thread(),threading.active_count())

print("num:",num)


#%%
# setDaemon() 方法(注意在 ipython与run 中结果不一样，ipython会无视setDaemon方法)。主线程A中，创建了子线程B，并且在主线程A中调用了B.setDaemon(),这个的意思是，把主线程A设置为守护线程，这时候，要是主线程A执行结束了，就不管子线程B是否完成,一并和主线程A退出.这就是setDaemon方法的含义，这基本和join是相反的。此外，还有个要特别注意的：必须在start() 方法调用之前设置，如果不设置为守护线程，程序会被无限挂起。

import threading
import time

def run(n):
    print("task ",n )
    time.sleep(1)
    print("task done",n,threading.current_thread())

start_time = time.time()
t_objs = [] #存线程实例
for i in range(10):
    t = threading.Thread(target=run,args=("t-%s" %i ,))
    t.setDaemon(True) # 把当前线程设置为守护线程
    t.start()
    t_objs.append(t) # 为了不阻塞后面线程的启动，不在这里join，先放到一个列表里

# for t in t_objs: # 循环线程实例列表，等待所有线程执行完毕
#     t.join()

time.sleep(2)
print("----------all threads has finished...",threading.current_thread(),threading.active_count())
print("cost:",time.time() - start_time)
# run("t1")
# run("t2")


#%% ipython中会无视守护线程。只有python的run才行。
#守护进程，即主线程结束以后所有的其它线程也立即结束，不用等其它线程执行完毕；正常情况即使没加join主线程执行完毕当其它线程未执行完毕程序也不会退出，必须等待所有线程执行完毕程序才结束，类似主程序在末尾有默认的join
def test1(x):
    time.sleep(2)
    print("i an other Thread",x**x)

for i in range(5):
    t = threading.Thread(target=test1, args=(i,))
    t.setDaemon(True)
    t.start()

print("Main Thread is done") #整个程序结束，不会等待守护线程打印操作执行完毕就直接结束了

