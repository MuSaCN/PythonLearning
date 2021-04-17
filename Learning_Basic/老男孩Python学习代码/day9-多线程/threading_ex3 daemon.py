__author__ = "Alex Li"

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




