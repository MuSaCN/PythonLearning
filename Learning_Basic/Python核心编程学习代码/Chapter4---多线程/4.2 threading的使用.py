# Author:Zhang Yuan
#threading模块不需要繁琐的设置就可以轻松关闭线程。




#可调用的函数实例----------------------------------------------------
import threading
import time
SleepLoops = [2, 1] #休息时间
#定义功能函数
def loop(nloop, nsec):
    print('start loop', nloop, 'at:', time.ctime())
    time.sleep(nsec)
    print('loop', nloop, 'done at:', time.ctime())
#主要的多线程函数
def main():
    print('starting at:', time.ctime())
    threads = [] #储存线程句柄
    nSleep = range(len(SleepLoops))
    #传递函数到线程，并且添加线程句柄，这里不执行功能函数。
    for i in nSleep:
        t = threading.Thread(target=loop,args=(i, SleepLoops[i]))
        threads.append(t)
    #开始线程，cup会执行功能函数
    print("开始执行功能函数")
    for i in nSleep:
        threads[i].start()
    #join()函数让线程在执行完后自动结束线程，不需要去设置繁琐的状态监控。
    for i in nSleep:            # wait for all
        threads[i].join()       # threads to finish
    print('all DONE at:', time.ctime())

if __name__ == '__main__':
    main()

#可调用的类实例-----------------------------------------------------
import threading
from time import sleep, ctime
SleepLoops = [2, 1]
class ThreadFunc(object):
    def __init__(self, func, args, name=''):
        self.name = name
        self.func = func
        self.args = args
    def __call__(self):
        self.func(*self.args)
def loop(nloop, nsec):
    print('start loop', nloop, 'at:', ctime())
    sleep(nsec)
    print('loop', nloop, 'done at:', ctime())

def main():
    print('starting at:', ctime())
    threads = [] #存档线程句柄
    n = range(len(SleepLoops))
    for i in n:        # create all threads
        #本质是调用类中__call__
        t = threading.Thread(target=ThreadFunc(loop, (i, SleepLoops[i]),loop.__name__))
        threads.append(t)
    print("开始执行功能类")
    for i in n:        # start all threads
        threads[i].start()
    for i in n:        # wait for completion
        threads[i].join()
    print('all DONE at:', ctime())

if __name__ == '__main__':
    main()

#子类的实例---------------------------------------------------
import threading
from time import sleep, ctime
SleepLoops = [2, 1]
class MyThread(threading.Thread):
    def __init__(self, func, args, name=''):
        threading.Thread.__init__(self, name=name)
        self.func = func
        self.args = args
    #类似于__call__()，这里必须要写成run()
    def run(self):
        self.func(*self.args)
def Func(nloop, nsec):
    print('start loop', nloop, 'at:', ctime())
    sleep(nsec)
    print('loop', nloop, 'done at:', ctime())

def main():
    print('starting at:', ctime())
    threads = []
    n = range(len(SleepLoops))
    for i in n:
        t = MyThread(Func, (i, SleepLoops[i]),Func.__name__)
        threads.append(t)
    print("开始执行子类")
    for i in n:
        threads[i].start()
    for i in n:
        threads[i].join()
    print('all DONE at:', ctime())

if __name__ == '__main__':
    main()









