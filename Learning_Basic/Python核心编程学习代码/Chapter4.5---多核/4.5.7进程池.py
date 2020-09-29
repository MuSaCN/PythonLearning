# Author:Zhang Yuan
#进程池用于限制一次性加载的进程数，防止一次性暴力加载的进程数过多，让cpu瘫痪。
from  multiprocessing import Process, Pool,freeze_support
import os,time

def Foo(i):
    time.sleep(0.5)
    print("in process",os.getpid())
    return i + 100

def Bar(arg):
    print('-->exec done:', arg,os.getpid())

if __name__ == '__main__':
    #freeze_support()
    pool = Pool(processes=2) #允许进程池同时放入5个进程
    print("主进程",os.getpid())
    for i in range(10):
        pool.apply_async(func=Foo, args=(i,), callback=Bar) #callback=回调
        #pool.apply(func=Foo, args=(i,)) #串行
        #pool.apply_async(func=Foo, args=(i,)) #串行
    print('end')
    pool.close() #先要把关闭句柄声明，再进入join等待才可以
    pool.join() #进程池中进程执行完毕后再关闭，如果注释，那么程序直接关闭。.join()




