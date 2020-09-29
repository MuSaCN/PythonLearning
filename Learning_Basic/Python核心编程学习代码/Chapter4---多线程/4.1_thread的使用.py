# Author:Zhang Yuan
#不建议使用_thread模块，因为主线程退出后，其他线程没有清理直接退出。而使用threading模块，会确保在所有重要的子线程退出前，保持整个进程存活。

import _thread
from time import sleep, ctime
def loop0():
    print('start loop 0 at:', ctime())
    sleep(4)
    print('loop 0 done at:', ctime())
def loop1():
    print('start loop 1 at:', ctime())
    sleep(2)
    print('loop 1 done at:', ctime())
def main():
    print('starting at:', ctime())
    _thread.start_new_thread(loop0,())
    _thread.start_new_thread(loop1,())
    sleep(6) #没有这句话就直接结束了,相当于多线程等待
    print('all DONE at:', ctime())
if __name__ == '__main__':
    main()

#下面不需要特别指定等待时间，本质是通过锁对象的状态监测来执行等待
import _thread as thread
from time import sleep, ctime
SleepLoops = [4, 2] #休息时间
def Func(nloop, nsec, lock):
    print('start loop', nloop, 'at:', ctime())
    sleep(nsec)
    print('loop', nloop, 'done at:', ctime())
    lock.release() #针对锁对象使用，改变锁的状态，释放锁
def main():
    print('starting threads...')
    #锁住线程的对象
    LocksHandle = []
    #对象数量
    nSleep = range(len(SleepLoops))
    #建立锁对象
    for i in nSleep:
        lock = thread.allocate_lock() #得到锁对象
        lock.acquire() #取得锁对象
        LocksHandle.append(lock)
    #执行功能函数
    for i in nSleep:
        #传递3个参数到新线程的功能函数中
        thread.start_new_thread(Func, (i, SleepLoops[i], LocksHandle[i]) )
    #本质通过状态监测来处理多线程等待问题
    for i in nSleep:
        #如果锁对象是锁住状态，一直等待
        while LocksHandle[i].locked():
            pass
    print('all DONE at:', ctime())
if __name__ == '__main__':
    main()















