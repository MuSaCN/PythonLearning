# Author:Zhang Yuan
# import sys,pprint
# ModulePath=sys.path
# ModulePath.append("C:\\Users\\i2011\\PycharmProjects\\Python核心编程学习代码\\Chapter4---多线程")
#python多线程 不适合cpu密集操作型的任务，适合io操作密集型的任务，就是大量的高密度运算，多线程不一定提高效率。多线程适合轻量级多个任务。

import MyThread
from time import ctime, sleep

def fib(x):
    sleep(0.005)
    if x < 2: return 1
    return (fib(x-2) + fib(x-1))
def fac(x):
    sleep(0.1)
    if x < 2: return 1
    return (x * fac(x-1))
def sum(x):
    sleep(0.1)
    if x < 2: return 1
    return (x + sum(x-1))

funcs = (fib, fac, sum)
n = 12

def main():
    nfuncs = range(len(funcs))

    print('*** SINGLE THREAD')
    for i in nfuncs:
        print('starting', funcs[i].__name__,'at:', ctime())
        print(funcs[i](n))
        print(funcs[i].__name__, 'finished at:',ctime())

    print('\n*** MULTIPLE THREADS')
    threads = []
    for i in nfuncs:
        t = MyThread.MoreThread(funcs[i], (n,),funcs[i].__name__)
        threads.append(t)
    for i in nfuncs:
        threads[i].start()
    for i in nfuncs:
        threads[i].join()
        print(threads[i].getResult())

    print('all DONE')

if __name__ == '__main__':
    main()