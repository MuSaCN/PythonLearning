# Author:Zhang Yuan
#python多线程 不适合cpu密集操作型的任务，适合io操作密集型的任务，就是大量的高密度运算，多线程不一定提高效率。多线程适合轻量级多个任务。

import multiprocessing #多进程
import threading #多线程

def thread_run(index_process,index_thread):
    print("进程:",index_process,"线程:",index_thread," thread id:",threading.get_ident())
#一个进程下多线程
def run(index_process):
    print('hello', index_process)
    tlist=[]
    nthread=10
    for i in range(nthread):
        t = threading.Thread(target=thread_run,args=(index_process,i))
        tlist.append(t)
    for i in range(nthread):
        tlist[i].start()
    for i in range(nthread):
        tlist[i].join()

print(__name__)
if __name__ == '__main__':
    plist=[]
    pnum=8
    for i in range(pnum):
        #设置多进程
        p = multiprocessing.Process(target=run, args=(i,))
        plist.append(p)
    for i in range(pnum):
        plist[i].start()
    for i in range(pnum):
        plist[i].join()


