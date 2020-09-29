# Author:Zhang Yuan
from time import sleep
from queue import Queue,PriorityQueue
from MyThread import MoreThread

def writer(queue, loops):
    for i in range(loops):
        print('producing object for Q...')
        queue.put('xxx', 1)
        print("size now", queue.qsize())

def reader(queue, loops):
    for i in range(loops):
        val = queue.get(1)
        print('consumed object from Q... size now', queue.qsize())

funcs = [writer, reader]
nfuncs = range(len(funcs))

def main():
    nloops = 5
    q = Queue(32) #建立一个先入先出队列
    threads = []
    for i in nfuncs:
        t = MoreThread(funcs[i], (q, nloops), funcs[i].__name__)
        threads.append(t)
    for i in nfuncs:
        threads[i].start()
    for i in nfuncs:
        threads[i].join()
    print('all DONE')

if __name__ == '__main__':
    main()

#优先级队列
q = PriorityQueue()
q.put((-1,"A"))
q.put((3,"B"))
q.put((10,"C"))
q.put((6,"D"))
print(q.get())
print(q.get())
print(q.get())
print(q.get())
