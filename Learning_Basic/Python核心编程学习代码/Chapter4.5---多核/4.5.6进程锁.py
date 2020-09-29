# Author:Zhang Yuan
#进程锁目的在于打印时不会乱，不会出现打印插值
from multiprocessing import Process, Lock

def f(l, i):
    #加上锁，i的顺序依然是乱的，但是print内容不会错乱
    l.acquire()
    print('hello world', i)
    l.release()

if __name__ == '__main__':
    lock = Lock()
    for num in range(100):
        Process(target=f, args=(lock, num)).start()

