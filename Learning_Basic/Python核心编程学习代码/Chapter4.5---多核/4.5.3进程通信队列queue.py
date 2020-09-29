# Author:Zhang Yuan
#进程queue与线程queue不同

from multiprocessing import Process, Queue #这与下面的queue不同

def f(qq):
    print("in child:",qq.qsize())
    qq.put([42, None, 'hello'])

if __name__ == '__main__':
    q = Queue()
    q.put("test123")
    p = Process(target=f, args=(q,))
    p.start()
    p.join()
    print("444",q.get_nowait())
    print("444",q.get_nowait())