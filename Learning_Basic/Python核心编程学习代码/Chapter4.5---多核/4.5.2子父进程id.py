# Author:Zhang Yuan
from multiprocessing import Process
import os
#每一个子进程都是由父进行启动的

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid()) #这里指pycharm的进程pid
    print('process id:', os.getpid())
    print("\n\n")


def f(name):
    info('\033[31;1mcalled from child process function f\033[0m')
    print('hello', name)

if __name__ == '__main__':
    info('\033[32;1mmain process line\033[0m')
    p = Process(target=f, args=('bob',))
    p.start()
    # p.join()