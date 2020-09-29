# Author:Zhang Yuan
#对于有限资源的应用，使用信号量是个不错的选择
#threading.BoundedSemaphore()限制多线程的访问
import threading, time
# 最多允许5个线程同时运行：0-4运行完，才可以运行下一组
semaphore = threading.BoundedSemaphore(5)
def run(n):
    semaphore.acquire()
    time.sleep(1)
    print("run the thread: %s\n" % n)
    semaphore.release()
    with semaphore:
        time.sleep(1)
        print("run the thread: %s\n" % n)

if __name__ == '__main__':
    for i in range(22):
        t = threading.Thread(target=run, args=(i,))
        t.start()
while threading.active_count() != 1:
    pass  # print(threading.active_count())
else:
    print('----all threads done---')
    #print(num)




