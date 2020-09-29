# Author:Zhang Yuan
#锁能够防止多个线程同时进入公共临界区
import threading,time
num = 0
#创建lock对象，这句必须要有，指向一个锁。且只能acquire()一次------------------------------------
lock = threading.Lock()
def run(n):
    #如果不加入锁，print的内容会混乱
    lock.acquire() #获得锁，锁住状态。每次只能有一个线程访问
    global num
    print("first",n,num)
    num +=1
    print("second",n,num)
    time.sleep(0.01)
    lock.release() #释放锁，没有锁状态。排队的进程可以访问。
    #或者用with语句也可以
    with lock:
        print("third", n, num)
        num += 1
        print("fourth", n, num)
        time.sleep(0.01)
t_objs = [] #存线程实例
for i in range(50):
    t = threading.Thread(target=run,args=(i,))
    t.start()
    t_objs.append(t) #为了不阻塞后面线程的启动，不在这里join，先放到一个列表里
for t in t_objs: #循环线程实例列表，等待所有线程执行完毕
    t.join()
print("----------all threads has finished...",threading.current_thread(),threading.active_count())
print("num:",num)

#----------------------------------------------------------------------------------------------
import threading, time
num, num2 = 0, 0
Relock = threading.RLock() #RLock()可以多次上锁，且不会阻塞----------------------------------------
def run1():
    print("grab the first part data")
    Relock.acquire()
    global num
    num += 1
    Relock.release()
    return num

def run2():
    print("grab the second part data")
    Relock.acquire()
    global num2
    num2 += 1
    Relock.release()
    return num2

def run3():
    Relock.acquire()
    res = run1()
    print('--------between run1 and run2-----')
    res2 = run2()
    Relock.release()
    print(res, res2)

for i in range(1):
    t = threading.Thread(target=run3)
    t.start()

while threading.active_count() != 1:
    print(threading.active_count())
else:
    print('----all threads done---')
    print(num, num2)