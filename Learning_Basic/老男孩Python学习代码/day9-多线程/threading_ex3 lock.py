__author__ = "Alex Li"

import threading
import time
lock = threading.Lock()

# 不同线程间的数据共享
# 一个进程所含的不同线程间共享内存，这就意味着任何一个变量都可以被任何一个线程修改，因此线程之间共享数据最大的危险在于多个线程同时改一个变量，把内容给改乱了。如果不同线程间有共享的变量，其中一个方法就是在修改前给其上一把锁lock，确保一次只有一个线程能修改它。threading.lock()方法可以轻易实现对一个共享变量的锁定，修改完后release供其它线程使用。比如下例中账户余额balance是一个共享变量，使用lock可以使其不被改乱。

#%%
num=0
def run(n):
    lock.acquire() # 会慢速打印
    print(n)
    # lock.acquire() # 会告诉打印
    time.sleep(0.05)
    global num
    num +=1
    lock.release()


num = 0
t_objs = [] #存线程实例
for i in range(50):
    t = threading.Thread(target=run,args=[i])
    t.start()
    t_objs.append(t) #为了不阻塞后面线程的启动，不在这里join，先放到一个列表里

for t in t_objs: #循环线程实例列表，等待所有线程执行完毕
    t.join()

print("----------all threads has finished...",threading.current_thread(),threading.active_count())

print("num:",num)
