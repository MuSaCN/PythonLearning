__author__ = "Alex Li"

import threading
import time

def run(n):
    print("task ",n)
    time.sleep(1)
    print("task done", n)
    return n

# for i in range(10):
#     run(i)


start_time = time.time()
t_objs = [] #存线程实例
for i in range(10):
    t = threading.Thread(target=run,args=[i])
    t.start()
    t_objs.append(t) #为了不阻塞后面线程的启动，不在这里join，先放到一个列表里

# 有了join会
for t in t_objs: # 循环线程实例列表，等待所有线程执行完毕
    t.join()


print("----------all threads has finished...")
print("cost:",time.time() - start_time)
# run("t1")
# run("t2")


#%%
#如果多个子线程一些join一些没有join主线程怎么处理？？？部分子线程join主线程会等join时间最长的子线程结束后才继续，未参与join的子线程仍然和主线程并行运行
t5 = threading.Thread(target=run, args=(5,))
t6 = threading.Thread(target=run, args=(6,))
t5.start()
t6.start()
t5_join_start_time = time.time()
t5.join()
t5_join_end_time = time.time()
print(123)
print("t5 join time is %s"%(t5_join_end_time - t5_join_start_time)) #实际耗时15s

