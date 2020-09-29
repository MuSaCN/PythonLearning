# Author:Zhang Yuan
import time
def consumer(name):
    print("%s准备吃包子啦"%name)
    while True:
        baozi=yield
        print("包子[%s]来了，被[%s]吃了"%(baozi,name))

#c=consumer("ZhangYuan")
# c.__next__()
# b1="韭菜馅"
# c.send(b1)
# c.__next__()

#单线程并行运算
def producer(name):
    c=consumer("A")  #相当于把函数变成生成器，不执行里面的print()
    c2=consumer("B") #相当于把函数变成生成器，不执行里面的print()
    c.__next__()     #相当于走到yield
    c2.__next__()    #相当于走到yield
    print("我开始准备做包子啦")
    for i in range(10):
        time.sleep(0)
        print("做了一个包子，分两半")
        c.send(i)
        c2.send(i)
producer("ZhangYuan")

A=[1,2,3,4,5,6,7,8,9]
#把A变成Iterator迭代器,Iterator对象表示一个数据流，可以无穷大
#python3底层很多都是迭代器
D=iter(A)
print(D)
print(D.__next__())
print(D.__next__())
print(D.__next__())


