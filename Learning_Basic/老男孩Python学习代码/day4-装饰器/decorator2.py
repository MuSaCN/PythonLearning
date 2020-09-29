# Author:Zhang Yuan
import time
#函数中返回函数
def bar():
    print("in the bar")
def test(func):
    print(func)
    start_time=time.time()
    func()
    stop_time=time.time()
    print("the func run time is %s" %(stop_time-start_time))
test(bar)


def bar1():
    time.sleep(1)
    print("in the bar1")
def test2(func):
    print(func)
    return func
#test2(bar1())相当于把bar1()函数返回值传给函数，不符合高阶函数
test2(bar1())
#t=test2(bar1)才是高阶函数
t=test2(bar1)
t()
bar1=test2(bar1)
