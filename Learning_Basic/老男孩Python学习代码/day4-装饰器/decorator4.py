# Author:Zhang Yuan
import time
def test1():
    time.sleep(0)
    print('in the test1')
'''
def deco(func):
    start_time=time.time()
    func()
    end_time=time.time()
    print("the func is cost time %s" %(-start_time+end_time))
    return func
'''
def timer(func):
    def deco():
        start_time = time.time()
        func()
        end_time = time.time()
        print("the func is cost time %s" % (-start_time + end_time))
    return deco

# test1=deco(test1)
# test1()
test1=timer(test1)
test1()

@timer #test2=timer(test2)
def test2():
    time.sleep(0)
    print('in the test2')
test2()

############################通用性质的装饰器#####################################
def timer_common(func):
    def deco(*args,**kwargs):
        start_time = time.time()
        func(*args,**kwargs)
        end_time = time.time()
        print("the func is cost time %s" % (-start_time + end_time))
    return deco

@timer_common #此处用@timer就不行了，因为那个装饰器不能传递name参数
def test3(name):
    print("test3:")
test3()


