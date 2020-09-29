# Author:Zhang Yuan
# 装饰器：本质是函数，为给其他函数添加附加功能
# 原则：1.不能修改被装饰函数的源代码
#       2.不能修改被装饰函数的调用方式
# 实现装饰器知识储备：
# 1.函数即“变量”
# 2.高阶函数
#   a:把一个函数名当做实参传给另一个函数(在不修改被装饰函数源代码的情况下为其添加功能)
#   b:返回值中包含函数名(不修改函数的调用方式)
# 3.嵌套函数
# 高阶函数+嵌套函数=>装饰器

import time
def timmer(func):
    def warpper(*args,**kwargs):
        start_time=time.time()
        func()
        stop_time=time.time()
        print("the func run time is %s" %(stop_time-start_time))
    return warpper  #注意这里不是warpper()

@timmer
def test1():
    time.sleep(1)
    print("in the test1")
test1()