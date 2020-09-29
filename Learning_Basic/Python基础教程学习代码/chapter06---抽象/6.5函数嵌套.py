# Author:Zhang Yuan

def func_first(num):
    def func_second(count):
        return "%s+%s"%(num,count)
    return func_second
print(func_first(2)(4))

def func_first(num):
    def func_second(count):
        return "%s+%s"%(num,count)
    return func_second(5)
print(func_first(2))

def func_first(num):
    def func_second(count):
        return "%s+%s"%(num,count)
    return func_second(num)
print(func_first(2))

