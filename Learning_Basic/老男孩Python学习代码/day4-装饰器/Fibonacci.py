# Author:Zhang Yuan
#普通函数形式
def fib_func(max):
    n,a,b=0,0,1
    while n<max:
        #yield b
        print(b)
        #不是依次赋值，相当于：
        #t=(b,a+b) t是一个tuple
        #a=t[0] b=t[1]
        a,b=b,a+b #tuple式赋值
        n=n+1
    return "done"

#生成器形式
def fib(max):
    n,a,b=0,0,1
    while n<max:
        yield b
        #print(b)
        #不是依次赋值，相当于：
        #t=(b,a+b) t是一个tuple
        #a=t[0] b=t[1]
        a,b=b,a+b #tuple式赋值
        n=n+1
    return "done" #带上这句异常时返回的值

f=fib(5)
print("**********************")
'''由于数量不知道，存在异常
print(f.__next__())
print(f.__next__())
print(f.__next__())
print(f.__next__())
print(f.__next__())
print(f.__next__())
print(f.__next__())
print(f.__next__())
'''
while True:
    try:
        x=next(f) #x=f.__next__
        print(x)
    except StopIteration as e:
        print("Finish:",e.value)
        break


