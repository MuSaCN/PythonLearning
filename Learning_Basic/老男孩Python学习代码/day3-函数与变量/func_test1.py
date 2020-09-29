# Author:Zhang Yuan

#定义一个函数
def function1():
    """testing1,注释最好写上"""
    print("you are in the Testing")
    return True

#定义一个过程(没有返回值)
#python中过程也有返回值None
def function2():
    """Testing2"""
    print("you are in the Testing2")

x=function1()
y=function2()

print("from func1 return is %s"%x)
print("from func2 return is %s"%y)






