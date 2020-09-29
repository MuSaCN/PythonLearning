# Author:Zhang Yuan
#doctest可以用来测试模块。这要求模块内的函数注释必须写上测试例子
import sys
ModulePath=sys.path
ModulePath.append("C:\\Users\\i2011\\PycharmProjects\\Python基础教程学习代码\\chapter14---测试基础")
import MyModule
import doctest

if __name__ == "__main__":
    print(doctest.testmod(MyModule))



