# Author:Zhang Yuan

#python定义函数可以无序，不用于C++需要编译（要有序）
def foo():
    print("in the foo")
    bar()
def bar():
    print("in the bar")
foo()

