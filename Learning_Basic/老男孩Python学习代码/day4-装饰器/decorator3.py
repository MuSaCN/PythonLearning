# Author:Zhang Yuan
#函数的嵌套:在一个函数体内再次def定义一个函数
def foo():
    print("in the foo")
    def bar(): #局部定义函数，不能外部直接调用
        print("in the bar")
    bar()
foo()

x=0
def grandpa():
    x=1
    def dad():
        x=2
        def son():
            x=3
            print(x)
        son()
    dad()
grandpa()