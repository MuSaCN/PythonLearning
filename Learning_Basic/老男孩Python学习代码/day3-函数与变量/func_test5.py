# Author:Zhang Yuan

#*args：参数组传递,*表示接受的参数个数不固定
#接受N个位置参数（不是关键字参数），转换成元组方式
def test(*args):
    print(args)
test(1,2,3,4,5)
test(*[1,2,3,4,5,6])  #加*把列表变成元组

#**kwargs：把N个关键字参数，转换成字典的方式
#接受关键字参数
def test2(**kwargs):
    print(kwargs)
test2(name="ABC",age=12,sex="male")
test2(**{"name":"BCDE","age":12,"Sex":"female"})

def test3(name,**kwargs):
    print(name)
    print(kwargs)
test3('ales',age=18,sex="male")

def test4(name,age=18,**kwargs):
    print(name)
    print(age)
    print(kwargs)
test4('abc',sex='m',hobby='testla')
test4('abc',sex='m',hobby='testla',age=3)

def test5(name,age=18,*args,**kwargs):
    print(name)
    print(age)
    print(args)
    print(kwargs)
    logger("Test5")

#在一个函数里引用其他函数
def logger(source):
    print("from %s"%source)

test5("Test5",age=34,sex="m",hobby="Tesla")

#局部变量
def change_name(name):
    print("before change",name)
    name="ABCDE" #局部变量，只在函数内生效
    print("after change",name)

name="musa"
change_name(name)
print("origianl name=",name)

def change_name1(name1):
    global name#加入这个才可以在里面改全局的变量,加入global声明这个变量是全局的
    print("before change",name)
    name="ABCDE" #局部变量，只在函数内生效
    print("after change",name)

change_name1(name)
print("origianl name=",name)

'''
# 以下内容不要用，语法正确，但非常容易出错
def change2():
    global A
    A="GFDSA"
change2()
print(A)
'''

#列表，字典，集合，类 都可以通过函数改
school ="oldboy"
names=["A","B","C"]
def change2():
    print(names)
    names[0]="D"
    print(names)
change2()
print(names)




