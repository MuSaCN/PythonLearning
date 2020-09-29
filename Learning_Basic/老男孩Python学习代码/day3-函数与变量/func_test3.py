# Author:Zhang Yuan
#没有返回结果，返回：None
#1个返回结果，返回：对象object
#多个返回结果，返回：元组tuple

def test1():
    print("in the test1")
def test2():
    print("in the test2")
    return 0
def test3():
    print("in the test3")
    return 1,"Hello",["A","B","C"],{"name":"Angle"}
def test4():
    print("in the test4")
    return test1()

x=test1()
y=test2()
z=test3()
out=test4()
print(x,y,z)
print(out)


