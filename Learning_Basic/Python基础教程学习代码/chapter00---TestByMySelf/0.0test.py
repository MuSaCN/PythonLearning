# Author:Zhang Yuan
#study
print(round(1.3),round(1.7))

a=[None,2]
print(a*3)

print(2 in ["a",2])
print("b" in ("b",2))
print("C" in "China")

a=[1,2,3,["a","b"]]
b=a #建立指针
c=a.copy()  #浅复制，只能复制第一层，更高层建立指针
from copy import deepcopy
d=deepcopy(a) #深复制，复制内存

print("My name is %s"%"ABC")
print("My {a1} is {a2}".format(a1="age",a2="32"))

phone={"A1":"123","A2":"456"}
print("He is phone number is {A1}".format_map(phone))

print(phone.get("A3","ABC"))
print(phone.setdefault("A3"))
print(phone.setdefault("A3","ABC"))
print(phone.get("A3"))  # --- None
print(phone.get("A3","ABC"))  # ---ABC
print(phone.setdefault("A3"))  # --- None，还添加了A3：None
print(phone.setdefault("A3","ABC"))  # ---None，访问A3，因为有了

list1=[1,2,3,4,5]
list2=["a","b","c","d","e"]
print(list(zip(list1,list2)))

a=[x for x in range(10)]
print(a) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
b=[x*x for x in range(10)]
print(b) #[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
c=[(x,y) for x in range(3) for y in range(4) if x<y]
print(c) #[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

s={i:"{} squared is {}".format(i,i**2) for i in range(4)}
print(s) #{0: '0 squared is 0', 1: '1 squared is 1', 2: '2 squared is 4', 3: '3 squared is 9'}

exec("print(123)") #不返回结果，只是执行

print(eval("print(123),1+1")) #返回(None, 2)

def function(x):
    "This is function document test."
    pass
print(function.__doc__)
help(function)

#切片赋值，也只能复制第一层
list1=[1,"a",2,["ABC",123]]
n=list1[:]
print(n,list1) #[1, 'a', 2, ['ABC', 123]] [1, 'a', 2, ['ABC', 123]]
n[0]=2
n[-1][-1]=456
print(n,list1) #[2, 'a', 2, ['ABC', 456]] [1, 'a', 2, ['ABC', 456]]
list1=[1,"a",2]
def test(l1):
    l1.append(123)
test(list1)
print(list1) #[1, 'a', 2, 123]

#参数收集
def func(*args,**kwargs):
    print(args)
    print(kwargs)
func(1,2,3,"a","b",key1="name",key2="age",key3="sex")
#(1, 2, 3, 'a', 'b')
#{'key1': 'name', 'key2': 'age', 'key3': 'sex'}

#如果以序列形式分配，要加上*
def test1(a,b,c,d):
    print(a,b,c,d)
List1=[1,"a",2,"b"]
Tuple1=(4,3,2,1)
test1(*List1) #1 a 2 b
test1(*Tuple1) #4 3 2 1
#如果以字典形式分配，要加上**，且key相同
def test2(m,n):
    print(m,n)
Dict1={"m":"name","n":"age"}
test2(**Dict1) #name age

x,y,z=1,2,3
def testA():
    global x #全局变量，指向最外层
    x,y,z=10,20,30 #全局x重新赋值
    print("first:",x,y,z) # first: 10 20 30
    def testB():
        global y #指向最外层，不是上一层
        y+=10 #所以y不是上一层的10，而是最外部的2
        x=20 #第二层没有声明x，所以x是局部
        print("second:",x,y,z) # second: 20 12 30
        def testC():
            nonlocal x
            x+=10
            global z
            print("third:",x,y,z) # third: 30 12 3
        testC()
        print("second2:",x,y,z) # second2: 30 12 30
    testB()
print(testA(),x,y,z) # None 10 12 3


def test1():
    def test2():
        print("test2 in test1")
    return test2() #返回的是函数test2()的运行
test1() # test2 in test1

def test3():
    def test4():
        print("test4 in test3")
    return test4 #返回的是函数test4的本体
test3()() # test4 in test3,其中的test3()就相当于test4,所以需要加()才可以运行


class A:
    a=123                  #类变量
    def function(self):
        self.var1=456      #实例化变量
        A.var2=456         #在功能中定义类变量：实例化且调用此功能才有。
        var3=789           #在函数中定义变量
ca=A()
print(A.a) #123
ca.function()
print(A.a,A.var2) #123 456

cb=A()
print(cb.a,cb.var2) #123 456，由于类被改变，再次新的实例会有

class TEST:
    test="ABC"
    pass
#t.***为实例化操作
#TEST.***为类操作
t=TEST()
t.test="DEF"  #实例化操作，单变量赋值等同于定义
print(t.test) #DEF
del t.test
print(t.test) #ABC

class TEST:
    __var1=123 #双下划线设置了私有化，无法正常访问
a=TEST()
print(a._TEST__var1)  #以此方式才可以访问私有化
print(TEST._TEST__var1)


def foo(x):return x*x
foo=lambda x:x*x

from abc import ABC,abstractmethod
class A(ABC):
    @abstractmethod
    def func(self):
        pass
class B(A):
    def func(self): #限定的抽象方法，必须被重写才可以实例化
         print("OK")
    pass
b=B()
b.func()
#一些内置的异常类
# raise Exception
# raise AttributeError
# raise OSError
# raise IndexError
# raise KeyError
# raise NameError
# raise SyntaxError
# raise TypeError
# raise ValueError
# raise ZeroDivisionError


try:
    #...执行内容
    raise TypeError #执行了raise异常
#设定当前异常怎么处理
except TypeError:
    print("raise TypeError") #设定了print内容
except KeyError:
    print("raise KeyError")
#没有异常时
else:
    print("No Error")
#无论try子句发生什么finally都将运行。常用于清理
finally:
    print("unless any try")

class Test:
    # 构造函数，创造实例默认执行
    def __init__(self):
        self.date={} #这个方法通常要设置字典为数据类型
        print("Default start")
    # 析构函数，程序全部运行完执行
    def __del__(self):
        pass
    # 让实例可直接设置:A[key]=...
    def __setitem__(self, key, value):
        self.date[key]=value
    # 让实例可直接访问:A[item]
    def __getitem__(self, item):
        return self.date[item]
    # 让实例可直接运行del A[key]
    def __delitem__(self, key):
        del self.date[key]
    # 让实例可直接运行len(A)
    def __len__(self):
        return len(self.date)

A=Test()
#默认运行实例中的__setitem__
A["k1"]="abc"
A["k2"]="ABC"
#默认运行实例中的__getitem__
print(A["k1"],A["k2"]) #abc ABC
print(A.date) #{'k1': 'abc', 'k2': 'ABC'}
#默认运行实例中的__len__
print(len(A)) #2
#默认运行__delitem__
del A["k1"]
print(A.date) #{'k2': 'ABC'}

class C:
    def __init__(self):
        self._x = 123
    def getx(self):
        return self._x
    def setx(self, value):
        self._x = value
    def delx(self):
        del self._x
    x = property(getx, setx, delx, "I'm the 'x' property.")
c=C()
print(c.x) #c.x will invoke the getter
c.x=456 #c.x = value will invoke the setter
del c.x #del c.x the deleter


print("----------------------------")
it=iter(range(10))
for i in it:
    print(i)



