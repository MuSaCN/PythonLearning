# Author:Zhang Yuan

class B(Exception):pass
class C(B):pass
class D(C):pass
class E:pass
#如果发生的异常和 except 子句中的类是同一个类或者是它的基类，则异常和except子句中的类是兼容的
for cls in [B, C, D]:
    try:
        raise cls()
    #引发异常时执行
    except B:
        print("B")
    except D:
        print("D")
    except C:
        print("C")
    #不引发异常时执行
    else:
        print("OK")
    #在所有情况下执行
    finally:
        print("End")

#global 语句可以用来指明某个特定的变量位于全局作用域并且应该在那里重新绑定； nonlocal 语句表示否定当前命名空间的作用域，寻找父函数的作用域并绑定对象。
def scope_test():
    def do_local():
        spam = "local spam"
    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam"
    def do_global():
        global spam
        spam = "global spam"
    spam = "test spam"
    do_local()
    print("After local assignment:", spam)
    do_nonlocal()
    print("After nonlocal assignment:", spam)
    do_global()
    print("After global assignment:", spam) #先找里面的，再外面的
scope_test()
print("In global scope:", spam)

#Python3的多重继承中，继承的类有共同的基类：按照广度优先、从左到右寻找父类和兄弟类。继承的类没有共同基类：按照深度优先、从左到右寻找不同的父类
class Base:
    def show(self):
        print("Base")
class A(Base):
    pass
class B(Base):
    def show(self):
        print("B")
class C():
    def show(self):
        print("C")
#D1继承的A、B都有继承共同基类，属于兄弟类，所以按照广度优先、从左到右
class D1(A,B):
    pass
d1=D1()
d1.show() #B
#D2继承的A、C没有共同基类，所以按照深度优先、从左到右
class D2(A,C):
    pass
d2=D2()
d2.show() #Base

#Python中的类或类实例，可以自定义添加变量。不同于C++类的严格封装作用。
class Employee:
    pass
john = Employee()  # Create an empty employee record
# Fill the fields of the record
john.name = 'John Doe'
john.dept = 'computer lab'
john.salary = 1000
print(john.name)
J=Employee
J.age=23
print(john.age)

#for语句中底层会默认调用容器对象中的 iter()，产生迭代器，所以会有for循环。
# 在幕后，for 语句会调用容器对象中的 iter()
for element in [1, 2, 3]:
    print(element)
#__iter__(),__next__()让类迭代器化
class Reverse:
    """Iterator for looping over a sequence backwards."""
    def __init__(self, data):
        self.data = data
        self.index = len(data)
    def __iter__(self):
        return self
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]
rev = Reverse('spam')
print(rev,iter(rev))
for char in rev:
    print(char)
#生成器中用yield写生成器使得语法更为紧凑，因为它会自动创建 __iter__() 和 __next__() 方法。
def reverse(data):
    for index in range(len(data)-1, -1, -1):
        yield data[index]
for char in reverse('golf'):
    print(char)
#生成器表达式语法类似列表推导式，将外层为圆括号而非方括号。这种表达式返回生成器，且可以直接被外部函数直接使用(重叠的括号可以省略)。
print( sum(i*i for i in range(10))  )






