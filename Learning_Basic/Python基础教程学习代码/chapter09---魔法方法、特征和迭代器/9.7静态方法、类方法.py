# Author:Zhang Yuan

class A(object):
    bar = 1
    def foo(self):
        print('foo')
    @staticmethod
    # 这里的self不是实例中的self，仅仅是个参数，如果没有@staticmethod，self表示实例
    def static_foo(self):
        print('static_foo')
        print(A.bar)
    @classmethod
    # 这里必须要有cls，表示当前的类
    def class_foo(cls):
        print('class_foo')
        print(cls.bar)
        A().foo()

A.static_foo(123)  #static_foo ， 1
A.class_foo()  #class_foo， 1， foo

class Dog(object):
    def __init__(self,name):
        self.name = name
    @staticmethod #实际上跟类没什么关系了
    def eat(self):
        print("%s is eating %s" %(self.name,'dd'))
d = Dog("ChenRonghua")
d.eat(d) #实例下调用，要把实例本身d传进去,但是这样影响直接调用

class Dog(object):
    name = "huazai"
    def __init__(self,name):
        self.name = name
    @classmethod
    def eat(cls,self):
        print("%s is eating %s" %(cls.name,'dd'),self.name)
    def talk(self):
        print("%s is talking"% self.name)
#Dog.eat()
d=Dog("ChenRongHua")
d.eat(d) #类方法也可以传递实例变量，但是这样影响直接调用