# Author:Zhang Yuan
# __new__() 表示类实例化过程中，是否要使用该类下 __init__() 方法，__new__() 在 __init__() 之前执行
# __new__()是针对当前类的，与__init__()相同，不是针对继承类和父类的
# __call__()是实例化后，实例加()执行
# 执行顺序__new__() --> __init__() --> __call__()
class Stranger(object):
    pass
class Foo(object):
    def __init__(self, *args, **kwargs):
        print("Foo:__init__")
    #new在init之前，所以并没有实例化的参数self
    def __new__(cls, *args, **kwargs):
        print("Foo:__new__")
        # return object.__new__(Stranger) #如果是这句，指向不是本类，而是其他类，则不执行__init__
        return object.__new__(cls) #这句不写也是不执行__init__
foo=Foo() #Foo:__new__ , Foo:__init__

class Goo(Foo):
    def __init__(self, *args, **kwargs):
        super().__init__()
        print("Goo:__init__")
    #子类不写new，会自动调用父类的new
    def __new__(cls, *args, **kwargs):
        print("Goo:__new__")
        return object.__new__(cls) #通常写法，写此句显然不会执行父类的new
        #return Foo.__new__(cls) #通过父类的new传递了当前cls，写此句显然会执行父类的new
    #实例后加()执行
    def __call__(self, *args, **kwargs):
        print("Goo:__call__")
goo=Goo() #Goo:__new__， Foo:__init__， Goo:__init__
goo() #Goo:__call__，相当于goo.__call__()，Goo()()