__author__ = "Alex Li"

import os
# os.system()
# os.mkdir()

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

