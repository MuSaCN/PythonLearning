# Author:Zhang Yuan

class Rectangle:
    def __init__(self):
        self.width=0
        self.height=0
    def set_size(self,size):
        self.width,self.height=size
    def get_size(self):
        return self.width,self.height
    psize=property(get_size,set_size)

cR=Rectangle()
cR.set_size((100,50))
print(cR.get_size())
cR.psize=(200,100)
print(cR.get_size())
print(cR.psize)

class MyClass:
    @staticmethod #smeth=staticmethod(smeth)
    def smeth(content):
        print("This is a static method",content)
    @classmethod
    def cmeth(cls,content):
        print("This is a class methon of",cls,content)
MyClass.smeth("ABC")
MyClass.cmeth("DEF")

class NewRectangle:
    def __init__(self):
        self.width=0
        self.height=0
    def __setattr__(self, key, value):
        if key=="size":
            self.width,self.height=value
        else:
            self.__dict__[key]=value
    def __getattr__(self, item):
        if item=="size":
            return self.width,self.height
        else:
            raise AttributeError()
a=NewRectangle()
a.__setattr__("test",(123,321))
print(a.__dict__)
a.__setattr__("size",(14,12))
print(a.__dict__)
print(a.__getattr__("size"))

print("-----------------------------------------------------")
#传统写法
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

#装饰器写法@property
class C:
    def __init__(self):
        self._x = 456
    @property #属性化当前函数,getx相当于变成了属性，所以访问不能加括号
    def getx(self):
        return self._x
    @getx.setter #getx属性进行设置
    def getx(self, value):
        self._x = value
    @getx.deleter #getx属性进行设置
    def getx(self):
        del self._x
c=C()
print(c.getx) #操作不能加()，因为变成了属性
c.getx=123456789
print(c.getx)
del c.getx
#print(c.getx)
