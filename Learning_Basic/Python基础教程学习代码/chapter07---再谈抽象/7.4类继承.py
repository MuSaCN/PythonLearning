# Author:Zhang Yuan
class Filter:
    def init(self):
        self.box=[]
    def filter(self,sequence):
        return [x for x in sequence if x not in self.box]
f=Filter()
f.init()
print(f.filter([1,2,4,5]))

class SPAMFilter(Filter):
    def init(self):
        self.box=["SPAM"]
Sf=SPAMFilter()
Sf.init()
print(Sf.filter(["SPAM","ABC","DED","SPAM"]))

#类来源于哪里
print(issubclass(SPAMFilter,Filter))             #前是否为后的子类
print(SPAMFilter.__bases__,Filter.__bases__)     #返回基类
print(isinstance(Sf,SPAMFilter))                 #前是否为后的实例
print(Sf.__class__)                              #实例对象属于哪个类
print(type(Sf),type(SPAMFilter))                 #返回类型

#类特征查询
print(hasattr(Sf,"init"))#类中是否有**方法
print(callable(getattr(Sf,"init",None)))#类中**方法是否可以调用
setattr(Sf,"name","x+1") #设置对象属性
print(Sf.name)
print(Sf.__dict__)

