# Author:Zhang Yuan
from abc import ABC,abstractmethod

class Talker(ABC):
    @abstractmethod
    def talk(self):
        pass

class A(Talker):
    pass

class B(Talker):
    def talk(self):
        print("123")

b=B()
b.talk()
print(isinstance(b,B))



