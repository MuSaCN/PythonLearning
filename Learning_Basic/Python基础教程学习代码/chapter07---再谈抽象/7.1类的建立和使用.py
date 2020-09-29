# Author:Zhang Yuan
class bird:
    song="Default None"
    def show(self):
        print(self.song)
        print("This is test %s"%(self.song))

c=bird()
print("song=",c.song)
c.show()
c.song="Change"
print("song=",c.song)
c.show()

class bird2:
    song = "Default None"
    def set(self):
        self.song="In Change"
    def show(self):
        print(self.song)
        print("This is test %s"%(self.song))
c2=bird2()
c2.set()
c2.show()
c2.song="2123455"
c2.show()

class person:
    def set_name(self,name):
        self.name=name
    def get_name(self):
        return self.name
c3=person()
c3.set_name("gfdsagfdsag")
print(c3.get_name())
c3.name="AGJKLD"
print(c3.get_name())

class person2:
    def show1(self):
        print("123")
    def show2(self):
        self.show1()
        print("456")
c4=person2()
c4.show1()
c4.show2()
def change():
    print("OK")
    return "False"
c4.show1=change() #关联的是change()函数的结果
print(c4.show1)   #返回False
c4.show1=change   #关联的是change这个函数
c4.show1()
c4.show2()

class test:
    song="ABC"
    def show(self):
        print(self.song)
#var1指向类的运行
var1=test()
var1.show()
#var2指向类的代码
var2=test
var2.song="DEF"
print(var1.song)
#var3指向类的运行，但是类的代码已经变了
var3=test()
var3.show()
var3.song="GHI"
var3.show()
