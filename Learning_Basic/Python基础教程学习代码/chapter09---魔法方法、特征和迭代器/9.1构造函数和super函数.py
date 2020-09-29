# Author:Zhang Yuan

class person:
    def __init__(self,name="ZhangYuan"):
        self.country="China"
        self.name=name
    def show(self,code="123456789"):
        print("this person's name is {},he comes from {}".format(self.name,self.country))
        print("This is person,code: "+code)
p=person("MuSa")
p.show()
print("--------------------------------------------")

# class brother(person):
#     def __init__(self):
#         person.__init__(self)
#         self.sex="man"
#     def show1(self):
#         person.show(self,"ABCDEF")
#         print("This is my brother.he is %s"%(self.sex))
# b=brother()
# b.show1()

class brother(person):
    def __init__(self):
        super().__init__("ZYMuSa")
        self.sex="man"
    def show1(self):
        super().show("abcdefg")
        print("This is my brother.he is %s"%(self.sex))
b=brother()
b.show1()




