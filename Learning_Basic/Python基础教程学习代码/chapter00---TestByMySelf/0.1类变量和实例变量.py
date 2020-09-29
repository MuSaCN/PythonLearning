# Author:Zhang Yuan

class TEST:
    a=123 #类变量
    CommonList=[]
    def __init__(self):
        TEST.var="ABC" #这表示，创建实例时，增加类变量TEST.var="ABC"
        TEST.a+=10
        self.mylist = []

#print(TEST.a,TEST.var) #无法执行，因为没有增加进去

A=TEST()
B=TEST()
print(TEST.a,A.a,B.a)
print(TEST.var,A.var,B.var)

TEST.var="abcdef"
print(TEST.var,A.var,B.var)

A.a=A.a+10 #单变量在赋值时，等于自定义语句
print(A.a,B.a,TEST.a)

#类中增加类变量
TEST.b="ADD"
print(TEST.b,A.b,B.b)

#实例a中增加self.a变量
A.a=456
print(TEST.a,A.a,B.a) #实例中，先索引实例变量，再类变量

#删除实例a中的self.a变量，但是依然有类变量a
del A.a
print(TEST.a,A.a,B.a)

#mylist操作,实例操作
A.mylist.append("MyA")
B.mylist.append("MyB")
print(A.mylist,B.mylist)

#CommonList操作,直接操作。实例中并没有增加，而是对类列表的操作
A.CommonList.append("add a")
B.CommonList.append("add b")
TEST.CommonList.append("add TEST")
print(A.CommonList,B.CommonList,TEST.CommonList)

#实例中增加列表
A.NewList=[]
A.NewList.append("NewListA")
print(A.NewList)

#实例中增加列表且与类列表同名
A.CommonList=[]
print(A.CommonList,B.CommonList,TEST.CommonList)
del A.CommonList
print(A.CommonList,B.CommonList,TEST.CommonList)

def foo(x):return x*x
foo=lambda x:x*x

