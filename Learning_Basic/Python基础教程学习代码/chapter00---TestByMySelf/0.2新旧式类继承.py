# Author:Zhang Yuan
#除非万不得已，否则应该避免用多重继承。
#class base: #经典类
class base(object):  #新式类
    def __init__(self,name):
        print("base:",name)
    def show(self):
        print("base show")
class inherited(base):
    def __init__(self,name,age):
        #base.__init__(self,name)  #经典类写法
        super(inherited,self).__init__(name)     #新式类写法
        print("inherited:",name,age)
    def inshow(self):
        print("Inherited show")
CI=inherited("ZhangYuan",34)

#super是按照广度优先的顺序来寻找父类，父类不一定一定是父类。super是根据拓扑结构整体来运算寻找的
#一般如果新式类要使用super，父类都要加上super指向，这样防止错误。
class A(object):
    def __init__(self):
        print("A")
class B(A):
    def __init__(self):
        print("B")
        super(B,self).__init__()
    pass
class C(A):
    def __init__(self):
        print("C")
        super(C,self).__init__()
    pass
class F(A):
    def __init__(self):
        print("F")
        super(F,self).__init__()
    def showF(self):
        print("show F")
class G(F):
    def __init__(self):
        print("G")
        super(G,self).__init__()
    pass
class D(G,B,C):
    def __init__(self):
        print("D")
        super(D,self).__init__()
testD=D()
testG=G()
print("----------------super功能挖掘------------------")
#super(某类-->指向某类的下一个super,类实例-->实例决定了广度优先顺序的拓扑结构)
#在多重继承中，super指向的是拓扑结构上顺序，不一定是父类
super(G,testG).showF() #testG实例中，类G的super指向F
super(G,testD).showF() #testD实例中，类G的super指向F
super(B,testD).__init__() #testD实例中，类B的super指向C，再指向A
print("----------------------------------")
#旧式写法:根据深度优先（根据代码安排）
class A_Old(object):
    def __init__(self):
        print("A_Old")
class B_Old(A_Old):
    def __init__(self):
        print("B_Old")
        A_Old.__init__(self)
        #super().__init__()
    pass
class C_Old(A_Old):
    def __init__(self):
        print("C_Old")
        A_Old.__init__(self)
        #super().__init__()
    pass
class D_Old(B_Old,C_Old):
    def __init__(self):
        print("D_Old")
        B_Old.__init__(self)
        C_Old.__init__(self)
test_Old=D_Old()



