# Author:Zhang Yuan
#Python隐藏函数是否可以被继承。其中"仅前面单下划线_：_a"可以被继承，"仅前面双下划线__：__b"不可以被继承，但是前后都加下划线"_a_, __b__"则可以被继承！
    # _a     可继承   # _a_    可继承
    # __b    不可继承  # __b_   不可继承  # __b__  可继承

#Python中用list、dict或class实例变量，可以不需要global声明，就可作为全局变量；其他需要global声明才可以。

#Python中的字符串不能被修改，它们是 immutable 的

#python中固定的不可变的对象内存地址是一样的
a=3
b=3
print(a is b,a==b,id(a),id(b)) #True True 140716530557648 140716530557648
c=(3,)
d=(3,)
print(c is d,c==d,id(c),id(d)) #True True 2623132500824 2623132500824
class A:
    pass
e=A()
f=A()
print(e is f,e==f,id(e),id(f)) #False False 2623134376288 2623134314392

#PS:列表List与List[:]值相同，但不是一个对象
words = ['cat', 'window', 'defenestrate']
print(words == words[:],words is words[:]) #True False

#迭代器在内存中结束后，则不能迭代了。
a=range(10)
b=iter(a)
for i in b:
    print(i,end=",") #
#第二次再次访问则没有结果，因为b为迭代器.两次都换成a则可以。a直接写入了内存，b在内存中迭代
for j in b:
    print(j,end=",")
print("OK")
#0,1,2,3,4,5,6,7,8,9,OK

#列表作为栈使用效率好（后进先出“last-in，first-out”）。
#PS:列表作为队列使用(“先进先出”)，单纯使用效率低。若要实现一个队列， collections.deque 被设计用于快速地从两端操作，它类似于列表，但从左端添加和弹出的速度较快，而在中间查找的速度较慢。
from collections import deque
queue=deque([1,2,"a","b","c"]) #双队列
queue.append("Terry") #右端添加
print(queue.popleft()) #最端出

#嵌套的列表推导式，一层层表述
matrix=[[i+4*j for i in range(4)] for j in range(3)]
print(matrix) #[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
matrix2=[[row[i] for row in matrix] for i in range(4)]
print(matrix2) #[[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]
#注意下面的区别：单独的列表，相当于各成员与空内容zip成元组
print(list(zip(matrix))) #[([0, 1, 2, 3],), ([4, 5, 6, 7],), ([8, 9, 10, 11],)]
#注意下面的区别：加*让列表的各个部分独立开来，然后再zip成元组
print(list(zip(*matrix))) #[(0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11)]

#花括号或 set() 函数可以用来创建集合。注意：要创建一个空集合你只能用 set() 而不能用 {}，因为后者是创建一个空字典
#类似于 列表推导式，集合也支持推导式形式
a = {x for x in 'abracadabra' if x not in 'abc'} #{'r', 'd'}

#字典推导式可以从任意的键值表达式中创建字典
d={x: x ** 2 for x in (2, 4, 6)} #{2: 4, 4: 16, 6: 36}
#字典一个Key只能对应一个值，所以下面的4会被5覆盖
D={a:b for a in "ABCDEF" for b in (4,5)} #{'A': 5, 'B': 5, 'C': 5, 'D': 5, 'E': 5, 'F': 5}

#循环的技巧
#当在字典中循环时，用 items() 方法可将关键字和对应的值同时取出
#当在序列中循环时，用 enumerate() 函数可以将索引位置和其对应的值同时取出
#当同时在两个或更多序列中循环时，可以用 zip() 函数将其内元素一一匹配。
#有时可能会想在循环时修改列表内容，一般来说改为创建一个新列表是比较简单且安全的

#布尔运算符 and 和 or 也被称为 短路 运算符：它们的参数从左至右解析，一旦可以确定结果解析就会停止。

#序列对象可以与相同类型的其他对象比较。它们使用 字典顺序 进行比较：首先比较两个序列的第一个元素，再比较每个序列的第二个元素，以此类推.
#字典顺序对字符串来说，是使用单字符的 Unicode 码的顺序
print(  (1, 2, ('aa', 'ab'))   <  (1, 2, ('abc', 'a'), 4)  )  #True

#必须要有 __init__.py 文件才能让 Python 将包含该文件的目录当作包。
#在 __init__.py 文件设置 __all__=["..."]，遇到 from package import * 时会自动导入的模块名列表，若没有这句则不自动导入。

#请注意，相对导入可以导入中涉及的兄弟模块和父包。
#如果有下面语句，作为模块导入其他，正常运行。由于主模块的名称总是 "__main__"，作为主模块会运行错误，因此用作Python应用程序主模块的模块必须始终使用绝对导入。
# from .import module_test1,module_test2

#设计到需要加载句柄，关闭句柄的操作，多用with关键字。

#异常执行：try执行，except发生异常时执行，else没有异常时执行，finally在所有情况执行。
#raise抛出异常

#global 语句可以用来指明某个特定的变量位于全局作用域并且应该在那里重新绑定； nonlocal 语句表示否定当前命名空间的作用域，寻找父函数的作用域并绑定对象。

#Python3的多重继承中，继承的类有共同的基类：按照广度优先、从左到右寻找父类和兄弟类。继承的类没有共同基类：按照深度优先、从左到右寻找不同的父类

#Python中的类或类实例，可以自定义添加变量。不同于C++类的严格封装作用。

#Python的类可以迭代器化：__iter__(),__next__()让类迭代器化
#for语句中底层会自动调用容器对象中的 iter()，产生迭代器，所以会有for循环。
#用yield写生成器使得语法更为紧凑，因为它会自动创建 __iter__() 和 __next__() 方法。
#生成器表达式语法类似列表推导式，将外层为圆括号而非方括号。这种表达式返回生成器，且可以直接被外部函数直接使用(重叠的括号可以省略)。
print( sum(i*i for i in range(10))  )

#大多数在浮点数方面都无法精确计算，因为部分小数无法用二进制精确表示
#如果要做数值相等判断时要注意，可能不能绝对精确
print((0.1+0.1+0.1)==0.3) #False
from decimal import *
print((Decimal("0.1")+Decimal("0.1")+Decimal("0.1"))==Decimal("0.3")) #True
#round的四舍五入不准确，因为部分小数无法用二进制精确表示
#要准确需要自己写函数判定
print(round(2.675,2)) #2.67 wrong
print(round(2.685,2)) #2.69
#decimal 模块提供了一种 Decimal 数据类型用于十进制浮点运算
from decimal import *
print(round(Decimal('0.70') * Decimal('1.05'), 2)) #0.74
print(round(0.70 * 1.05, 2)) #0.73
#精确表示特性使得 Decimal 类能够执行对于二进制浮点数来说不适用的模运算和相等性检测
print(Decimal('1.00') % Decimal('.10')) #0.00
print(1.00 % 0.10) #0.09999999999999995
print(sum([Decimal('0.1')]*10) == Decimal('1.0')) #True
print(sum([0.1]*10) == 1.0) #False

#Python3中下面定义为False：
# •被定义为假值的常量: None 和 False。
# •任何数值类型的零: 0, 0.0, 0j, Decimal(0), Fraction(0, 1)
# •空的序列和多项集: '', (), [], {}, set(), range(0)

#Python3中，让代码可控制的执行，可以把代码字符串化。然后用exec()、eval()、compile()控制执行。

#Python3类的进化：
#类方法：@classmethod 把一个方法装饰成类方法。可以实现类中函数的直接调用。必须有隐式参数cls
#静态方法：@staticmethod 把一个方法装饰成静态方法。也可以实现类中函数的直接调用。不需要隐式参数。与@classmethod不同处有，cls在继承时指向类会变化。
#类迭代器化：__iter__(),__next__()让类迭代器化
#函数方法属性化：property与@property 让类中设置的函数方法属性化(变量化)，让实例可以直接以变量的形式操作。
#类的构造函数：__init__(self)；析构函数：__del__(self)
#类的实例序列化：__init__(self)这个方法通常要设置字典为数据类型；__setitem__(self, key, value): 让实例可直接设置:A[key]=...；__getitem__(self, item): 让实例可直接访问:A[item]；__delitem__(self, key): 让实例可直接运行del A[key]；__len__(self): 让实例可直接运行len(A)




