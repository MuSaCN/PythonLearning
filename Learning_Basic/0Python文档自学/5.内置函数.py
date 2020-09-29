# Author:Zhang Yuan

#abs(x)返回一个数的绝对值。实参可以是整数或浮点数。如果实参是一个复数，返回它的模。
print(abs(-12.3)) #12.3

#all(iterable)如果 iterable 的所有元素为真（或迭代器为空），返回 True 。
print(all(range(2,10))) #True

#any(iterable)如果*iterable*的任一元素为真则返回``True``。
print(any(range(10))) #True

#ascii(object)返回一个对象可打印的字符串，但是 repr() 返回的字符串中非 ASCII 编码的字符，会使用 \x、\u 和 \U 来转义。
print(ascii("阿斯蒂芬")) #'\u963f\u65af\u8482\u82ac'

#bin(x)将一个整数转变为一个前缀为“0b”的二进制字符串。
print(bin(-123)) #-0b1111011

#class bool([x])返回一个布尔值，True 或者 False。
print(bool("")) #False

#breakpoint(*args, **kws)此函数会在调用时将你陷入调试器中。就是设置断点。具体来说，它调用 sys.breakpointhook() ，直接传递 args 和 kws 。
# def divide(divisor, dividend):
#     breakpoint()     #出现异常后在（Pdb）后输入args,可以打印变量的具体值
#     return dividend / divisor
# divide(1,4000)

#class bytearray([source[, encoding[, errors]]])返回一个 bytes 数组。 bytearray 类是一个可变序列，包含范围为 0 <= x < 256 的整数。
print(bytearray("aabbcc",encoding='utf-8')) #bytearray(b'aabbcc')

#class bytes([source[, encoding[, errors]]])返回一个“bytes”对象， 是一个不可变序列，包含范围为 0 <= x < 256 的整数。
print(bytes(range(10))) #b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t'

#callable(object) 检查对象object是否可调用。如果实参 object 是可调用的，返回 True，否则返回 False。如果返回真，调用仍可能会失败；但如果返回假，则调用 object 肯定会失败。注意类是可调用的（调用类会返回一个新的实例）。如果实例的类有 __call__() 方法，则它是可调用。
print(callable(int)) #True

#chr(i)返回 Unicode 码位为整数 i 的字符的字符串格式。这是 ord() 的逆函数。
print(chr(97)) #a

#@classmethod 把一个方法封装成类方法。第一个实参为cls。类方法的调用可以在类上进行 (例如 C.f()) 也可以在实际上进行 (例如 C().f())。
class C:
    @classmethod
    def f(cls):
        print("CLS")
C.f() #CLS

#compile(source, filename, mode, flags=0, dont_inherit=False, optimize=-1)将 source 编译成代码或 AST 对象。代码对象可以被 exec() 或 eval() 执行。
str = "for i in range(0,10): print(i)"
c = compile(str,'','exec')   # 编译为字节代码对象
print(c) #<code object <module> at 0x000001952B53A270, file "", line 1>
exec(c)

#class complex([real[, imag]])返回值为 real + imag*1j 的复数，或将数字类字符串或数字转换为复数。
print(complex(1,5),complex("2+4j"))

#delattr(object, name) setattr() 相关的函数。函数用于删除属性。delattr(x, 'foobar') 相等于 del x.foobar
class Ctest:
    x = 10
    y = -5
delattr(Ctest,"y")
#print(Ctest.y) #error

#class dict(**kwarg)  class dict(mapping, **kwarg)  class dict(iterable, **kwarg)创建一个新的字典。
print(dict(a='a', b='b', t='t'))

#dir([object])如果没有实参，则返回当前本地作用域中的名称列表。如果有实参，它会尝试返回该对象的有效属性列表。
print(dir(Ctest))

#divmod(a, b)它将两个（非复数）数字作为实参，并在执行整数除法时返回一对商和余数。
print(divmod(10,3)) #(3, 1)

#enumerate(iterable, start=0)返回一个枚举对象。iterable 必须是一个序列，或 iterator，或其他支持迭代的对象。
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
print(list(enumerate(seasons))) #[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]

#eval(expression, globals=None, locals=None) 用来执行一个字符串表达式，并返回表达式的值。实参是一个字符串，以及可选的 globals 和 locals。globals 实参必须是一个字典。locals 可以是任何映射对象。
#只能是单个表达式（注意eval不支持任何形式的赋值操作），而不能是复杂的代码逻辑。
x=7
print(eval( '3 * x' ))

#exec(object[, globals[, locals]])这个函数支持动态执行 Python 代码。object 必须是字符串或者代码对象。该函数返回值是 None 。也就是说exec可以执行复杂的python代码。
expr = """
y,z = 20,30
sum = x + y + z   #一大包代码
print(sum)
"""
exec(expr) #57  eval(expr)就会出错

#filter(function, iterable)用 iterable 中函数 function 返回真的那些元素，构建一个新的迭代器。
def func(x):
    return x>10
print(list(filter(func,[1,5,7,10,22,55,66]))) #当 function 不是 None 的时候为 (item for item in iterable if function(item))；
print(list(filter(None,[-1,0,1]))) #function 是 None 的时候为 (item for item in iterable if item) 。

#class float([x])返回从数字或字符串 x 生成的浮点数。
print(float('1e-003'))

#format(value[, format_spec])将 value 转换为 format_spec 控制的“格式化”表示。
print("{A} is {B},and {C} is {D}".format(**{"A":"a","B":"b","C":"c","D":"d"}))

#frozenset() 返回一个冻结的集合，冻结后集合不能再添加或删除任何元素。
print(frozenset('aaabbbdddeeeadsfsaabc') )

#getattr(object, name[, default])返回对象命名属性的值。
class Test(object):
    a=1
print(getattr(Test,"a")) #1 print(getattr(Test,"b")) 出错
print(getattr(Test,"c",3)) #3 属性 c 不存在，但设置了默认值有返回。注意不是添加属性c到Test

#globals()返回表示当前全局符号表的字典。
print(globals())

#hasattr() 函数用于判断对象是否包含对应的属性。
print(hasattr(Test,"a")) #True

#hash() 用于获取取一个对象（字符串或者数值等）的哈希值。
print(hash("ZhangYuan"))

#help() 函数用于查看函数或模块用途的详细说明。
print(help(Test))

#hex(x)将整数转换为以“0x”为前缀的小写十六进制字符串。
print(hex(-42))

#id() 函数用于获取对象的内存地址。
print(id(Test))

#isinstance(object, classinfo) 函数来判断一个对象是否是一个已知的类型，类似 type()。用于判断实例的类型
print(isinstance(Test(),Test))

#issubclass(class, classinfo) 方法用于判断参数 class 是否是类型参数 classinfo 的子类。用于判断类的继承
print(issubclass(Test,object))

#locals()更新并返回表示当前本地符号表的字典。
def runoob(arg):    # 两个局部变量：arg、z
    z = 1
    print (locals())
runoob("abc") #{'arg': 'abc', 'z': 1}

#map(function, iterable, ...)产生一个将 function 应用于迭代器中所有元素并返回结果的迭代器。会根据提供的函数对指定序列做映射。
print( list(map(lambda x:x**2, range(10))) )

#memoryview(obj)返回由给定实参创建的“内存视图”对象。
print(list(memoryview("ABCDEFG".encode())))

#next(iterator[, default])通过调用 iterator 的 __next__() 方法获取下一个元素。
a=iter(range(2))
print(next(a,"over")) #0
print(next(a,"over")) #1
print(next(a,"over")) #over

#ord(c)对表示单个 Unicode 字符的字符串，返回代表它 Unicode 码点的整数。这是 chr() 的逆函数。
print(ord("a")) #97

#pow(x, y[, z])返回 x 的 y 次幂；如果 z 存在，则对 z 取余（比直接 pow(x, y) % z 计算更高效）。
print(pow(10,2)) #100
print(pow(10,2,3)) #1

#class property(fget=None, fset=None, fdel=None, doc=None) 返回 property 属性
class C:
    def __init__(self):
        self._x = None
    def getx(self):
        return self._x
    def setx(self, value):
        self._x = value
    def delx(self):
        del self._x
    x = property(getx, setx, delx, "I'm the 'x' property.")
print(C.x.__doc__) #I'm the 'x' property.
c=C() #如果 c 是 C 的实例，c.x 将调用getter，c.x = value 将调用setter， del c.x 将调用deleter。
#@property
class D:
    def __init__(self):
        self._x = None
    @property
    def x(self):
        """I'm the 'x' property."""
        return self._x
    @x.setter
    def x(self, value):
        self._x = value
    @x.deleter
    def x(self):
        del self._x
d=D()
d.x=10
print(d.x)

#repr() 函数将对象转化为供解释器读取的形式。
print(repr(Test)) #<class '__main__.Test'>

#reversed(seq)返回一个反向的 iterator。
print(list(reversed(range(10))))

#round(number[, ndigits])返回 number 舍入到小数点后 ndigits 位精度的值。
print(round(2.675,2)) #2.67 不是期望的 2.68。 这不算是程序错误：这一结果是由于大多数十进制小数实际上都不能以浮点数精确地表示。

#setattr() 函数对应函数 getattr()，用于设置属性值，该属性不一定是存在的。
class TEST:
    a=1
setattr(TEST,"a",2)
setattr(TEST,"b",5)
print(TEST.a,TEST.b)

#slice() 函数实现切片对象，主要用在切片操作函数里的参数传递。
print(slice(5))
print(list(range(10)[slice(5)]))

#sorted(iterable, *, key=None, reverse=False)根据 iterable 中的项返回一个新的已排序列表。sorted() 函数对所有可迭代的对象进行排序操作。
# sort 与 sorted 区别：sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。list 的 sort 方法返回的是对已经存在的列表进行操作，而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。
array = [{"age":20,"name":"a"},{"age":25,"name":"b"},{"age":10,"name":"c"}]
array = sorted(array,key=lambda x:x["age"])
print(array)
example_list = [5, 0, 6, 1, 2, 7, 3, 4]
result_list = sorted(example_list, key=lambda x: x*-1)
print(result_list)

#@staticmethod将方法转换为静态方法。静态方法不会接收隐式的第一个参数。静态方法的调用可以在类上进行 (例如 C.f()) 也可以在实例上进行 (例如 C().f())。
#类方法与静态方法区别在于cls有指向性，在继承时会变化。
class base:
    x=7
    @classmethod
    def class_method(cls):
        print("类方法cls=",cls,"  x=",cls.x)
    @staticmethod
    def static_method():
        print("静态方法","  x=",base.x)
base.class_method() #类方法cls= <class '__main__.base'>   x= 7
base.static_method() #静态方法   x= 7
class Exbase(base):
    x=2
Exbase.class_method() #类方法cls= <class '__main__.Exbase'>   x= 2
Exbase.static_method() #静态方法   x= 7

#super([type[, object-or-type]])返回一个代理对象，它会将方法调用委托给 type 指定的父类或兄弟类。
class A:
    def add(self, x):
        y = x + 1
        print(y)
class B(A):
    def add(self, x):
        super().add(x)
B().add(10)

#vars([object])返回模块、类、实例或任何其它具有 __dict__ 属性的对象的 __dict__ 属性。
print(vars(A))

#zip(*iterables)创建一个聚合了来自每个可迭代对象中的元素的迭代器。
a = [1,2,3]
b = [4,5,6,7,8]
print(list(zip(a,b)))





