# Author:Zhang Yuan

import AAA

#模块目录的查找和打印-------------------------------------------------------
import sys,pprint
#查目录
ModulePath=sys.path
#可以逐行打印
pprint.pprint(ModulePath)


#模块的导入------------------------------------------------------------------
#添加模块目录，这种做法不常用
ModulePath.append("C:\\Users\\i2011\\PycharmProjects\\Python基础教程学习代码\\chapter10---模块和标准库")
# #添加后可以导入模块
# import Module_Test

# #也可以把模块放入"lib\\site-packages\\"下
# import test_hello

#导入目录包，目录下必须要有__init__.py文件
#导入site-packages下的包，不import其他py文件
import MyPackage #仅默认导入__init__.py
import MyPackage.test_hello
#导入添加的目录的包
import Package_Test
import Package_Test.Module_Test

#模块重新导入
import importlib
MyPackage=importlib.reload(MyPackage)
MyPackage.test_hello=importlib.reload(MyPackage.test_hello)
Package_Test=importlib.reload(Package_Test)


#模块的探索---------------------------------------------------------------------
import copy
print(dir(copy)) #包含隐藏的内容
a=[n for n in dir(copy) if not n.startswith("_")] #这样将过滤到_开头的隐藏内容
print(a)
#模块中__all__，指定的内容
#在模块中设置__all__，在from copy import* 语句中导入__all__中指定内容，否则导入所有不以_打头的全局名称
print(copy.__all__)

#查看模块的帮助
help(copy.copy)

#查看函数的内置文档
print(range.__doc__)

#查看模块的位置
print(copy.__file__)




