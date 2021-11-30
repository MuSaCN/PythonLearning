# Author:Zhang Yuan

#必须要有 __init__.py 文件才能让 Python 将包含该文件的目录当作包。
#在 __init__.py 文件设置 __all__=["..."]，遇到 from package import * 时会自动导入的模块名列表，若没有这句则不自动导入
from TEST1 import *
module_test1.AAA()
module_test2.BBB()
print(123)

#f字符串
year = 2016
event = 'Referendum'
print(f'Results of the {year} {event}')
import math
print(f'The value of pi is approximately {math.pi:.5f}.')
table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 7678}
for name, phone in table.items():
    print(f'{name:10} ==> {phone:10d}')

#字典传递到字符串
table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 8637678}
print('Jack: {0}; Sjoerd: {1}; '
      'Dcab: {2}'.format(table["Jack"],table["Sjoerd"],table["Dcab"]))
print('Jack: {0[Jack]}; Sjoerd: {0[Sjoerd]}; '
      'Dcab: {0[Dcab]}'.format(table))
table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 8637678}
print('Jack: {Jack:d}; Sjoerd: {Sjoerd:d}; Dcab: {Dcab:d}'.format(**table))

#在字符串输出时，{}里面有 ':' ，且后传递一个整数可以让该字段成为最小字符宽度。这在使列对齐时很有用。
for x in range(1, 11):
    print('{0:2d} {1:3d} {2:4d}'.format(x, x*x, x*x*x))

#字符串对象的 str.rjust() 方法通过在左侧填充空格来对给定宽度的字段中的字符串进行右对齐。类似的方法还有 str.ljust() 和 str.center()
for x in range(1, 11):
    print(str(x).rjust(1), str(x*x).rjust(3), end=' ')
    # Note use of 'end' on previous line
    print(repr(x*x*x).rjust(4))

#文件操作
with open('TESTFile.txt') as f:
    print(f.read())
with open('TESTFile.txt') as f:
    print(f.readline())
    print(f.readline())
with open('TESTFile.txt') as f:
    for line in f:
        print(line,end="")
with open('TESTFile.txt') as f:
    print(f.readlines())

#json序列化：把内存数据变成字符串，只能处理简单基础的结构，json用于不同语言之间进行数据交互
info={"name":"ZhangYuan","age":22}
import json
print(json.dumps(info))












