# Author:Zhang Yuan

#一定要使用 import os 而不是 from os import * 。这将避免内建的 open() 函数被 os.open() 隐式替换掉，它们的使用方式大不相同。
import os
print(os.getcwd()) # Return the current working directory

import glob
print(glob.glob('*.py'))

import sys
print(sys.argv)
#终止脚本的最直接方法是使用 sys.exit()
#sys.exit() #有这句，下面的都不运行了

import math
print(math.cos(math.pi / 4))
print(math.log(1024, 2))

from timeit import Timer
print(Timer('t=a; a=b; b=t', 'a=1; b=2').timeit())
print(Timer('a,b = b,a', 'a=1; b=2').timeit())

def average(values):
    """Computes the arithmetic mean of a list of numbers.
    >>> print(average([20, 30, 70]))
    40.0
    """
    return sum(values) / len(values)
import doctest
doctest.testmod()

from array import array
a = array('H', [4000, 10, 700, 22222])
print(sum(a))
print(a[1:3])

# bisect 模块具有用于操作排序列表的函数
import bisect
scores = [(100, 'perl'), (200, 'tcl'), (400, 'lua'), (500, 'python')]
bisect.insort(scores, (300, 'ruby'))
print(scores)

#round的四舍五入不准确，因为部分小数无法用二进制精确表示
#要准确需要自己写函数判定
print(round(2.675,2)) #2.67
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
print((0.1+0.1+0.1)==0.3) #False
a=0.1+0.1+0.1
print(a.as_integer_ratio())
from decimal import *
print((Decimal("0.1")+Decimal("0.1")+Decimal("0.1"))==Decimal("0.3")) #True


