# Author:Zhang Yuan
#只摘取name变量
from module_1 import name
print(name)

#不需要前缀，本质是摘取代码，不推荐
from module_1 import *
sayhello()
print(name)

#相当于module_1的所有代码赋值给了对象变量module_1
import module_1
module_1.sayhello()
print(module_1.name)

#以这种方式相当于更改别名
from module_1 import logger as logger_ZhangYuan
logger_ZhangYuan()

