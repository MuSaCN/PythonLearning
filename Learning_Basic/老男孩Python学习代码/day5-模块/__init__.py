## Author:Zhang Yuan
'''
1.定义：
模块：*.py
包：本质就是一个目录（必须带有一个__init__.py）
000
2.导入方法
import module_1
import module_1,module_2
from module_1 import * #不推荐，因为不要前缀，本质是摘取代码
from module_1 import name,logger #只摘取name,logger()
from module_1 import logger as logger_ZhangYuan #相当于更改别名

3.import本质（路径搜索和搜索路径）
导入模块的本质就是把python文件解释一遍
#导入包，本质是import该目录下__init__.py，不import其他py文件

4.导入优化
from modulue_test import test #锁定了test()函数，不需要在代码中再次检索

5.模块的分类：
a：标准库；b：开源模块；c：自定义模块



'''