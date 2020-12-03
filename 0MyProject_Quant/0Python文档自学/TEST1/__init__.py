# Author:Zhang Yuan
#遇到 from package import * 时应该导入的模块名列表
__all__=["module_test1","module_test2"]

#请注意，相对导入可以导入中涉及的兄弟模块和父包
#如果有下面语句，作为模块导入其他，正常运行。由于主模块的名称总是 "__main__"，作为主模块会运行错误，因此用作Python应用程序主模块的模块必须始终使用绝对导入。
# from .import module_test1,module_test2

