# Author:Zhang Yuan
#参数arg、*args、**kwargs三个参数的位置必须是一定的，必须是(arg,*args,**kwargs)这个顺序
def func_new(arg1,arg2,arg3,*args,**kwargs):
    print("arg1=%s"%arg1)
    print("arg2=%s" % arg2)
    print("arg3=%s" % arg3)
    print("args:",args)
    print("kwargs:",kwargs)
func_new(4,5,43,1,2,3,index1="ABC",index2="DEF")

#匿名函数
#calc为变量,lambda没有起名字
calc= lambda x:x*3
print(calc(3))