# Author:Zhang Yuan
#字符串转成列表
var=list("12345")
print(var)
#列表转成字符串
print("abc_".join(var))
#切片赋值，[]相当于删除del
var[1:3]=[]
print(var)
#删除
del var[0]
print(var)
