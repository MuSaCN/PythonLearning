# Author:Zhang Yuan
#设置py文件编码为GBK，不是内部代码为GBK
#-*- coding:GBK -*-
s="你好"
print(s)

print(s.encode("GBK"))
print(s.encode("utf-8"))
print(s.encode("utf-8").decode("utf-8"))
print(s.encode("utf-8").decode("utf-8").encode("gb2312"))
print(s.encode("utf-8").decode("utf-8").encode("gb2312").decode("gb2312"))




