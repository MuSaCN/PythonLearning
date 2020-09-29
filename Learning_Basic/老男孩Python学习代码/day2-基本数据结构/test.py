# Author:Zhang Yuan
import copy
#浅copy本质上是引用，只能第一层不治，列表中的列表只复制指针

var1=['name',['a',100]]
#引用的三种方式
p1=copy.copy(var1)#string会新建立内存，但是列表只复制指针
p2=var1[:]        #string会新建立内存，但是列表只复制指针
p3=list(var1)     #string会新建立内存，但是列表只复制指针
print(p1,p2,p3)

p4=p1 #若直接用=，则建立指针

p1[0]='abc' #只修改p1，不影响p2,p3
p2[0]='def' #只修改p2，不影响p1,p3
p3[1][0]="ABC" #指向的内容修改，影响其他
print(p1,p2,p3,p4)




