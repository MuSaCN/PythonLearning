# Author:Zhang Yuan

names = 'A1 A2 A3 A4'
print(names)

names=['A1','A2','A3','A4']
print(names[0],names[2])
print(names[1]+' '+names[2])
#切片(顾头不顾尾)
print(names[1:3])
print(names[0:2])
print(names[3:])  #取后面的
print(names[-2:])

names.append("A5") #追加一个到最后
print(names)
names.insert(1,"A1.5") #插入到指定位置
print(names)
names[1]='A1.55' #修改
print(names)

#delete
names.remove("A2")
print(names)
del names[1]
print(names)
names.pop(2)
print(names)

#查找元素
print(names.index("A5"))
print(names[names.index("A5")])

#列表有相同元素
names.append("A3") #追加一个到最后
print(names)
print(names.count("A3")) #返回列表中多少个指定元素


names.reverse() #列表翻转
print(names)

names.sort() #排序列表

names2=[1,2,3,4]
names.extend(names2) #合并列表name2到name1末尾
#del names2 #删除列表name2

#列表中可以再追加列表，追加列表本质是追加指针
names.append(["B1","B2"])

names2 = names.copy() #普通copy只能复制第一层
import copy
names3=copy.deepcopy(names) #深度copy相当于重新建立内存

names[-1].append("B3") #指针指向的列表追加的数据，所以下面names,names2两个变量都显示变化
print(names,names2,names3)

#不同于c++数组，python内存和变量的关系与c++不同！！！
a=[1,2,3] #分配内存，a指向数据
b=a       #b也指向a指向的数据
c=a.copy()#复制，新增加内存储存数据
a[-1]=66  #通过a修改内存数据
print(a,b,c)  #所以b变化,c不变
a[0]=66   #通过a修改内存数据
print(a,b,c)  #所以b变化,c不变
a=[1,2,3] #a重新赋值，相当于重新分配内存，但是b指向的内存依然不变
print(a,b,c)  #所以b不变,c不变

#列表的有步长的切片
print(names,names[0::2]) #从开始到末尾，包括末尾
print(names,names[::2])  #0可以省略

#列表的循环
for i in names:
    print(i)


#清空列表
#names.clear()
#print(names)