# Author:Zhang Yuan

list_1=[3,2,1,4,5,7,6,8,8,9,1,4,5,9]
list_1=set(list_1) #设置为集合，得到元素集
print(list_1,type(list_1))
list_2=set([2,6,1,4,66,97,88])

print(list_1,list_2)

#交集
print(list_1.intersection(list_2))
#并集
print(list_1.union((list_2)))
#差集
print(list_1.difference(list_2))
print(list_2.difference(list_1))
#是否为子集\父集
list_3=set([1,3,5])
print(list_3.issubset(list_1))   #是否为子集
print(list_1.issuperset(list_3)) #是否为父集
#对称差集:两个集合并后取出交的
print(list_1.symmetric_difference(list_2))



