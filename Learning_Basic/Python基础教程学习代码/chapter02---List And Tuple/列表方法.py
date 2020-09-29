# Author:Zhang Yuan

a=[1,2,3]
a.append("abc")
#lst.clear()
b=a#建立指针
a[1]=5
print(a,b)
# b=a.copy()#建立内存
# a[2]=11
# print(a,b)

x=["12","abc","asfdf","ZhangYuan"]
x.sort(key=len,reverse=True)
print(x)
