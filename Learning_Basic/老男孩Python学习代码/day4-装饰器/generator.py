# Author:Zhang Yuan

a=[1,2,3]
b=[i*2 for i in range(10)]
c=[]
for i in range(10):
    c.append(i*2)
print(a)
print(b)
print(c)

#生成器，生成算法，不同于上面，一个一个生成不占内存
#生成器，只有在调用时才会生成相应的数据
#生成器，只记住当前的位置
b=(i*2 for i in range(10))
print(b)
for i in b:
    print(i)
    if(i>10):
        break
#访问b的下一个，无法访问前一个，生成器只记住当前的位置
b.__next__()
#print(b[0])不支持列表式的访问
