# Author:Zhang Yuan
#1
x=[1,2,3]
def permutation(x):
    x[0],x[-1]=x[-1],x[0]
    return(x)
y=permutation(x)
print(y,x,y is x)

#2
def sum_lists(x,y):
    z=[x[i]+y[i] for i in range(len(x))]
    return z
print(sum_lists([1,4,7],[2,9,8]))
def sum_lists(x,y):
    return [x[i]+y[i] for i in range(len(x))]

#3
def sum2(*lists):
    result=[]
    for i in range(len(lists[0])):
        a=0
        for j in range(len(lists)):
           a+=lists[j][i]
        result.append(a)
    return result
print(sum2([1,2,3],[4,5,6],[7,8,9]))
print(sum2([1,2,3],[4,5,6]))
print(sum2([1,2,3]))

def sum22(*lists):
    if len(lists)==1:
        return lists
    elif len(lists)==2:
        return sum_lists(lists[0],lists[1])
    else:
        return sum_lists(lists[0],sum22(*lists[1:]))
print(sum22([1,2,3],[4,5,6],[7,8,9]))
print(sum22([1,2,3],[4,5,6]))
print(sum22([1,2,3]))

#4
def fibo(n):
    if n==0:
        return 0
    if n==1:
        return 1
    return (fibo(n-1)+fibo(n-2))
def setfibo(n):
    l=[]
    for i in range(n):
        l.append(fibo(i))
    return l
print(setfibo(20))

#5
def arrayabs(*numbers):
    return list(map(abs,numbers))
print(arrayabs(10,-1,-2,-3))

#6
risk=[0.1,-0.2,0.3,-0.4]
print(sum(list(map(lambda x:x>0,risk))))





