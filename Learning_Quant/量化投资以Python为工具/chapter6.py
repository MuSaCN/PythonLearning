# Author:Zhang Yuan
#1
def one(num):
    if(type(num)==int):
        if(num%2==1):return num
        else:return 0
    else:
        print("ERROR,Please input an integer.")
print(one(1.2))

#2
import random
l=[random.normalvariate(0,1) for i in range(5)]
l1=[]
for i in l:
    if i>=0:l1.append("Big")
    else:l1.append("Small")
print(l,"\n",l1)

#3
L=[]
for i in range(5):
    ii=[0 for j in range(5)]
    ii[i]=1
    L.append(ii)
print(L)

#4
a=(-1,0,1,2,39)
b=(1,2,3,4)
for i in a:
    for j in b:
        if(i==j):
            print(i,j)
for i in (-1,0,1,2,39):
    if(i in range(1,5)):
        print(i)

#5
Hilbert=[]
for i in range(4):
    row=[1/(i+j+1) for j in range(4)]
    Hilbert.append((row))
print(Hilbert)

