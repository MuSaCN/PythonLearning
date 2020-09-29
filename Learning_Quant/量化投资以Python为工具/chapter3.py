# -*- coding: utf-8 -*-
#1
type((1,3,5,7))
type("pi")
type(5>6)
type({"high":7.45,"low":7.30})

#2
print((3+4j)*(4+3j))
print(3/4)
print(True*3)
print(0.003-0.0022222)

#3
a=[1,2,3]
b=a
print(b)
print(id(a)==id(b),"id")
b[0]=3
print(a)
print(id(a)==id(b),"id2")

#4
a=tuple(range(1,11))
print(a)
b=tuple(i for i in range(21) if i%2==1)
b1=tuple(i+1 for i in range(20) if (i+1)%2==1)
print(b,b1)
c=[i for i in range(51) if i%5==0]
print(c)
d=list(range(1,6))*3
d1=[]
for i in range(1,6):
    j=0
    while(j<3):
        d1.append(i)
        j+=1
print(d,d1)
e=set(["NASDAQ","Dowjones","DAX","FTSE"])
e1=set({"NASDAQ","Dowjones","DAX","FTSE"})
print(e,e1)

#5
print(tuple("abc"))
print(list("abc"))
print(set("abc"))

#6
dict={}
dict["A"]=ord("A")
dict["b"]=ord("b")
dict["\n"]=ord("\n")
print(dict)
