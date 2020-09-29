# Author:Zhang Yuan
#列表推导生成
CreateList=[i*i for i in range(10) if i%3==0]
print(CreateList)

CreateList2=["{0}+{1}".format(i,j) for i in range(10) for j in range(10) if i%3==0 and j%2==0]
print(CreateList2)

###名称匹配
boyname=["A1","A11","B1","C1","D1"]
girlname=["A2","A22","C2","C22","E2","F2","G2"]
#girlname按字典分类
girldic={}
for i in girlname:
    girldic.setdefault(i[0],[]).append(i)
print(girldic)
#生成列表
boygirl=[b+"+"+g for b in boyname if (b[0] in girldic) for g in girldic[b[0]]]
print(boygirl)

#字典推导生成
CreateDict={i:"%s squared is %s"%(i,i*i) for i in range(10) if i%2==0}
print(CreateDict)