# Author:Zhang Yuan
import numpy as np
import pandas as pd

#1
list1=[i for i in range(21) if  i%2==1]
nd1=np.array(list1)

#2
tuple1=tuple([i for i in range(41) if i%3==0])
nd2=np.array(tuple1)

#3
nd3=np.array([i for i in nd1 if i in nd2])
nd3=nd1[np.in1d(nd1,nd2)]
print(nd3[:round(len(nd3)/2)])
for i in range(round(len(nd3)/2)):
    print(nd3[i])

#4
nd4=np.random.uniform(0,10,10)

#5
nd5=np.array([i for i in np.arange(21) if i%2==0])
nd5=np.arange(0,22,2)

#6
nd5[-5:]
nd5[len(nd5)-5:]

#7
dict={"id":["a","b","c","d","e","f"],
      "name":["Alice","Bob","Charlie","David","Esther","Fanny"],
      "age":[34,36,30,29,32,36]}
dp7=pd.DataFrame(dict)
dp7.T.iloc[2]

#8
dp8=dp7.append(pd.DataFrame({"id":["g"],"name":["John"],"age":[19]}),True)
dp8.index=dp8.age
dp8.drop(30)
