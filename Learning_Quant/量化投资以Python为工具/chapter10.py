# Author:Zhang Yuan
import numpy as np

#1
#Hilbert matrix，矩阵的一种，其元素A（i,j）=1/(i+j-1)，i,j分别为其行标和列标。
Hilbert=1/(np.array([1,2,3,4])+np.array([[0],[1],[2],[3]]))
Helbert2=1/(np.arange(4)+np.expand_dims(np.arange(1,5),1))

#2
import datetime as dt
datelist=[dt.datetime(2015,1,13)+dt.timedelta(i) for i in range(5)]
pricelist=[7.31,7.28,7.40,7.43,7.41]
close_dict=dict(zip(datelist,pricelist))
close_dict[dt.datetime(2015,1,20)]=7.44
datelist.append(dt.datetime(2015,1,20))
pricelist.append(7.44)
close_dict[dt.datetime(2015,1,16)]=7.5
import math
cash=10000
volumn={datelist[0]:0}
for i in range(1,len(close_dict)):
    if close_dict[datelist[i]]>close_dict[datelist[i-1]]:
        tempvolumn= math.floor( (cash*0.5) / close_dict[datelist[i]] )
        volumn[datelist[i]]=tempvolumn
        cash=cash - tempvolumn*close_dict[datelist[i]] + volumn[datelist[i-1]]*close_dict[datelist[i]]
    else:
        volumn[datelist[i]]=0
D=np.array([[t for t in volumn.keys()],[v for v in volumn.values()]])
D

#3
a=np.cos(np.linspace( 0, 2*np.pi, 1000+1, endpoint=True, dtype=float   ))

#4
b=np.array([0.5,1.43,-1.36,-0.16,0.29,-0.59,1.16,-0.33,0.07,-1.36])
np.mean(b)
np.var(b)
np.std(b)
np.median(b)
