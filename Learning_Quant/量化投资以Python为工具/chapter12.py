# Author:Zhang Yuan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MyPackage.MyClass_Path import MyClass_Path
__Class_Path__=MyClass_Path("\\量化投资以Python为工具")
from MyPackage.MyClass_Plot import MyClass_Figure
datapath="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答\\data1\\Part1\\012"

#1
datapath1=datapath+"\\Money.csv"
Canada=pd.read_csv(datapath1,index_col="date")
money=Canada.m
gdp=Canada.y
Class_Figure=MyClass_Figure()
Class_Figure.PlotLine2D(0,money,cla=True,show=True,grid="x",PlotLabel=["加拿大货币量","时间","货币量"])
Class_Figure.PlotLine2D(0,gdp,cla=False,twinXY="X",color="red",PlotLabel=["12345679","tiadsfadsfme","dsafdsagpd"])
#---
Money = pd.read_csv(datapath1,index_col='date')
axis1 = plt.subplot()
axis1.plot(Money.m)
plt.title('Money Supply of Canada')
plt.xlabel('Year')
axis2 = axis1.twinx()
axis2.plot(Money.y)
axis3 = axis1.twinx()
axis3.plot(Money.r)
plt.show()

#2
datapath2=datapath+"\\Journals.csv"
JourData=pd.read_csv(datapath2)
Class_Figure1=MyClass_Figure()
Class_Figure1.plot_scatter(0,JourData.citestot,JourData.libprice,"Scatter",PlotLabel=["JourData","citestot","libprice"])
# ---
Journals = pd.read_csv(datapath2)
plt.scatter(Journals.citestot,Journals.libprice)
plt.title('Price vs Citations')
plt.xlabel('Citations')
plt.ylabel('Price')
plt.show()

#3
datapath3=datapath+"\\mtcars.csv"
cars=pd.read_csv(datapath3)

# ---
mtcars = pd.read_csv(datapath3)
types=np.array([0.2,0.6])
number = mtcars.groupby(['gear','vs'])['vs'].agg(len)
plt.bar(types,number.ix[3],width=0.3,label='gear=3')
plt.show()
plt.bar(types,number.ix[4],width=0.3,bottom=number.ix[3],color='r',label='gear=4')
plt.show()
plt.bar(types,number.ix[5],width=0.3,bottom=number.ix[3]+number.ix[4],color='y',label='gear=5')
plt.show()
plt.xlim([0,1])
plt.ylim([0,25])
plt.legend()
plt.xticks(types+0.3/2,[0,1])
plt.show()


#4
datapath4=datapath+"\\Arthritis.csv"
data=pd.read_csv(datapath4)
age=data.Age
Class_Figure4=MyClass_Figure()
Class_Figure4.PlotFreHistogram(0,age)
# ---
Arthritis = pd.read_csv(datapath4)
plt.hist(Arthritis.Age)
plt.xlabel('Age')
plt.title('Histogram of Age')
plt.show()

#5
norm1 = np.random.normal(4,1,100)
norm2 = np.random.normal(4,2,100)
norm3 = np.random.normal(4,3,100)
norm4 = np.random.normal(4,4,100)
Class_Figure5=MyClass_Figure()
Class_Figure5.plot_box(0,[norm1,norm2,norm3,norm4],["std=1","std=2","std=3","std=4"],PlotLabel=["Normal Distributions with Different Standard Deviation","Standard Deviation",""])
# ---
plt.boxplot([norm1,norm2,norm3,norm4])
plt.xlabel('Standard Deviation')
plt.title('Normal Distributions with Different Standard Deviation')
plt.show()

#6
norm1 = np.random.normal(4,1,100)
norm2 = np.random.normal(4,2,100)
norm3 = np.random.normal(4,3,100)
norm4 = np.random.normal(4,4,100)
Class_Figure6=MyClass_Figure([221,222,223,224])
Class_Figure6.plot_scatter(0,range(100),norm1,show=False)
Class_Figure6.plot_scatter(1,range(100),norm2,show=False)
Class_Figure6.plot_scatter(2,range(100),norm3,show=False)
Class_Figure6.plot_scatter(3,range(100),norm4,show=True)

#---
figure,axes = plt.subplots(2,2)
axes[0,0].scatter(range(100),norm1)
axes[0,1].scatter(range(100),norm2)
axes[1,0].scatter(range(100),norm3)
axes[1,1].scatter(range(100),norm4)
plt.show()

#7

#---
tan_value = np.tan(np.linspace(0,2*np.pi,10001))[np.newaxis,:]
tan_value = np.concatenate((np.linspace(0,2*np.pi,10001)[np.newaxis,:],
                            tan_value),0)
tan_value=np.where(tan_value>200,200,
                   np.where(tan_value<-200,-200,tan_value))
plt.plot(tan_value[0],tan_value[1],'o')
plt.show()
