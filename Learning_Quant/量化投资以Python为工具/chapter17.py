# Author:Zhang Yuan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels as sm
from scipy import stats
import MyPackage

__mypath__ = MyPackage.MyClass_Path.MyClass_Path("\\量化投资以Python为工具")  #路径类
myfile = MyPackage.MyClass_File.MyClass_File()            #文件操作类
myplt = MyPackage.MyClass_Plot.MyClass_Plot()             #直接绘图类(单个图窗)
myfig = MyPackage.MyClass_Plot.MyClass_Figure()           #对象式绘图类(可多个图窗)
mypltpro = MyPackage.MyClass_PlotPro.MyClass_PlotPro()    #Plot高级图系列
myfigpro = MyPackage.MyClass_PlotPro.MyClass_FigurePro()  #Figure高级图系列
mynp = MyPackage.MyClass_Array.MyClass_NumPy()            #多维数组类(整合Numpy)
mypd = MyPackage.MyClass_Array.MyClass_Pandas()           #矩阵数组类(整合Pandas)
mypdpro = MyPackage.MyClass_ArrayPro.MyClass_PandasPro()  #高级矩阵数组类
mytime = MyPackage.MyClass_Time.MyClass_Time()            #时间类
myDA = MyPackage.MyClass_DataAnalysis.MyClass_DataAnalysis()  #数据分析类
#------------------------------------------------------------
Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"



#1.
import matplotlib.pyplot as plt
x = list(range(1952,2016,4))
y = (29.3,28.8,28.5,28.4,29.4,27.6,27.7,27.7,27.8,27.4,27.8,27.1,27.3,27.1,27.0,27.5)
plt.plot(x,y)
plt.show()

import statsmodels.api as sm
model=sm.OLS(y ,sm.add_constant(x)).fit()
print(model.summary())
data=pd.DataFrame({"x":x,"y":y})
data
myDA.ols("y~x",data,True)

#2.
import pandas as pd
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"
EU = pd.read_csv(Path2+'/Part2/005/EuStockMarkets.csv')

plt.plot(EU.DAX,EU.FTSE,'.')

plt.xlabel('DAX')

plt.ylabel('FTSE')
plt.show()
#3.
import statsmodels.api as sm

model = sm.OLS(EU.DAX,sm.add_constant(EU.FTSE)).fit()

print(model.summary())

plt.plot(EU.FTSE,model.fittedvalues,'-')

plt.plot(EU.FTSE,EU.DAX,'.',EU.FTSE,model.fittedvalues,'-')
plt.xlabel('FTSE')
plt.ylabel('DAX')
plt.show()
#4.
plt.plot(model.fittedvalues,model.resid,'.')

plt.xlabel('Fitted')

plt.ylabel('Residual')
plt.show()

import scipy.stats as stats

sm.qqplot(model.resid_pearson,stats.norm,line='45')
plt.show()

plt.plot(model.fittedvalues,model.resid_pearson**0.5,'.')

plt.xlabel('Fitted')

plt.ylabel('Square Root of Standardized Residual')
plt.show()

#5
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

x = [20,25,30,35,40,50,60,65,70,75,80,90]

y = [1.81,1.7,1.65,1.55,1.48,1.4,1.3,1.26,1.24,1.21,1.2,1.18]

myfig.plot_scatter(x,y)
plt.plot(x,y,'.')
plt.show()

independent = np.array([x,[i**2 for i in x]]).T

model = sm.OLS(y,sm.add_constant(independent)).fit()

print(model.summary())
model.param
model.predict(np.array([1,95,95**2]).T)
model.predict([   1, 0, 10])

#6.
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"
cps = pd.read_csv(Path2+'/Part2/005/CPS1988.csv')

cps.head()

model = smf.ols('np.log(wage)~experience+education+ethnicity-1', data=cps).fit()
model = smf.ols('wage~experience', data=cps).fit()

print(model.summary())

model.fittedvalues.head()

model2 = smf.ols('np.log(wage)~experience+np.power(experience,2)+education+ethnicity',data=cps).fit()

print(model2.summary())

















