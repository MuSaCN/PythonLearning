# Author:Zhang Yuan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels as sm
import MyPackage

__mypath__ = MyPackage.MyClass_Path.MyClass_Path("\\量化投资以Python为工具")  #路径类
myfile = MyPackage.MyClass_File.MyClass_File()  #文件操作类
myplt = MyPackage.MyClass_Plot.MyClass_Plot()  #直接绘图类(单个图窗)
myfig = MyPackage.MyClass_Plot.MyClass_Figure()  #对象式绘图类(可多个图窗)
mypltpro = MyPackage.MyClass_PlotPro.MyClass_PlotPro() #Plot高级图系列
myfigpro = MyPackage.MyClass_PlotPro.MyClass_FigurePro()  #Figure高级图系列
mynp = MyPackage.MyClass_Array.MyClass_NumPy()  #多维数组类(整合Numpy)
mypd = MyPackage.MyClass_Array.MyClass_Pandas()  #矩阵数组类(整合Pandas)
mypdpro = MyPackage.MyClass_ArrayPro.MyClass_PandasPro()  #高级矩阵数组类
mytime = MyPackage.MyClass_Time.MyClass_Time()  #时间类
myDA = MyPackage.MyClass_DataAnalysis.MyClass_DataAnalysis()  #数据分析类
#------------------------------------------------------------

Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"


#1
Bwages=myfile.read_pd(Path2+"\\Part2\\002\\Bwages.csv")
Bwages
myfigpro.hist_density(Bwages["wage"],50)
mypltpro.density_cumsun(Bwages["wage"],1000)

#2
history = myfile.read_pd(Path2+'/Part2/001/history.csv',index_col="Date",parse_dates=True)
EMarket=history["Emerging.Markets"]
1-myDA.r_binom_Prob(EMarket,6,12,True)

#3
myfig.reset_figure_axes()
myfig.prob_norm(0,1,200,True,show=False)
myfig.prob_norm(0,0.5**0.5,200,True,show=False)
myfig.prob_norm(0,2**0.5,200,True,show=False)
myfig.prob_norm(2,1,200,True,show=True)

myfig.prob_chi(1,0,5,200,True,"1",show=False)
myfig.prob_chi(2,0,5,200,True,"2",show=False)
myfig.prob_chi(3,0,5,200,True,"3",show=False)
myfig.prob_chi(4,0,5,200,True,"4",show=True)

myfig.prob_t(1,-5,5,200,False,"1",show=False)
myfig.prob_t(2,-5,5,200,False,"2",show=False)
myfig.prob_t(3,-5,5,200,False,"3",show=False)
myfig.prob_t(4,-5,5,200,False,"4",show=True)

myfig.prob_f(1,40,0,5,200,True,"1",show=False)
myfig.prob_f(2,40,0,5,200,True,"2",show=False)
myfig.prob_f(3,40,0,5,200,True,"3",show=False)
myfig.prob_f(4,40,0,5,200,True,"4",show=True)

#4







