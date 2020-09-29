# Author:Zhang Yuan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置当前工作路径
from MyPackage.MyClass_Path import MyClass_Path
__Class_Path__=MyClass_Path("\\PythonLearning\\Python文档自学")

# 设置导入数据文件的位置
filepath=__Class_Path__.GetPath(2)+"\\QuantitativeInvestmentUsingPython\\数据及源代码\\part 1\\012\\ChinaBank.csv"
# 导入csv数据
import pandas as pd
ChinaBank=pd.read_csv(filepath,index_col="Date")
ChinaBank=ChinaBank.iloc[:,1:]
ChinaBank.head()
ChinaBank.index=pd.to_datetime(ChinaBank.index)
Close=ChinaBank.Close
Open= ChinaBank.Open
# 导入对象绘图类
from MyPackage.MyClass_Plot import MyClass_Figure
# 建立figure对象
Class_Figure=MyClass_Figure([231,232,233,234,235,236])
# 调试ReSetFigureAxes功能
# Class_Figure.ReSetFigureAxes([222,221,223,224])
# 调试AxesListSetting功能
# Class_Figure.AxesListSetting(0,grid="y",PlotLabel=["DEFE","X轴","Y轴"])
# Class_Figure.AxesListSetting(0,grid="both",PlotLabel=["ADSF","X轴","Y轴"])
# Class_Figure.AxesListSetting(Class_Figure.AxesList[0],grid="x",PlotLabel=["ABCD","X轴","Y轴"])

# 画图
Class_Figure.PlotLine2D(0,Close,"收盘价")
Class_Figure.plot_bar(1,[1,3,5,7],[10,20,30,40],1,"柱状图",False,True,None,True)
Class_Figure.PlotFreHistogram(2,Close,5,"直方图",False,True,[2.5,3.5],False)
Class_Figure.PlotFreHistogram(3,Close,5,"直方图",False,True,[2.5,3.5],True)
Close.describe()
a=[0,0,0,0]
for i in Close:
    if (i>2)&(i<=3):a[0]+=1
    elif (i>3)&(i<=4):a[1]+=1
    elif (i>4)&(i<=5):a[2]+=1
    else:a[3]+=1
Class_Figure.plot_pie(4,a,["(2,3]","(3,4]","(4,5]","(5,6]"])
Class_Figure.plot_box(5,Close,"Close")

Class_Figure.reset_figure_axes([221,222,223,224])
Class_Figure=MyClass_Figure()
Class_Figure.PlotLine2D(0,Close,objectname="close",cla=False,show=True,grid="x",PlotLabel=["ABCD","X轴","Y轴"])
Class_Figure.PlotLine2D(0,Open,objectname="Open",cla=False,show=True)
Class_Figure.PlotLine2D(1,Open,objectname="open",cla=False,show=True,linewidth=5,grid="both",PlotLabel=["ABCD","X轴","Y轴"])
Class_Figure.PlotLine2D(2,Close,objectname="close",cla=False,show=True,grid="x",PlotLabel=["ABCD","X轴","Y轴"])
Class_Figure.PlotLine2D(3,Open,objectname="open",cla=False,show=True,linewidth=5,grid="both",PlotLabel=["ABCD","X轴","Y轴"])

#多个Figure对象画图
# Class_Figure=MyClass_Figure()
# Class_Figure.PlotLine2D(0,Close,objectname="收盘价",cla=False,show=True,grid="x",PlotLabel=["ABCD","X轴","Y轴"])
# Class_Figure1=MyClass_Figure()
# Class_Figure1.PlotLine2D(0,Close,objectname="收盘价",cla=False,show=True,grid="x",PlotLabel=["ABCD","X轴","Y轴"])
# Class_Figure2=MyClass_Figure()
# Class_Figure2.PlotLine2D(0,Close,objectname="收盘价",cla=False,show=True,grid="x",PlotLabel=["ABCD","X轴","Y轴"])



