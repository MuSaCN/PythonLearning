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
# 导入Plot库
from MyPackage.MyClass_Plot import MyClass_Plot
Class_Plot=MyClass_Plot()
# 坐标轴范围更改
Class_Plot.PlotLine2D(-Close,xlim=["",pd.datetime(2020,1,1)],grid="x")
Class_Plot.PlotLine2D(Open,ylim=[0,10])
Class_Plot.PlotLine2D(Close["2014"],xlim=(pd.datetime(2013,1,1),pd.datetime(2016,1,1)),ylim=[0,10],grid="y")
Class_Plot.PlotLine2D(Close["2014"])
# 坐标点的标签更改
Class_Plot.PlotLine2D(Close["2014"],xPointLabel=(["2014-01-02","2014-12-01"],None))
d=[1,1,0,0,-1,0,1,1,-1]
Class_Plot.PlotLine2D(d,xPointLabel=[[1,2,3],["a","b","c"]],yPointLabel=[[0,1],["牛逼","厉害"]],PlotLabel=["测试Plot","X轴","Y轴"])
#
Class_Plot.PlotLine2D(Close["2014"],"收盘价",show=False,linewidth=3,linestyle="dashed",color="red",pointstyle=".",PlotLabel=["ChinaBank","Time","Price"],grid="both")
Class_Plot.PlotLine2D(Open["2014"],"开盘价",show=True)

# 柱状图
Close.describe()
a=[0,0,0,0]
for i in Close:
    if (i>2)&(i<=3):a[0]+=1
    elif (i>3)&(i<=4):a[1]+=1
    elif (i>4)&(i<=5):a[2]+=1
    else:a[3]+=1
Class_Plot.setting_2D(PlotLabel=["柱状图","柱点","数量"],grid="both")
Class_Plot.plot_bar([1.5,2.5,3.5,4.5],a,0.8,"Bar1",show=False,bottom=100)
Class_Plot.plot_bar([1.5,2.5,3.5,4.5],a,0.8,"Bar2",show=True,horizontal=True)

#频率直方图/频率累计直方图
Class_Plot.setting_2D(PlotLabel=["直方图","柱点","频率"],grid="both")
Class_Plot.PlotFreHistogram(Close,5,"直方图",True,[2.5,3.5],False)
Class_Plot.setting_2D(PlotLabel=["直方图","频率","柱点"],grid="both")
Class_Plot.PlotFreHistogram(Close,50,"直方图",True,[2.5,3.5],True)
Class_Plot.PlotFreHistogram(Close,50,"频率直方图",True,None,cumulative=True,type="bar")
Class_Plot.PlotFreHistogram(Close,50,"频率累计直方图",True,None,cumulative=False,type="bar")
Class_Plot.PlotFreHistogram(Close,50,"直方图",True,None,cumulative=True,type="barstacked")
Class_Plot.PlotFreHistogram(Close,50,"直方图",True,None,cumulative=True,type="step")
Class_Plot.PlotFreHistogram(Close,50,"直方图",True,None,cumulative=True,type="stepfilled")
Class_Plot.setting_2D(PlotLabel=["直方图","频率","柱点"],grid="both")
Class_Plot.PlotFreHistogram(Open,250,"开盘价频率直方图",True)

#饼图
Class_Plot.plot_pie(a,["(2,3]","(3,4]","(4,5]","(5,6]"])

#箱型图
Class_Plot.plot_box(Close,["Close"])
Class_Plot.plot_box(Close,"Close")
prcData=ChinaBank.iloc[:,:4]
Class_Plot.plot_box(prcData,('Open','High','Low','Close')) #可以传递DataFrame数据类型
Class_Plot.plot_box(ChinaBank,('Open','High','Low','Close',"volume"))
Class_Plot.plot_box(Close,"Close")
Class_Plot.plot_box(Close,"Close",whis=0.3)
Class_Plot.plot_box(ChinaBank,('Open','High','Low','Close',"volume"))
Class_Plot.plot_box(ChinaBank,('Open','High','Low','Close',"volume"),whis=0.5)







































