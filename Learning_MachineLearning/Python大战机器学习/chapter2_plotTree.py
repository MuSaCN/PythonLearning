from MyPackage import *


myplt = MyPlot.MyClass_Plot()  # 直接绘图类(单个图窗)
mypltpro = MyPlot.MyClass_PlotPro()  # Plot高级图系列
myfig = MyPlot.MyClass_Figure(AddFigure=False)  # 对象式绘图类(可多个图窗)
myfigpro = MyPlot.MyClass_FigurePro(AddFigure=False)  # Figure高级图系列

# 画决策树需要恢复style
myplt.set_style("defaults")

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
X, y = load_iris(return_X_y=True)
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)


tree.plot_tree(clf)
plt.show()


