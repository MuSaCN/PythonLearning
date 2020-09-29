# Author:Zhang Yuan
from MyPackage import *
myML = MyMachineLearning.MyClass_MachineLearning()  # 机器学习综合类
from MyPackage.bookcode.preamble import *

from sklearn.tree import DecisionTreeClassifier
cancer = myML.DataPre.load_datasets("breast_cancer")
X_train, X_test, y_train, y_test = myML.DataPre.train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# %%
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
# myML.TreeModel.PlotTree_Tree(tree)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# %% md

#### Analyzing Decision Trees

# %%
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)

# %%
import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


