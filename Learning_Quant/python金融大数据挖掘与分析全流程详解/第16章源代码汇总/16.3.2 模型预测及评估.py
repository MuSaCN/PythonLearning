# =============================================================================
# 16.3.2 模型预测及评估 by 王宇韬
# =============================================================================

# # 一、模型搭建
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
df = pd.read_excel('客户信息及违约表现.xlsx')
# 1.提取特征变量和目标变量
X = df.drop(columns='是否违约')
y = df['是否违约']

# 2.划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 3.模型训练及搭建
clf = DecisionTreeClassifier(max_depth=3)
clf = clf.fit(X_train, y_train)

# # 二、模型预测及评估
# 1.直接预测是否违约
y_pred = clf.predict(X_test)
print(y_pred)

# 将预测值和实际值汇总看一下
a = pd.DataFrame()  # 创建一个空DataFrame
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
print(a.tail())

# 查看模型预测准确度
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)

# 2.预测不违约&违约概率
y_pred_proba = clf.predict_proba(X_test)
print(y_pred_proba)  # 打印看看预测的不违约&违约概率，此时获得y_pred_proba是个二维数组，共两列，左列为不违约概率，右列为违约概率

# 只查看违约概率,其中中括号中第一个元素：冒号表示全部行，第二个元素：1表示第二列，如果把
print(y_pred_proba[:, 1])

# 3.模型预测效果评估
# ROC曲线相关知识
from sklearn.metrics import roc_curve
fpr, tpr, thres = roc_curve(y_test.values, y_pred_proba[:, 1])

# 将阈值tpr、假警报率fpr、命中率tpr汇总看一下
a = pd.DataFrame()  # 创建一个空DataFrame
a['阈值'] = list(thres)
a['假警报率'] = list(fpr)
a['命中率'] = list(tpr)
print(a)

# 绘制ROC曲线，注意图片展示完要将其关闭才会执行下面的程序
import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.show()

# 求出AUC值
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test.values, y_pred_proba[:, 1])
print(score)

















