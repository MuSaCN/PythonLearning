# =============================================================================
# 16.3.1 模型搭建 by 王宇韬
# =============================================================================

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
print(clf)  # 这里可以将训练好的模型打印出来看看

# 此时的模型已经训练好了，在下一小节就可以利用该模型来进行预测了
