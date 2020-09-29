# =============================================================================
# 16.3.3 模型可视化呈现 by 王宇韬
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

# # 二、模型可视化呈现（供感兴趣的读者参考，其中绘图插件graphviz的安装及使用见如下网址：https://shimo.im/docs/lUYMJX0TEjoncFZk /）
# 1.如果不用显示中文，那么通过如下代码即可：
from sklearn.tree import export_graphviz
import graphviz
import os  # 以下这两行是手动进行环境变量配置，防止在本机环境的变量部署失败
os.environ['PATH'] = os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'
dot_data = export_graphviz(clf, out_file=None, class_names=['0', '1'])
# print(dot_data)
graph = graphviz.Source(dot_data)
# print(graph)
graph.render("result")
print('可视化文件result.pdf已经保存在代码所在文件夹！')

# 2.如果想显示中文，需要使用如下代码
from sklearn.tree import export_graphviz
dot_data = export_graphviz(clf, out_file=None, feature_names=X_train.columns, class_names=['不违约', '违约'], rounded=True, filled=True)
import os  # 以下这两行是手动进行环境变量配置，防止在本机环境的变量部署失败
os.environ['PATH'] = os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'

with open('dot_data.txt', 'w') as f:
    f.writelines(dot_data)
import codecs
import re
txt_dir = r'dot_data.txt'
txt_dir_utf8 = r'dot_data_utf8.txt'
with codecs.open(txt_dir, 'r') as f, codecs.open(txt_dir_utf8, 'w', encoding='utf-8') as wf:
    for line in f:
        if 'fontname' in line:
            font_re = 'fontname=(.*?)]'
            old_font = re.findall(font_re, line)[0]
            line = line.replace(old_font, 'SimHei')
        newline = line
        wf.write(newline + '\t')
wf.close()
os.system('dot -Tpng dot_data_utf8.txt -o example.png')  # 以PNG的图片形式存储生成的可视化文件
print('可视化文件example.png已经保存在代码所在文件夹！')

os.system('dot -Tpdf dot_data_utf8.txt -o example.pdf')  # 以PDF的形式存储生成的可视化文件
print('可视化文件example.pdf已经保存在代码所在文件夹！')
















