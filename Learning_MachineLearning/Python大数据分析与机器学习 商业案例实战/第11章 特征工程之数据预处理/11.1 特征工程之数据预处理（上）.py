# # 第十一章 特征工程之数据预处理（11.1至11.4）
# # 11.1 非数值类型数据处理
# # 11.1.1 Get_dummies哑变量处理
# 1.简单示例：“男”和“女”的数值转换
import pandas as pd
df = pd.DataFrame({'客户编号': [1, 2, 3], '性别': ['男', '女', '男']})
print(df)

df = pd.get_dummies(df, columns=['性别'])
print(df)

df = df.drop(columns='性别_女') 

print(df)

df = df.rename(columns={'性别_男':'性别'})
print(df)

# 2.稍复杂点的案例：房屋朝向的数值转换
import pandas as pd
df = pd.DataFrame({'房屋编号': [1, 2, 3, 4, 5], '朝向': ['东', '南', '西', '北', '南']})
print(df)

df = pd.get_dummies(df, columns=['朝向'])
print(df)

df = df.drop(columns='朝向_西')

print(df)

# # 11.1.2 Label Encoding编号处理
import pandas as pd
df = pd.DataFrame({'编号': [1, 2, 3, 4, 5], '城市': ['北京', '上海', '广州', '深圳', '北京']})

print(df)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label = le.fit_transform(df['城市'])

print(label)

df['城市'] = label

print(df)

df = pd.DataFrame({'编号': [1, 2, 3, 4, 5], '城市': ['北京', '上海', '广州', '深圳', '北京']})

df['城市'].value_counts()

df['城市'] = df['城市'].replace({'北京': 0, '上海': 1, '广州': 2, '深圳':3})
print(df)
