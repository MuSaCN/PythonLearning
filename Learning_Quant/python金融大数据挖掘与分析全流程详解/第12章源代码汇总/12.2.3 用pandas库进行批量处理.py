# =============================================================================
# 12.2.3 利用pandas进行批处理 by 王宇韬 & 房宇亮
# =============================================================================

# 1.apply()函数
# 1.1 先创建一个DataFrame
import pandas as pd
data = pd.DataFrame([[-1, -2, -3], [1, 2, 3], [4, 5, 6]], columns=['c1', 'c2', 'c3'])
print(data)

# 1.2 apply()函数演示 - 对整张表应用
def y(x):  # 定义一个函数，函数返回值为x+1
    return x + 1

a = data.apply(y)  # 传入函数名称，这个函数会作用于整张表格
print(a)

# 1.3 apply()函数演示 - 对单列应用
b = data['c1'].apply(y)  # 这里返回的是一个Series一维对象
print(b)
c = data[['c1']].apply(y)  # 这样返回的是一个二维的DataFrame表格
print(c)

# 1.4 修改原表格
data['c1'] = data['c1'].apply(y)
print(data)


# 2.常规批处理方式
data = pd.DataFrame([[-1, -2, -3], [1, 2, 3], [4, 5, 6]], columns=['c1', 'c2', 'c3'])
a = data + 1  # 对于整张表格加一
b = data['c1'] + 1  # 对于c1列加一
data['c1'] = data['c1'] + 1  # 对于c1列加一，并赋值给原来的c1列
print(a)
print(b)
print(data)

# 可以看到通过常规方式反而更加简洁，但是对于一些较为复杂的函数的话，比如函数中涉及if判断语句，最好还是利用apply()库，代码如下：
data = pd.DataFrame([[-1, -2, -3], [1, 2, 3], [4, 5, 6]], columns=['c1', 'c2', 'c3'])
def y(x):
    if x > 0:
        return x + 100
    else:
        return x - 100


a = data['c1'].apply(y)
print(a)

# 3.lambda()函数
y = lambda x: x+1
print(y(1))

# lambda()函数使得代码更简洁
a = data.apply(lambda x: x+1)  # 这里的x针对data整张表格
print(a)
b = data['c1'].apply(lambda x: x+1)  # 这里的x针对c1这一列
print(b)

# 对c1列取绝对值
c = data['c1'].apply(lambda x: abs(x))
print(c)
# 可以直接按下面的方法写：
c = abs(data['c1'])

# 如果想将c1列和c2列相加并生成新的一列，可以采用如下代码：
data['c4'] = data.apply(lambda x: x['c1'] + x['c2'], axis=1)
# 可以直接按下面的方法写：
data['c4'] = data['c1'] + data['c2']

'''
有的读者可能就有疑问了，既然可以pandas库可以支持一些简单运算，
为什么还要学apply()及lambda()函数呢？一是补充下apply()及lambda()函数知识点，
因为还有很多人习惯使用apply()函数，学完后再看到别人用apply()函数的时候不至于看不懂；
二是因为有的函数比较复杂，那么这时用apply()函数就进行批量操作就比较方便。
'''