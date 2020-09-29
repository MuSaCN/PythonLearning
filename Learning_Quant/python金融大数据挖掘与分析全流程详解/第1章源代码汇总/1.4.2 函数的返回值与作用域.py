# =============================================================================
# 1.4.2 函数返回值、作用域 by 华能信托-王宇韬
# =============================================================================

# 这里的company只是个代号，你可以换成任何东西，比如keyword，cat，dog都可以，注意函数内容里的company也要相应改变
def baidu(keyword):
    # 这里是具体爬虫代码，第七章会讲
    print(keyword + 'completed!')

companys = ['华能信托', '阿里巴巴', '百度', '腾讯', '京东', '万科', '建设银行']
for i in companys:
    baidu(i)

'''2.返回值'''
def y(x):
    return(x+1)
y(1)

'''return相当于看不见的print，它是把原来该print的值赋值给了y(x）这个函数，学术点的说法就是该函数的返回值为：x+1'''

def y(x):
    return(x+1)
a = y(1)
print(a) # 这样才能把 y(1) 打印出来，或者直接写print(y1)

# 在实战中的应用
def baidu(keyword):
    # 这里是具体爬虫代码，之后讲
    return(keyword + 'completed!')

a = baidu('华能信托')
print(a)

# return不加括号也是可以的
def baidu(keyword):
    # 这里是具体爬虫代码，第七章会讲
    return keyword + 'completed!'  #这里把括号去掉了

a = baidu('华能信托')
print(a)

'''3.变量作用域（了解即可）'''
x = 1
def y(x):
    x = x + 1
    print(x)
y(3)

print(x)

# 其实函数参数只是个代号，可以换成任何内容
x = 1
def y(z):
    z = z + 1
    print(z)
y(3)

print(x)
