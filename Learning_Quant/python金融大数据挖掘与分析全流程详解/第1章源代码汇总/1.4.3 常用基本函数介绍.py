# =============================================================================
# 1.4.3 一些重要的基本函数介绍 by 华能信托-王宇韬
# =============================================================================

# 1 print函数，记得加括号即可,快捷方式，写pri的时候按一下tab键会自动变成print()
print('hello world')

# 2 str函数与int函数 - 字符串与数字转换
score = 85
print('A公司今日评分为' + str(score) + '分。')  # 不加str就成了字符串与数字相加会报错的

score = '85'
score = int(score)  # 不加int的话，下面字符串'85'和数字80没法比较
if score > 80:
    print('OK')

# 2 len函数
# len函数可以统计列表里元素的个数
title = ['标题1', '标题2', '标题3', '标题4', '标题5']
href = ['网址1', '网址2', '网址3', '网址4', '网址5']
for i in range(len(title)):  # len(title)表示一个有多少个新闻，这里是5
    href[i] = 'www.baidu.com/' + href[i]  # 这个其实就相当于 a = a + 1
    print(str(i+1) + '.' + title[i])  # 这个其实把字符串进行一个拼接
    print(href[i])

# len函数还可以统计字符串个数
a = '123华小智abcd'
print(len(a))

# 4 replace函数 - 替换你想替换的内容
a ='<em>阿里巴巴</em>电商脱贫成“教材” 累计培训逾万名县域干部'
a = a.replace('<em>','')
a = a.replace('</em>','')
print(a)

# 5 strip函数 - 删除空白符
a ='        华能信托2018年上半年行业综合排名位列第5        '
a = a.strip()
print(a)

# 6 split函数 - 分割字符串
a = '2018年12月12日 08:07'
a = a.split(' ')[0]
print(a)

# 注意，split分割完是一个列表
a = '2018年12月12日 08:07'
a = a.split(' ')
print(a)

# 7 异常处理函数try except函数，其实应该叫异常处理语句，后来调整到1.3.4小节了
try:
    print(1 + 'a')
except:
    print('主代码运行失败')

try:
    # 这里是百度新闻爬取的代码
    print(1 + '1')
    print('百度新闻爬取成功')
except:
    print('百度新闻爬取失败')

try:
    # 这里是百度新闻爬取的代码，之后第五第七章会讲
    print(1 + 'a')
    print('百度新闻爬取成功')
except:
    print('百度新闻爬取失败')

try:
    # 这里是新浪财经新闻爬取的代码，之后爬虫进阶课会讲
    print('新浪财经新闻爬取成功')
except:
    print('新浪财经新闻爬取失败')

try:
    # 这里是微信推文爬取的代码，之后爬虫进阶课会讲
    print('微信推文爬取成功')
except:
    print('微信推文爬取失败')

