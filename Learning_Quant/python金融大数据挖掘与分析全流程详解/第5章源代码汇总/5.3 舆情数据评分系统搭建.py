# =============================================================================
# 5.3 舆情评分系统搭建 by 王宇韬
# =============================================================================

import requests  # 首先引用requests库和正则表达式re库
import re

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'}

'''舆情评分系统 - 通过新闻标题和新闻正文来进行评分'''
def baidu(company):
    url = 'https://www.baidu.com/s?tn=news&rtt=4&bsst=1&cl=2&wd=' + company  # 其中设置rtt=4则为按时间排序，如果rtt=1则为按焦点排序
    res = requests.get(url, headers=headers).text
    # print(res)

    # 正则表达式编写，这边为了代码简洁，只演示了标题和链接
    p_href = '<h3 class="c-title">.*?<a href="(.*?)"'
    p_title = '<h3 class="c-title">.*?>(.*?)</a>'
    href = re.findall(p_href, res, re.S)
    title = re.findall(p_title, res, re.S)

    # 舆情评分版本4
    score = []
    keywords = ['违约', '诉讼', '兑付', '阿里', '百度', '京东', '互联网']  # 这个关键词列表可以自己定义，这里只是为了演示
    for i in range(len(title)):
        num = 0
        # 获取新闻正文
        try:
            article = requests.get(href[i], headers=headers, timeout=10).text
        except:
            article = '爬取失败'
        # 解决可能存在的乱码问题
        try:
            article = article.encode('ISO-8859-1').decode('utf-8')  # 方法3
        except:
            try:
                article = article.encode('ISO-8859-1').decode('gbk')  # 方法2
            except:
                article = article  # 方法1
        # 只筛选真正的正文内容，旁边的滚动新闻之类的内容忽略
        p_article = '<p>(.*?)</p>'
        article_main = re.findall(p_article, article)  # 获取<p>标签里的正文信息
        article = ''.join(article_main)  # 将列表转换成为字符串  
        for k in keywords:
            if (k in article) or (k in title[i]):
                num -= 5
        score.append(num)

    for i in range(len(title)):
        title[i] = title[i].strip()
        title[i] = re.sub('<.*?>', '', title[i])
        title[i] = re.sub('&quot;', '', title[i])
        print(str(i + 1) + '.' + title[i])
        print(company + '该条新闻舆情评分为' + str(score[i]))  # 这边注意，不要写score[i]，因为它是数字，需要通过str()函数进行字符串拼接
        print(href[i])
    print('——————————————————————————————')  # 这个是当分隔符使用


companys = ['阿里巴巴', '万科集团', '百度', '腾讯', '京东']
for i in companys:
    try:
        baidu(i)
        print(i + '该公司百度新闻爬取成功')
    except:
        print(i + '该公司百度新闻爬取失败')


