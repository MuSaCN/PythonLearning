# =============================================================================
# 3.5.1 搜狗新闻数据挖掘实战 by 王宇韬
# =============================================================================

import requests
import re

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'}

company = "阿里巴巴"
def sogou(company):
    url = 'https://news.sogou.com/news?mode=1&sort=0&fixrank=1&query=' + company + '&shid=djt1'
    res = requests.get(url,headers=headers, timeout=10).text
    # print(res)

    # 编写正则提炼数据
    p_title = '<a href=".*?" id="uigs.*?" target="_blank">(.*?)</a>'
    title = re.findall(p_title, res)
    p_href = '<a href="(.*?)" id="u.*?" target="_blank">'
    href = re.findall(p_href, res)
    p_date = '<p class="news-from">(.*?)</p>'
    date = re.findall(p_date, res)

    # 数据清洗及打印输出
    for i in range(len(title)):
        title[i] = re.sub('<.*?>', '', title[i])
        title[i] = re.sub('&.*?;', '', title[i])
        date[i] = re.sub('<.*?>', '', date[i])
        print(str(i+1) + '.' + title[i] + '-' + date[i])
        print(href[i])


companys = ['华能信托', '阿里巴巴', '万科集团', '百度', '腾讯', '京东']
companys = ['阿里巴巴']
for i in companys:
    try:
        sogou(i)
        print(i + '搜狗新闻爬取成功')
    except:
        print(i + '搜狗新闻爬取失败')
