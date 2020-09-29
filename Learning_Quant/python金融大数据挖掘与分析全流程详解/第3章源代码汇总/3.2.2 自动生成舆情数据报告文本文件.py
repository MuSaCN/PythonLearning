# =============================================================================
# 3.2.2 自动生成舆情数据txt报告 by 王宇韬
# =============================================================================
import requests
import re
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'}


def baidu(company):
    url = 'https://www.baidu.com/s?tn=news&rtt=1&bsst=1&cl=2&wd=' + company
    res = requests.get(url, headers=headers).text
    # print(res)

    p_info = '<p class="c-author">(.*?)</p>'
    p_href = '<h3 class="c-title">.*?<a href="(.*?)"'
    p_title = '<h3 class="c-title">.*?>(.*?)</a>'
    info = re.findall(p_info, res, re.S)
    href = re.findall(p_href, res, re.S)
    title = re.findall(p_title, res, re.S)

    source = []  # 先创建两个空列表来储存等会分割后的来源和日期
    date = []
    for i in range(len(info)):
        title[i] = title[i].strip()
        title[i] = re.sub('<.*?>', '', title[i])
        info[i] = re.sub('<.*?>', '', info[i])
        source.append(info[i].split('&nbsp;&nbsp;')[0])
        date.append(info[i].split('&nbsp;&nbsp;')[1])
        source[i] = source[i].strip()
        date[i] = date[i].strip()
        print(str(i + 1) + '.' + title[i] + '(' + date[i] + '-' + source[i] + ')')
        print(href[i])

    file1 = open('E:\\数据挖掘报告.txt', 'a')  # 如果把a改成w的话，则每次生成txt的时候都会把原来的txt清空
    file1.write(company + '数据挖掘completed！' + '\n' + '\n')
    for i in range(len(title)):
        file1.write(str(i + 1) + '.' + title[i] + '(' + date[i] + '-' + source[i] + ')' + '\n')
        file1.write(href[i] + '\n')  # '\n'表示换行
    file1.write('——————————————————————————————' + '\n' + '\n')
    file1.close()


companys = ['华能信托', '阿里巴巴', '万科集团', '百度集团', '腾讯', '京东']
for i in companys:
    baidu(i)
    print(i + '百度新闻爬取成功')

print('数据获取及生成报告成功')
