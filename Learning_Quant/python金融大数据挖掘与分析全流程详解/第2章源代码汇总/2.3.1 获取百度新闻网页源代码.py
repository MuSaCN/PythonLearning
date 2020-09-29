# =============================================================================
# 2.3.1 获取百度新闻网页源代码 by 王宇韬
# =============================================================================

import requests
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'}
url = 'https://www.baidu.com/s?tn=news&rtt=1&bsst=1&cl=2&wd=阿里巴巴'
res = requests.get(url, headers=headers).text
print(res)
