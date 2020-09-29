# Author:Zhang Yuan

import urllib.request as class_url
#返回一个可以读取的对象
webaddress="http://www.python.org"
webpage=class_url.urlopen(webaddress)
import re
text=webpage.read()
m=re.search(b'<a href="([^"]+)" .*?>about</a>',text,re.IGNORECASE) #正则表达式
print(m.group())
#获取远程文件
class_url.urlretrieve(webaddress,"download.html")
#class_url.urlcleanup() #上面不指定位置才需要用







