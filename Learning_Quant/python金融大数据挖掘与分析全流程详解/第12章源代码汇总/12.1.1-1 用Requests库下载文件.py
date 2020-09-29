# =============================================================================
# 12.1.1-1 requests库下载文件 by 王宇韬 & 房宇亮
# =============================================================================

# 1.下载CSV数据文件
# 这个文件比较大下载时间较长，所以如果想尝试的话，可以先将下面的代码批量注释掉（都选中后ctrl+/进行注释），先运行下面的图片下载进行文件下载体验
import requests
url = 'http://search.worldbank.org/api/projects/all.csv'
res = requests.get(url)  # 只要能够获得下载链接，像Excel文件、图片文件都可以进行下载
file = open('世界银行项目表.csv', 'wb')  # 可以修改所需的文件保存路径，这里得选择wb二进制的文件写入方式
file.write(res.content)
file.close()  # 通过close()函数关闭open()函数打开的文件，有助于释放内存，是个编程的好习惯
print('世界银行项目表.csv下载完毕')

# 2.通过requests库还可以下载图片
import requests
url = 'http://images.china-pub.com/ebook8055001-8060000/8057968/shupi.jpg'
res = requests.get(url)  # 只要能够获得下载链接，像Excel文件、图片文件都可以进行下载
file = open('图片.jpg', 'wb')  # 这里采用的是相对路径，也即代码所在的文件夹
file.write(res.content)
file.close()  # 通过close()函数关闭open()函数打开的文件，有助于释放内存，是个编程的好习惯
print('图片.jpg下载完毕，并保存在代码所在的文件夹')
