# =============================================================================
# 12.1.2 和讯研报网表格获取 by 王宇韬 & 房宇亮
# =============================================================================

# 它可能会弹出一个Warning警告，警告不是报错，不用在意
import pandas as pd
from selenium import webdriver
import re
# 设置Selenium的无界面模式
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
browser = webdriver.Chrome(options=chrome_options)

data_all = pd.DataFrame()  # 创建一个空列表用来汇总所有表格信息
for pg in range(1, 2):  # 可以将页码调大，比如2019-04-30该天，网上一共有176页，这里可以将这个2改成176
    url = 'http://yanbao.stock.hexun.com/ybsj5_' + str(pg) + '.shtml'
    browser.get(url)  # 通过Selenium访问网站
    data = browser.page_source  # 获取网页源代码
    table = pd.read_html(data)[0]  # 通过pandas库提取表格

    # 添加股票代码信息
    p_code = '<a href="yb_(.*?).shtml'
    code = re.findall(p_code, data)
    table['股票代码'] = code

    # 通过concat()函数纵向拼接成一个总的DataFrame
    data_all = pd.concat([data_all, table], ignore_index=True)

print(data_all)
print('分析师评级报告获取成功')
data_all.to_excel('分析师评级报告.xlsx')

'''如果上面代码运行出现报错(pandas旧版本问题导致获取的表格把表头当作了其中一行)，可以试试下面的代码'''
# import pandas as pd
# from selenium import webdriver
# import re
# # 设置Selenium的无界面模式
# chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument('--headless')
# browser = webdriver.Chrome(options=chrome_options)
#
# data_all = pd.DataFrame()  # 创建一个空列表用来汇总所有表格信息
# for pg in range(1, 2):  # 可以将页码调大，比如2019-04-30该天，网上一共有176页，这里可以将这个2改成176
#     url = 'http://yanbao.stock.hexun.com/ybsj5_' + str(pg) + '.shtml'
#     browser.get(url)  # 通过Selenium访问网站
#     data = browser.page_source  # 获取网页源代码
#     table = pd.read_html(data)[0]  # 通过pandas库提取表格
#     df = table.iloc[1:]  # 改变表格结构，从第二行开始选取
#     df.columns = table.iloc[0]  # 将原来的第一行内容设置为表头
#
#     # 添加股票代码信息
#     p_code = '<a href="yb_(.*?).shtml'
#     code = re.findall(p_code, data)
#     df['股票代码'] = code
#
#     # 通过concat()函数纵向拼接成一个总的DataFrame
#     data_all = pd.concat([data_all, df], ignore_index=True)
#
# print(data_all)
# print('分析师评级报告获取成功')
# data_all.to_excel('分析师评级报告.xlsx')
