# =============================================================================
# 12.1.1-2 通过pandas获取表格 by 王宇韬 & 房宇亮
# =============================================================================

import pandas as pd

url = 'http://vip.stock.finance.sina.com.cn/q/go.php/vInvestConsult/kind/dzjy/index.phtml'  # 新浪财经数据中心提供股票大宗交易的在线表格
table = pd.read_html(url)[0]  # 通过pd.read_html(url)获取的是一个列表，所以仍需通过[0]的方式提取列表的第一个元素
print(table)

print(table)
table.to_excel('大宗交易表.xlsx')  # 如果想忽略行索引的话，可以设置index参数为False

print('获取表格成功！')