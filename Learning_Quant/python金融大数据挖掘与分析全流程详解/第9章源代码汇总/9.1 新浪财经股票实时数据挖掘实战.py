# =============================================================================
# 9.1 新浪股票实时数据挖掘实战 by 王宇韬
# =============================================================================

from selenium import webdriver
import re
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
browser = webdriver.Chrome(chrome_options=chrome_options)
browser.get('http://finance.sina.com.cn/realstock/company/sh000001/nc.shtml')
data = browser.page_source
# print(data)
browser.quit()

p_price = '<div id="price" class=".*?">(.*?)</div>'
price = re.findall(p_price, data)
print(price)