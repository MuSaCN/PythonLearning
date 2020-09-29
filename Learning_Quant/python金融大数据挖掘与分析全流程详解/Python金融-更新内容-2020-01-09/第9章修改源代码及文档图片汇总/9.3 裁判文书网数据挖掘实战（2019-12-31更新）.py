# =============================================================================
# 9.3 裁判文书网数据挖掘实战 by 王宇韬
# =============================================================================

'''2019年8月份之后裁判文书网改版，其反爬非常强，所以模拟键盘鼠标操作后等待很久也等不到刷新，
所以这里主要给大家练习下如何通过selenium库模拟键盘鼠标操作。'''

from selenium import webdriver
import time
browser = webdriver.Chrome()
browser.get('http://wenshu.court.gov.cn/')
browser.maximize_window()
browser.find_element_by_xpath('//*[@id="_view_1540966814000"]/div/div[1]/div[2]/input').clear()  # 清空原搜索框
browser.find_element_by_xpath('//*[@id="_view_1540966814000"]/div/div[1]/div[2]/input').send_keys('房地产')  # 在搜索框内模拟输入'房地产'三个字
browser.find_element_by_xpath('//*[@id="_view_1540966814000"]/div/div[1]/div[3]').click()  # 点击搜索按钮
time.sleep(10)  # 如果还是获取不到你想要的内容，你可以把这个时间再稍微延长一些，现在裁判文书网反爬非常厉害，所以可能等待也等不到刷新，所以这里主要给大家练习下模拟键盘鼠标操作
data = browser.page_source
browser.quit()
print(data)
