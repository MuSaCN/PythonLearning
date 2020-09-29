# =============================================================================
# 8.2 爬虫进阶2-爬虫利器selenium库详解 by 王宇韬
# =============================================================================

# 1.打开及关闭网页+网页最大化
from selenium import webdriver
browser = webdriver.Chrome()
browser.maximize_window()
browser.get("https://www.baidu.com/")
browser.quit()

# 2.xpath方法来定位元素
from selenium import webdriver
browser = webdriver.Chrome()
browser.get("https://www.baidu.com/")
browser.find_element_by_xpath('//*[@id="kw"]').send_keys('python')
browser.find_element_by_xpath('//*[@id="su"]').click()

# 3.css_selector方法来定位元素
from selenium import webdriver
browser = webdriver.Chrome()
browser.get("https://www.baidu.com/")
browser.find_element_by_css_selector('#kw').send_keys('python')
browser.find_element_by_css_selector('#su').click()

# 4.browser.page_source方法来获取模拟键盘鼠标点击，百度搜索python后的网页源代码
from selenium import webdriver
import time
browser = webdriver.Chrome()
browser.get("https://www.baidu.com/")
browser.find_element_by_xpath('//*[@id="kw"]').send_keys('python')
browser.find_element_by_xpath('//*[@id="su"]').click()
time.sleep(3)  # 因为是点击按钮后跳转，所以最好休息3秒钟再进行源代码获取,如果是直接访问网站，则通常不需要等待。
data = browser.page_source
print(data)

# 5.browser.page_source方法来获取新浪财经股票信息
from selenium import webdriver
browser = webdriver.Chrome()
browser.get("http://finance.sina.com.cn/realstock/company/sh000001/nc.shtml")
data = browser.page_source
print(data)

# 6.Chrome Headless无界面浏览器设置
from selenium import webdriver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
browser = webdriver.Chrome(chrome_options=chrome_options)
browser.get("http://finance.sina.com.cn/realstock/company/sh000001/nc.shtml")
data = browser.page_source
print(data)
