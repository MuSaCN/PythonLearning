# Author:Zhang Yuan
from MyPackage import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

#------------------------------------------------------------
# __mypath__ = MyPath.MyClass_Path("\\Python大战机器学习")  # 路径类
myfile = MyFile.MyClass_File()  # 文件操作类
mytime = MyTime.MyClass_Time()  # 时间类
myplt = MyPlot.MyClass_Plot()  # 直接绘图类(单个图窗)
mypltpro = MyPlot.MyClass_PlotPro()  # Plot高级图系列
myfig = MyPlot.MyClass_Figure(AddFigure=False)  # 对象式绘图类(可多个图窗)
myfigpro = MyPlot.MyClass_FigurePro(AddFigure=False)  # Figure高级图系列
mynp = MyArray.MyClass_NumPy()  # 多维数组类(整合Numpy)
mypd = MyArray.MyClass_Pandas()  # 矩阵数组类(整合Pandas)
mypdpro = MyArray.MyClass_PandasPro()  # 高级矩阵数组类
myDA = MyDataAnalysis.MyClass_DataAnalysis()  # 数据分析类
# myMql = MyMql.MyClass_MqlBackups() # Mql备份类
# myMT5 = MyMql.MyClass_ConnectMT5(connect=False) # Python链接MetaTrader5客户端类
# myDefault = MyDefault.MyClass_Default_Matplotlib() # matplotlib默认设置
# myBaidu = MyWebCrawler.MyClass_BaiduPan() # Baidu网盘交互类
# myImage = MyImage.MyClass_ImageProcess()  # 图片处理类
myWebQD = MyWebCrawler.MyClass_WebQuotesDownload()  # 金融行情下载类
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector()  # 向量型回测类
myML = MyMachineLearning.MyClass_MachineLearning()  # 机器学习综合类
myWebC = MyWebCrawler.MyClass_WebCrawler() # 综合网络爬虫类
#------------------------------------------------------------

url = "https://www.baidu.com/s?rtt=1&bsst=1&cl=2&tn=news&word=阿里巴巴"
res = myWebC.get(url).text
print(res)

content = "fadsf 1234 3127 123 world"
result = myWebC.findall("\w\w\S",content)
print(result)


content = 'Hello 123 world 456 华小智Python基础教学135'
result = myWebC.findall('\d\d\d',content)
print(result)


# 非贪婪匹配之(.*?) 简单示例1
res = '文本A百度新闻文本B'
source = myWebC.findall('文本A(.*?)文本B', res)
print(source)

# 非贪婪匹配之(.*?) 简单示例2 注意获取到的结果是一个列表
res = '文本A百度新闻文本B，新闻标题文本A新浪财经文本B，文本A搜狗新闻文本B新闻网址'
p_source = '文本A(.*?)文本B'
source = myWebC.findall(p_source, res)
print(source)

# 非贪婪匹配之(.*?) 实战演练
res = '<p class="c-author"><img***>央视网新闻&nbsp;&nbsp;2019年04月13日 13:33</p>'
p_info = '<p class="c-author">(.*?)</p>'
info = myWebC.findall(p_info, res, 0)
print(info)



# 非贪婪匹配之.*? 简单示例
res = '<h3>文本C<变化的网址>文本D新闻标题</h3>'
p_title = '<h3>文本C.*?文本D(.*?)</h3>'
title = myWebC.findall(p_title, res)
print(title)

# 非贪婪匹配之.*? 实战演练
res = '<h3 class="c-title"><a href="网址" data-click="{一堆英文}"><em>阿里巴巴</em>代码竞赛现全球首位AI评委 能为代码质量打分</a>'
p_title = '<h3 class="c-title">.*?>(.*?)</a>'
title = myWebC.findall(p_title, res)
print(title)


# 2.4.4 正则表达式之换行
res = '''<h3 class="c-title">
 <a href="https://baijiahao.baidu.com/s?id=1631161702623128831&amp;wfr=spider&amp;for=pc"
    data-click="{
      一堆我们不关心的英文
      }"
                target="_blank"
    >
      <em>阿里巴巴</em>代码竞赛现全球首位AI评委 能为代码质量打分
    </a>
'''
p_href = '<h3 class="c-title">.*?<a href="(.*?)"'
p_title = '<h3 class="c-title">.*?>(.*?)</a>'
href = myWebC.findall(p_href, res)
title = myWebC.findall(p_title, res)
print(href)
print(title)


# 2.4.5 正则表达式之小知识点补充

# 1 re.sub()方法实现批量替换
# 1.1 传统方法-replace()函数
title = ['<em>阿里巴巴</em>代码竞赛现全球首位AI评委 能为代码质量打分']
title[0] = title[0].replace('<em>','')
title[0] = title[0].replace('</em>','')
print(title[0])

# 1.2 re.sub()方法
title = ['<em>阿里巴巴</em>代码竞赛现全球首位AI评委 能为代码质量打分']
title[0] = myWebC.sub('<.*?>', '', title[0])
print(title[0])

# 2 中括号[ ]的用法：使在中括号里的内容不再有特殊含义
company = '*华能信托'
company1 = myWebC.sub('[*]', '', company)
print(company1)


