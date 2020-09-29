# =============================================================================
# 2.4.2 正则表达式之非贪婪匹配1 by 王宇韬
# =============================================================================

# 非贪婪匹配之(.*?) 简单示例1
import re
res = '文本A百度新闻文本B'
source = re.findall('文本A(.*?)文本B', res)
print(source)

# 非贪婪匹配之(.*?) 简单示例2 注意获取到的结果是一个列表
import re
res = '文本A百度新闻文本B，新闻标题文本A新浪财经文本B，文本A搜狗新闻文本B新闻网址'
p_source = '文本A(.*?)文本B'
source = re.findall(p_source, res)
print(source)

# 非贪婪匹配之(.*?) 实战演练
import re
res = '<p class="c-author"><img***>央视网新闻&nbsp;&nbsp;2019年04月13日 13:33</p>'
p_info = '<p class="c-author">(.*?)</p>'
info = re.findall(p_info, res)
print(info)


