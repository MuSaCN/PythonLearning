# =============================================================================
# 2.4.4 正则表达式之换行 by 王宇韬
# =============================================================================

import re
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
href = re.findall(p_href, res, re.S)
title = re.findall(p_title, res, re.S)
print(href)
print(title)

# 清除换行符号
for i in range(len(title)):
    title[i] = title[i].strip()
print(title)
