# =============================================================================
# 2.4.3 正则表达式之非贪婪匹配2 by 王宇韬
# =============================================================================

# 非贪婪匹配之.*? 简单示例
import re
res = '<h3>文本C<变化的网址>文本D新闻标题</h3>'
p_title = '<h3>文本C.*?文本D(.*?)</h3>'
title = re.findall(p_title, res)
print(title)

# 非贪婪匹配之.*? 实战演练
import re
res = '<h3 class="c-title"><a href="网址" data-click="{一堆英文}"><em>阿里巴巴</em>代码竞赛现全球首位AI评委 能为代码质量打分</a>'
p_title = '<h3 class="c-title">.*?>(.*?)</a>'
title = re.findall(p_title, res)
print(title)
