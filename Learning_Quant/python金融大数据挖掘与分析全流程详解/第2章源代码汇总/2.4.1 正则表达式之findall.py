# =============================================================================
# 2.4.1 正则表达式之findall by 王宇韬
# =============================================================================

import re
content = 'Hello 123 world 456 华小智Python基础教学135'
result = re.findall('\d\d\d',content)
print(result)

# 注意获取到的是一个列表
print(result[0])
print(result[1])
print(result[2])

# 更简单的遍历方法，其中len表示列表长度，range(n)表示0到n-1
for i in range(len(result)):
    print(result[i])

