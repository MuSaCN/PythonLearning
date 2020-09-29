# Author:Zhang Yuan
#字符串转换符%
text="my name is %s, what is your name?"%"ZhangYuan"
print(text)
#模板字符串
# from string import Template
# tmpl=Template("Hello,$A and $B are friends")
# tmpl.substitute(A="Mars",B="ZhangYuan")
# print(tmpl)

#字符串方法format
content1="{},{} and {}".format("A","B","C")
content2="{2},{0} and {1}".format("A","B","C")
print(content1,content2)
content3="{name} is approximately {value:.2f}.".format(name="π",value=3.141592)
print(content3)

fullname=["ZhangYuan","Zhanxiang"]
print("Mr {name[1]} is here".format(name=fullname))

import math
tmpl="The {mod.__name__} module defines the value {mod.pi} for pi".format(mod=math)

print(tmpl,tmpl)


