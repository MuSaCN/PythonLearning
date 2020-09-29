# Author:Zhang Yuan
#1
from datetime import datetime
currenttime=datetime.now()
print(currenttime.strftime("%Y-%m-%d %H:%M:%S"))

#2
# while True:
#     name=input("Input your name:")
#     if(name[0].isalpha()==True):
#         break
#     else:
#         print("用户名必须以字母开头")
# while True:
#     code=input("Input your code:")
#     if(code[0].isalpha()==True):
#         if(any([str(x) in code for x in ("_","*","#")])):
#             if (len(code)>=6):
#                 break
#             else:
#                 print("密码长度必须大于6")
#         else:
#             print("要包括_ * #中任意一个")
#     else:
#         print("密码必须以字母开头")
# print("用户创建成功！")

#3
print(list([i for i in range(101) if i%2==0])     )

#4
import datetime as dt
datelist=[dt.datetime(2015,1,13)+dt.timedelta(i) for i in range(5)]
pricelist=[7.31,7.28,7.40,7.43,7.41]
close_dict=dict(zip(datelist,pricelist))
close_dict[dt.datetime(2015,1,20)]=7.44
datelist.append(dt.datetime(2015,1,20))
pricelist.append(7.44)
print(close_dict)
print(close_dict[dt.datetime(2015,1,21)-dt.timedelta(4)] )
close_dict[dt.datetime(2015,1,16)]=7.5
print(close_dict)

#5
import math
cash=10000
volumn={datelist[0]:0}
for i in range(1,len(close_dict)):
    if close_dict[datelist[i]]>close_dict[datelist[i-1]]:
        tempvolumn= math.floor( (cash*0.5) / close_dict[datelist[i]] )
        volumn[datelist[i]]=tempvolumn
        cash=cash - tempvolumn*close_dict[datelist[i]] + volumn[datelist[i-1]]*close_dict[datelist[i]]
    else:
        volumn[datelist[i]]=0
print(volumn)






