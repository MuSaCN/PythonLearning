# Author:Zhang Yuan

import sys
print(sys.path)  # 打印环境变量
print(sys.argv)  # 打印当前脚本路径

import os
cmd_res = os.system("dir")  # 执行win命令，不保存结果
print("-->", cmd_res)
cmd_res = os.popen("dir").read()  # 保存结果
print("-->", cmd_res)
# os.mkdir("test")  #在当前脚本所在目录创建目录

#import login

#python中数据类型：int,float,complex
print(type(10*24))
print(type(23.123465789))
print(type(5+4j))  #python中虚数用j表示，不同于数学用i表示.

#三元运算
a,b,c=100,200,300
d=a if a>b else c
print(d)
e= 100 if 200>1000 else 200
print(e)

#16进制表示法
'''
后缀：H
前缀：0x
'''

#python3中 byte字节与str字符串是区分的，两者可以转换
print('€20'.encode('utf-8'))               #str转二进制byte
print(b'\xe2\x82\xac20'.decode('utf-8'))    #byte转str

msg = '我爱北京天安门'
print(msg)
print(msg.encode('utf-8'))
print(msg.encode('utf-8').decode('utf-8'))

