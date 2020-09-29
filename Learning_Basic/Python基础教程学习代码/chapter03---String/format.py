# Author:Zhang Yuan

#设置宽度后，数和字符串的对齐方式不同
print("{num:10}".format(num=3))
print("{num:10}".format(num="abc"))

print("{num:.2f}".format(num=3.141592653))
print("{num:10.2f}".format(num=3.141592653))

print("{num:,}".format(num=10**10))
print("{num:30,.2f}".format(num=10**10))





