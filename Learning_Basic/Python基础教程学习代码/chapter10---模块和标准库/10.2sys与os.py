# Author:Zhang Yuan

import sys
print(sys.argv)

import os
#打开window应用程序，必须按照windows shell格式
#下面两句将执行win应用，第一句单双引号反了不行
#os.system(r'C:\"Program Files (x86)"\"Maxthon5"\"Bin"\Maxthon.exe')  #os.system打开外部应用，应用结束后才进入下一句代码
# os.startfile(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe")
# os.startfile(r"C:\Program Files (x86)\Maxthon5\Bin\Maxthon.exe")
# os.startfile(r"D:\Program Files (x86)\360Chrome\Chrome\Application\360chrome.exe")

#直接打开网页
# import webbrowser
# webbrowser.open("https://www.python.org")

# import fileinput
# for line in fileinput.input(inplace=True):
#     line=line.rstrip()
#     num=fileinput.lineno()
#     print("{:<50} # {:2d}".format(line,num))

#集合，没有重复的元素
a=set([1,2,3,4,2,1])
print(a) #{1, 2, 3, 4}
#{...}来设定集合，但是不能{}，{}表示字典
b={2,3,8,8,8,8,"abc","abc","abc"}
print(b) #{8, 2, 3, 'abc'}
print(a.union(b),a|b,a&b) #集合运算
#集合是可变的，不能用作字典中的键。集合只能包含不可变（可散列）的值，不能包含其他集合
a.add(frozenset(b)) #如果要包含其他集合，必须冻结住该集合
print(a) #{1, 2, 3, 4, frozenset({8, 2, 3, 'abc'})}





