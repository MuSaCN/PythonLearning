# Author:Zhang Yuan


# #-------------------以下只适合读小文件-----------------------
#打开当前目录下的文本文件
filename="testfile1.txt"
f=open(filename,"r",encoding="utf-8")  #文件句柄,"r表示只读(默认)
#.read()没有参数表示全部读完，将文件内容放到一个字符串变量中，字符串变量可能包含换行符
print(f.read())
print(f.read()) #上面已经读完，没有结果
f.close()

g=open(filename,"w")  #文件句柄,"w"表示重头只写(全部重新写，不是按行覆盖)
g.write("My name is ZhangYuan\n")
g.close()

k=open(filename,"a")  #文件句柄,"a"表示末尾只写(尾部追加)
k.write("TEST1") #写入时如果没有加入换行号"\n"则不换行
k.write("TEST2\n")
k.write("TEST3\n")
k.write("TEST4\n")
k.close()

# f.readline()表示逐行读取
f=open(filename,"r",encoding="utf-8")
for i in range(3):
    print(i,f.readline())
f.close()

# f.readlines()全部读取，自动将文件内容按行分成一个列表
f=open(filename,"r",encoding="utf-8")
R=f.readlines()
print(R,"Finish") #['My name is ZhangYuan\n', 'TEST1TEST2\n', 'TEST3\n', 'TEST4\n'] Finish
for index,line in enumerate(R):
    print(index,line.strip()) #直接print(line)包括空格和换行，用strip()可去除
f.close()

#-------------------------以下适合大文件----------------------------------
print("以下适合大文件")
f=open(filename,"r")
#每次读取只保存一行到内存，不会累积内存
count=0
for line in f:
    print(count,line,"不积累内存")
    count+=1
f.close()

#with语句自动关闭文件，下面的方式适合大文件读取
with open("testfile1.txt","r",encoding="utf-8") as f1:
    for line in f1:
        print(line,"with语句")

#用fileinput模块读取大文件
import fileinput
for line in fileinput.input(filename):
    print(line,"fileinput模块")
