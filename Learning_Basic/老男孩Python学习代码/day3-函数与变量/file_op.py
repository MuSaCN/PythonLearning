# Author:Zhang Yuan

#-------------------以下.readlines()只适合读小文件-----------------------
#data= open("FileTest",encoding="utf-8").read()
#print(data)
f=open("FileTest","r",encoding="utf-8")  #文件句柄,"r表示只读(默认)

data=f.read()
data2=f.read() #接着上面的位置继续读，不是重新读
print(data)
print('---------------------------')
print(data2,"OK")   #上面的已经读完，所以没有内容

g=open("FileTest","w",encoding="utf-8")  #文件句柄,"w"表示重头只写(重新写)
g.write("ABCDEFGG\n")
g.write("abcdefrg\n")
g.write("ABCDEFGG\n")
g.write("abcdefrg\n")
g.write("My name is ZhangYuan\n")
g.write("abcdefrg\n")
g.write("123467\n")
g.write("7654321\n")

k=open("FileTest","a",encoding="utf-8")  #文件句柄,"a"表示末尾只写(尾部追加)
k.write("TEST1")
k.write("TEST2\n")
k.write("TEST3\n")
k.write("TEST4\n")

f.close()
g.close()
k.close()

f=open("FileTest","r",encoding="utf-8")
#可用循环方式逐行读取
#for i in range(3):
#    print(f.readline())

print("--------我是分割线---------")
R=f.readlines()
print(R,"Finish")
for index,line in enumerate(R):
    print(index,line.strip()) #直接print(line)包括空格和换行，用strip()可去除
f.close()
#-------------以上.readlines()只适合读小文件---------------------------

#-------------------------以下适合大文件----------------------------------
print("以下适合大文件")
f=open("FileTest","r",encoding="utf-8")
#每次读取只保存一行到内存，不会累积内存
count=0
for line in f:
    print(count,line,"不积累内存")
    count+=1
f.close()
#-------------------------以上适合大文件----------------------------------

f=open("FileTest","r",encoding="utf-8")
print(f.tell()) #按字符来计数返回file读取位置
print(f.readline())
print(f.tell()) #按字符来计数返回file读取位置
f.seek(0) #光标回到0
print(f.tell())

print(f.encoding)#返回编码方式
#强制刷新（确认下内存或缓存中的数据已经写入硬盘，有时候写入的数据会保存到缓存，出问题就不能保存在硬盘了）
print(f.flush())

#截断前20个字符保留，其余删除。不输入则全部清空
'''
f=open("FileTest","a",encoding="utf-8") #如果是"w"，表示file以新文件打开，里面没有内容，截断内容只能用"a"
f.truncate(20) #截断前20个字符保留（跟光标位置无关）
f.close()
'''

#硬盘上File文件的写入都是覆盖原位置，所以不存在直接性的在指定光标位置写入，除非进一步处理
#文件先读后写(追加)
f=open("FileTest","r+",encoding="utf-8")
print(f.readline())
print(f.readline())
f.write("\nABCDEFG")
f.close()
'''
#文件先写后读
print("---先写后读---")
f=open("FileTest","w+",encoding="utf-8")
print(f.readline())
f.close()

#文件先追加后读
f=open("FileTest","a+",encoding="utf-8")
f.close()
'''

#以二进制方式
print("---二进制---")
f=open("FileTest","rb") #二进制不能传递encoding="utf-8"参数
print(f.readline())
print(f.readline())
print(f.readline())
g=open("FileTest","ab")
g.write("\nHello World\n".encode())


