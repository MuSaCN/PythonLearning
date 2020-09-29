# Author:Zhang Yuan

#f.tell(),f.seek()
filename="testfile1.txt"
f=open(filename,"r",encoding="utf-8")
print(f.tell()) #按字符来计数返回file读取位置---0
print(f.readline())
print(f.tell()) #按字符来计数返回file读取位置---22
f.seek(0) #光标回到0
print(f.tell()) #---0

#返回编码方式
print(f.encoding)

#强制刷新（确认下内存或缓存中的数据已经写入硬盘，有时候写入的数据会保存到缓存，出问题就不能保存在硬盘了）
print(f.flush())
f.close()

#截断前20个字符保留，其余删除。不输入则全部清空
# f=open(filename,"a",encoding="utf-8") #如果是"w"，表示file以新文件打开，里面没有内容，截断内容只能用"a"
# f.truncate(20) #截断前20个字符保留（跟光标位置无关）
# f.close()

#硬盘上File文件的写入都是覆盖原位置，所以不存在直接性的在指定光标位置插入，除非进一步处理
#文件先读后写(追加)
f=open(filename,"r+",encoding="utf-8")
print("first",f.readline())
print("second",f.readline())
f.write("\nABCDEFG") #在文件最后写入，不是插入
f.close()

# #文件先写后读：w+模式，会覆盖全部文件
# print("---先写后读---")
# f=open(filename,"w+",encoding="utf-8")
# f.write("\n先写后读")
# print(f.readline(),"先写后读") #在最后追加，所以读取内容为空
# print(f.readline(),"先写后读") #在最后追加，所以读取内容为空
# f.close()

# #文件先追加后读：a+模式
# f=open(filename,"a+",encoding="utf-8")
# f.write("\n先追加后读") #在文件最后写入，不是插入
# print(f.readline(),"a+模式") #在最后追加，所以读取内容为空
# print(f.readline(),"a+模式") #在最后追加，所以读取内容为空
# f.close()

#以二进制方式
print("---二进制---")
f=open(filename,"rb") #二进制不能传递encoding="utf-8"参数
print(f.readline())
print(f.readline())
print(f.readline())
f.close()
g=open(filename,"ab")
g.write("\nHello World\n".encode())
g.close()
