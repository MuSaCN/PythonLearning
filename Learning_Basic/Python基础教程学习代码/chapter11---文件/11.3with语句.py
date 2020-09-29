# Author:Zhang Yuan

#with语句自动关闭文件，下面的方式适合大文件读取
with open("testfile1.txt","r",encoding="utf-8") as f1:
    for line in f1:
        print(line)


