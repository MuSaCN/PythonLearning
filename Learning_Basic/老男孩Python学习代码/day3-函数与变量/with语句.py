# Author:Zhang Yuan

#with语句自动关闭文件
with open("FileTest","r",encoding="utf-8") as f1,open("FileTest2","r",encoding="utf-8") as f2:
    for line in f1:
        print(line)
    for line in f2:
        print(line)


