# Author:Zhang Yuan

f = open("FileTest", "r", encoding="utf-8")
f_new = open("FileTest2", "w", encoding="utf-8")


for line in f:
    if "ZhangYuan" in line:
        line=line.replace("ZhangYuan","ZhangTEST")
    f_new.write(line)
f.close()
f_new.close()






