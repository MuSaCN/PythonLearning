# Author:Zhang Yuan
#在Python中实现文件复制、移动、压缩、解压等高级功能
import shutil

f1=open("本节笔记",encoding="utf-8")

f2=open("笔记2","w",encoding="utf-8")

shutil.copyfileobj(f1,f2)
f1.close()
f2.close()

shutil.copyfile("笔记2","笔记3")

shutil.copystat("笔记2","笔记3")

import zipfile
z=zipfile.ZipFile("Day5_Zip.zip","w")
z.write("p_test.py")
z.write("笔记2")
z.close()


