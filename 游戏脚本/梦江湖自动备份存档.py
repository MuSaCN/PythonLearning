# Author:Zhang Yuan


# %% # 每隔5分钟备份一次存档
from MyPackage import MyFile
myfile = MyFile.MyClass_File()  # 文件操作类

import time
from datetime import datetime       #---引入datetime类(必须)


while(True):
    savefile = r"E:\games\MJH\S110000111ca0088.sav"
    now = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    tofile = r"E:\games\MJH\存档保存\S110000111ca0088.{}.sav".format(str(now))
    myfile.copy_dir_or_file(savefile,tofile,False)
    print("存档以保存", tofile)
    time.sleep(60*5) # 每隔5分钟备份一次存档



