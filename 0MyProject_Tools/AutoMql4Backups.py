# -*- coding: utf-8 -*-
# Author:Zhang Yuan
import time
from MyPackage.MyMql import MyClass_MqlBackups

Mql4Path = "C:\\Users\\i2011\\AppData\\Roaming\\MetaQuotes\\Terminal\\F7DC4A11FD305E0AA6ED39F4697586F7\\MQL4"

myMql4 = MyClass_MqlBackups(Mql4Path,isPrint=True)

# ---MyExperts_MQL4, MyClass_MQL4, MyIndicators_MQL4, MyScripts_MQL4的复制操作
# myMql4.dir_copy(myMql4.ExpertsPath,"MyExperts_MQL4")
# time.sleep(3)
# myMql4.dir_copy(myMql4.IncludePath,"MyClass_MQL4")
# time.sleep(3)
# myMql4.dir_copy(myMql4.IndicatorsPath,"MyIndicators_MQL4")
# time.sleep(3)
# myMql4.dir_copy(myMql4.ScriptsPath,"MyScripts_MQL4")
# time.sleep(3)

# ---Files, Logs的清理操作
myMql4.dir_remove(myMql4.FilesPath,ignoreFolder=[], ignoreFiles=[])
time.sleep(3)
myMql4.dir_remove(myMql4.LogsPath,ignoreFolder=[], ignoreFiles=[])
time.sleep(3)

# ---MQL4文件夹备份
print("------开始压缩MQL4文件夹------")
needZip = Mql4Path # 需压缩的目录
# 备份到OneDrive的Work-Python备份文件夹
OneDrive_Mql4 = myMql4.myfile.zip_dir(needZip, zipPath=myMql4.mypath.get_onedrive_path() + "\\Work-Mql_backups" , zipName=None, autoName=True)
print("MQL4压缩文件保存完成，{}".format(OneDrive_Mql4))
time.sleep(5)

# ---上传到Baidu云
print("------开始上传压缩文件到Baidu云盘------")
from MyPackage.MyWebCrawler import MyClass_BaiduPan
myBaidu= MyClass_BaiduPan()    #百度网盘交互类
needUpload = OneDrive_Mql4
remotePath = "\\MyMql4Backups\\"
# 开始批量上传
print("{} 开始上传.".format(needUpload))
out = myBaidu.upload(localpath=needUpload, remotepath=remotePath, ondup="overwrite")
myBaidu.feedback_upload(out = out)
print("{} 上传完成.".format(needUpload))

print("全部完成！")




