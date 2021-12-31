# Author:Zhang Yuan
from MyPackage.MyMql import MyClass_MqlBackups
import time

Mql5Path = "C:\\Users\\i2011\\AppData\\Roaming\\MetaQuotes\\Terminal\\6E8A5B613BD795EE57C550F7EF90598D\\MQL5"

myMql5 = MyClass_MqlBackups(Mql5Path,isPrint=True)

# ---My_Experts, My_Include, My_Indicators, My_Scripts复制备份操作
# myMql5.dir_copy(myMql5.ExpertsPath,"My_Experts")
# time.sleep(3)
# myMql5.dir_copy(myMql5.IncludePath,"My_Include")
# time.sleep(3)
# myMql5.dir_copy(myMql5.IndicatorsPath,"My_Indicators")
# time.sleep(3)
# myMql5.dir_copy(myMql5.ScriptsPath,"My_Scripts")
# time.sleep(3)

# ---Files, Logs清理操作
myMql5.dir_remove(Path=myMql5.FilesPath,ignoreFolder=["SPSS", "IndiRecord"], ignoreFiles=["Common Files.lnk"])
time.sleep(3)
myMql5.dir_remove(myMql5.LogsPath,ignoreFolder=[], ignoreFiles=[])
time.sleep(3)

# ---MQL5文件夹备份到OneDrive的Work-Python备份文件夹
print("------开始压缩MQL5文件夹------")
needZip = Mql5Path # 需压缩的目录
OneDrive_Mql5 = myMql5.myfile.zip_dir(needZip, zipPath=myMql5.mypath.get_onedrive_path() + "\\Work-Mql_backups" , zipName=None, autoName=True)
print("MQL5压缩文件保存完成，{}".format(OneDrive_Mql5))
time.sleep(5)

# ---上传到Baidu云
print("------开始上传压缩文件到Baidu云盘------")
from MyPackage.MyWebCrawler import MyClass_BaiduPan
myBaidu= MyClass_BaiduPan()    # 百度网盘交互类
needUpload = OneDrive_Mql5
remotePath = "\\MyMql5Backups\\"
# 开始批量上传
print("{} 开始上传.".format(needUpload))
out = myBaidu.upload(localpath=needUpload, remotepath=remotePath, ondup="overwrite")
myBaidu.feedback_upload(out = out)
print("{} 上传完成.".format(needUpload))

print("全部完成！")

