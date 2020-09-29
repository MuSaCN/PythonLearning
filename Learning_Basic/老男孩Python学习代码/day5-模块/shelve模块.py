# Author:Zhang Yuan
#shelve模块是一个简单的将内存数据通过文件持久化的模块
import shelve
import datetime
d=shelve.open("shelve_test1")

info={'age':22,'job':'IT'}
name=["ZhangYuan","Rain","Test"]
d["name"] =name
d["info"] = info
d["date"]= datetime.datetime.now()


print(d.get("name"))
print(d.get("info"))
print(d.get("date"))

#防止数据库文件受损，最后需要关闭它
d.close()






