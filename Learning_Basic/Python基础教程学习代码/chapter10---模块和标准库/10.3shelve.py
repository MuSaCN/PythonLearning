# Author:Zhang Yuan
#shelve模块是一个简单的将内存数据通过文件持久化的模块
#shelve模块用于创建一个永久性映射
import shelve
#s得到一个shelf对象
try:
    s=shelve.open("test.csv",writeback=True)
    s["y"]=["a","b","c"]
    s["y"].append("d")
    info={'age':22,'job':'IT'}
    name=["ZhangYuan","Rain","Test"]
    s["name1"] =name
    s["info1"] = info
    print(dict(s))
finally: #防止数据库文件受损，在最后应该关闭它
    s.close()



