# Author:Zhang Yuan
#json序列化：把内存数据变成字符串，只能处理简单基础的结构，json用于不同语言之间进行数据交互
import json,pickle
def sayhi(name):
    print("hello,",name)
info={
    "name":"ZhangYuan",
    "age":22
}
info1={
    "name":"ZhangYuan",
    "age":22,
    "func":sayhi
}
f=open("test.text","w")
print(json.dumps(info))
print(str(info))
#dumps可以执行很多次,但是不推荐。推荐只写一次，因为loads只能执行一次
f.write(json.dumps(info))
info["age"]=21
#f.write(json.dumps(info))


#f.write(pickle.dumps(info1))
#pickle.dump(info1,f) #f.write(pickle.dumps(info1))
f.close()

