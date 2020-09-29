# Author:Zhang Yuan
#反序列化，把字符串变成内存数据
import json,pickle
f=open("test.text","r")
data=json.loads(f.read()) #把data自动识别成字典，只能做简单规则读取
print(data)
print(data["name"])

# import pickle
# def sayhi(name):
#     print("hello2,",name)
# f = open("test.text","rb")
# #data = pickle.loads(f.read())
# data= pickle.load(f) #data = pickle.loads(f.read())
# print(data["func"]("ZhangYuan"))
