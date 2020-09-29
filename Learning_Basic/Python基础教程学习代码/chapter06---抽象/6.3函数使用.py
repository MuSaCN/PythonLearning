# Author:Zhang Yuan
#姓名储存与访问---------------------------------
# 数据结构：字典形式，一个名字会存储3次
# date{
#     "first":{
#         "A":["ABD","ACD",...]
#     }
# }
def initstore(store):
    "初始化数据结构"
    store["first"]={}
    store["second"]={}
    store["third"]={}
def accessname(store,lable,name):
    return store[lable].get(name)
def storename(store,fullname):
    "储存姓名"
    namelist=fullname.split()
    if len(namelist)==2:namelist.insert(1,"")
    lables=("first","second","third")
    for index,name in zip(lables,namelist):
        people=accessname(store,index,name)
        if people:
            people.append(fullname)
        else:
            store[index][name]=[fullname]

date={}
initstore(date)
storename(date,"A B C")
storename(date,"A C D")
storename(date,"B B C")
storename(date,"B C D")
storename(fullname="D D D",store=date) #关键字参数可以不考虑参数位置
print(accessname(date,"third","D"))

def storenamelist(store,*fullnamelist):
    for fullname in fullnamelist:
        namelist = fullname.split()
        if len(namelist) == 2: namelist.insert(1, "")
        lables = ("first", "second", "third")
        for index, name in zip(lables, namelist):
            people = accessname(store, index, name)
            if people:
                people.append(fullname)
            else:
                store[index][name] = [fullname]
date2={}
initstore(date2)
storenamelist(date2,"A B C","A C D","B B C","B C D","D D D")
print(accessname(date2,"third","D"))

