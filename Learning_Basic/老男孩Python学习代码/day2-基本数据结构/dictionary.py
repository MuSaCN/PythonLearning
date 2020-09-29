# Author:Zhang Yuan
#key-value
info={
    'stu1101':'A1',
    'stu1102':'A2',
    'stu1103':'A3',
}
info2={
    'stu1101':'B1',
    '123':'456'
}
info.update(info2)#相同索引覆盖，新索引添加
print(info)
print(info['stu1101'])
print(info.get('stu1105')) #这种方式不会出错
print(info.get('stu1102'))

info['stu1101']='B1'
info['stu1104']='Add'
print(info)

#delete
del info['stu1101']
info.pop('stu1104')
print(info)

#dictionary嵌套
av_catalog={
    "Euro":{
        "index1":["www.youporn.com","Free,General Quality"],
        "index2":["www.pornhub.com","Free,High Quality"],
        "index3":["letmedothistoyou.com","self"],
        "index4":["x-art.com","Very High Quality,Not Free"]
    },
    "Japan":{
        "index1":["tokyo-hot"]
    },
    "China":{
        "index1":["1024"]
    }
}
av_catalog["China"]["index1"][0]='ABCDE'
print(av_catalog.values())
print(av_catalog["Euro"].keys())

av_catalog.setdefault("TaiWan",{"index1":["www.baidu.com"]}) #setdefault()当没有时创建，当有时不变
av_catalog.setdefault("China",{"index1":["www.baidu.com"]}) #setdefault()当没有时创建，当有时不变
print(av_catalog)

print(info)
print(info.items())

c=dict.fromkeys([6,7,8],[1,{"index1":"abc"},444])#共同指向一个地址
print(c)
c[7][1]['index1']='def'#修改一个则全部修改
print(c)

for i in av_catalog:
    print(i,av_catalog[i])

#字典转列表，如果数据量大要很久
for k,v in av_catalog.items():
    print(k,v)












