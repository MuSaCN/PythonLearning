# Author:Zhang Yuan
items=[(1,2),(3,4),(5,6)]
print(dict(items))
#字典可直接添加
items2={}
items2["abc"]="bcd"
print(items2)

#找人的应用
people={
    "person0":{
        "Address":"China","Phone":"123"
    },
    "person1":{
        "Address":"England","Phone":"456"
    },
    "person2":{
        "Address":"USA","Phone":"789"
    }
}

name=input("Input the name you want to find:")
key=str(input("Input Address(a) or Phone(p):"))
if key=="a" : key="Address"
elif key=="p" : key="Phone"

if (name in people) :
    if (key=="Address" or key=="Phone"):
        print("{0}'s {1} is {2}".format(name,key,people[name][key])  )
    else:
        print("Invalid input,please input Address(a) or Phone(p).")
else:
    print("------Can not find this person------")