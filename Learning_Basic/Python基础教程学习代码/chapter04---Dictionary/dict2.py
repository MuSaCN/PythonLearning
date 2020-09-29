# Author:Zhang Yuan

content={"index0":"ZhangYuan","index1":"ZhanXiang"}
print("Hello {index0}".format_map(content))

x={}
y=x
x["index0"]="TEST"
x={}
print(y)

#方法get的应用
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
name=input("Input name:")
request=input("Address(a) or Phone(p)?")
key=request
if request=="a":key="Address"
if request=="p":key="Phone"
person=people.get(name,{})
result=person.get(key,"not available")
print("{}'s {} is {}".format(name,key,result))


