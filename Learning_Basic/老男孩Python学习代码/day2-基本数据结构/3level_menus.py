# Author:Zhang Yuan
data={
    "AnHui":{
        "HeFei":{
            "FeiXi":"肥西",
            "FeiDong":"肥东"
        },
        "LuAn":{
            "YuAn":"裕安",
            "JinAn":"金安"
        }
    },
    "ShangHai":{
        "PuDong":{
            "LuJiaZui":"陆家嘴"
        },
        "XuHui":{
            "XuJiaHui":"徐家汇"
        }
    },
}
print(data["AnHui"]["LuAn"]["YuAn"])

while True:
    for i in data:
        print(i)

    choice=input("Choice to get in 1:")
    if choice in data:
        while True:
            for j in data[choice]:
                print("\t",j)

            choice2=input("Choice to get in 2:")
            if choice2 in data[choice]:
                while True:
                    for k in data[choice][choice2]:
                        print("\t\t",k)

                    choice3=input("Choice to get in 3:")
                    if choice3 in data[choice][choice2]:
                        for l in data[choice][choice2][choice3]:
                            print("\t\t\t", l)
                        print("last get in,input b get out")
                    elif choice3=="b":
                        break
            elif choice2=="b":
                break
    elif choice=="b":
        break
