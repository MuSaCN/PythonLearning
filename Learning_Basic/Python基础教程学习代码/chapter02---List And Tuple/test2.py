# Author:Zhang Yuan
database=[
    ["a1","1234"],["a2","2345"],["a3","3456"],["a4","4567"]
]

name=input("name:")
code=input("code:")

if [name,code] in database:
    print("OK")
else:
    print("wrong")



