# Author:Zhang Yuan

name="my \tname is {name} and I am {year}"

print(name.capitalize())
print(name.count("a"))
print(name.center(50,"-"))
print(name.endswith("an"))
print(name.expandtabs(tabsize=30))
print(name[name.find("name"):])
print(name.format(name='ZhangYuan',year='30'))
print(name.format_map( {"name":"ZhanXiang","year":33} ))
print("AB123".isalnum())
print("AB123".isalpha())
print("1A".isdecimal())    #判断是否为十进制
print('A1'.isidentifier()) #判断是否为合法的标识符
print("33".isnumeric())
print("   ".isspace())
print("My Name Is".istitle())
print("My Name Is".isprintable()) #tty file,drive file
print("MY NAME".isupper())
print("+".join(["1","2","a"]))
print(name.ljust(50,'*'))
print(name.rjust(50,'*'))
print("ABCDE,ASF".lower())
print("ABCDE,ASF,asdfadsfg".upper())
print('  \nabcdasdf'.lstrip()) #从左边开始去除空格和回车
print('asdf   \n   '.rstrip())
print(' \n   asdf   \n   '.strip())
print("abcde".translate(str.maketrans("abcefg","123456")))
print('alex li'.replace("1",'L',1))
print('alex li'.rfind("l"))
print("123.132.321".split("."))
print('1+2\n+3+4'.splitlines())
print("Acd".swapcase())
print("fdsa erf df".title())
print('asdfdsa'.zfill(20))








