# Author:Zhang Yuan
#整数类型的按位运算:二进制按位运算
print(2|3) #3        x 和 y 按位 或
print(2^3) #1        x 和 y 按位 异或 a⊕b = (¬a ∧ b) ∨ (a ∧¬b)
print(2&3) #2        x 和 y 按位 与
print(2<<3) #16      x 左移 n 位,相当于乘以多少次2
print(32>>3) #4      x 右移 n 位，相当于除以多少次2
print(~4) #-5        x 逐位取反，包括正负号, ~x 类似于 -x-1

a=[x for x in range(10)]
print(a.__iter__(),a.__iter__().__iter__())

lists=[[1]]*3 #[[1]] 是一个包含了一个列表的单元素列表，所以 [[1]] * 3 结果中的三个元素都是对这一个列表的引用。
print(lists) #[[1], [1], [1]]
lists[0].append(3)
print(lists) #[[1, 3], [1, 3], [1, 3]]

lists=[[1] for i in range(3)] #这种表达创建了3个列表，而不是引用同一个
print(lists) #[[1], [1], [1]]
lists[0].append(3)
print(lists) #[[1, 3], [1], [1]]

lists.extend(["a","b"])
print(lists) #[[1, 3], [1], [1], 'a', 'b']

print("asdgAAF".capitalize()) #Asdgaaf
print("asdgAAF".casefold()) #asdgaaf
print("asdgAAF".center(12))
print("asdgAAF".count("a"))
print("asdgAAF".endswith("F"))
print("asdgAAF".expandtabs(3))
print("asdgAAF".find("d"))
D={'name': 'Jack', 'sex': 'm', 'age': 22}
print('his name is {name}, his age is {age}'.format(**D))
print('his name is {name}, his age is {age}'.format_map(D))
print("asdgAAF".index("d"))
print("asd123gAAF".isalnum())
print("asd123gAAF".isalpha())
print("asd123gAAF".isascii())
print("asd123gAAF".isdecimal())
print("asd123gAAF".isdigit())
print("asd123gAAF".isidentifier())
print("asd123gAAF".islower())
print("asd123gAAF".isnumeric())
print("asd123gAAF".isprintable())
print("asd123gAAF".isspace())
print("asd123gAAF".istitle())
print("asd123gAAF".isupper())
print("abcdefg123456".join("WXYZ"))
print("abcdefg123456".ljust(20))
print("abcdefg123456".lower())

#字符串转换
s1 = "asdfghjkl"
s2 = "123456789"
s=str.maketrans(s1,s2)
str = '123456789'
print(str.translate(s))

print("abcdefg123456".partition("12"))
print("abcdefg123456".replace("12","987qwer"))
print("123abcdefg123456".rfind("12"))
print("123abcdefg123456".rindex("12"))
print("123abcdefg123456".rjust(20))
print('ab c\n\nde fg\rkl\r\n'.splitlines())
print('ab c\n\nde fg\rkl\r\n'.splitlines(keepends=True))
print("123abcdefg123456".startswith("123"))
print("123abcdefg123456".swapcase())
print("123abcdefg123456".upper())
print("123abcdefg123456".zfill(20))








