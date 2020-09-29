# Author:Zhang Yuan
#PS:列表List与List[:]值相同，但不是一个对象
words = ['cat', 'window', 'defenestrate']
for w in words:
    print(w, len(w))
#如果写成 for w in words:，这个示例就会创建无限长的列表，一次又一次重复地插入 defenestrate。
for w in words[:]:  # words与words[:]不是一个对象，值相同而已
    if len(w) > 6:
        words.insert(0, w)
print(words,words == words[:],words is words[:]) #['defenestrate', 'cat', 'window', 'defenestrate'] True False

A=[1,2,3,4]
# list.append(x)
# 将项目添加到列表的末尾。相当于 a[len(a):] = [x].
A.append(3) #[1, 2, 3, 4, 3]

# list.extend(L)
# 通过附加给定列表中的所有项目来扩展列表。相当于 a[len(a):] = L.
B=["a","b"]
A.extend(B) #[1, 2, 3, 4, 3, 'a', 'b']

# list.insert(i, x)
# 在给定位置插入项目。第一个参数为被插入元素的位置索引，因此 a.insert(0, x) 在列表头插入值， a.insert(len(a), x)相当于 a.append(x).
A.insert(1,5) #[1, 5, 2, 3, 4, 3, 'a', 'b']

# list.remove(x)
# 从列表中删除值为x的第一个项目。如果没有这样的项目是一个错误。
A.remove(3) #[1, 5, 2, 4, 3, 'a', 'b']

# list.pop([i])
# 删除列表中给定位置的项目，并返回。如果没有给定位置，a.pop()将会删除并返回列表中的最后一个元素。（方法声明中i周围的方括号表示参数是可选的，而不是您应在该位置键入方括号。您将在Python库参考中频繁地看到此符号。）
A.pop() #[1, 5, 2, 4, 3, 'a']

# list.index(x)
# 返回值为x的第一个项目的列表中的索引。如果没有这样的项目是一个错误。
print(A.index(4)) #3

# list.count(x)
# 返回x出现在列表中的次数。
print(A.count(5)) #1

# list.sort(key=None, reverse=False)
# 排序列表中的项 (参数可被自定义, 参看 sorted() ).

# list.reverse()
# 列表中的元素按位置反转。
A.reverse() #['a', 3, 4, 2, 5, 1]

# list.copy()
# 返回列表的浅副本。相当于 a[:].

# list.clear()
# 从列表中删除所有项目。相当于 del a[:].
A.clear() #[]

#列表作为栈使用效率好（后进先出“last-in，first-out”）。
stack = [3, 4, 5]
stack.append(6) #
stack.pop()

#列表作为队列使用(“先进先出”)，单纯使用效率低。
#若要实现一个队列， collections.deque 被设计用于快速地从两端操作
from collections import deque
queue=deque([1,2,"a","b","c"]) #双队列
print(queue)
queue.append("Terry") #右端添加
print(queue.popleft()) #最端出

#列表推导式---------------------------------------------------------------
#注意这里创建（或被重写）的名为 x 的变量在for循环后仍然存在。
squares = []
for x in range(10):
    squares.append(x**2)
print(squares,x)
#列表推导式：一个表达式，后面跟一个 for 子句，然后是零个或多个 for 或 if 子句。
squares2 = [y**3 for y in squares]
print(squares2)
vec = [[1,2,3], [4,5,6], [7,8,9]]
print([num for elem in vec for num in elem])
#嵌套的列表推导式，一层层表述
matrix=[[i+4*j for i in range(4)] for j in range(3)]
print(matrix) #[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
matrix2=[[row[i] for row in matrix] for i in range(4)]
print(matrix2) #[[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]
#PS:
#注意下面的区别：单独的列表，相当于各成员与空内容zip成元组
print(list(zip(matrix))) #[([0, 1, 2, 3],), ([4, 5, 6, 7],), ([8, 9, 10, 11],)]
#注意下面的区别：加*让列表的各个部分独立开来，然后再zip成元组
print(list(zip(*matrix))) #[(0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11)]


#元组
empty = ()
singleton = 'hello', #最好写成("hello",)
print(len(empty))   #0
print(len(singleton))   #1

#花括号或 set() 函数可以用来创建集合。注意：要创建一个空集合你只能用 set() 而不能用 {}，因为后者是创建一个空字典
a=set()
print(a)
#类似于 列表推导式，集合也支持推导式形式
a = {x for x in 'abracadabra' if x not in 'abc'} #{'r', 'd'}

#字典推导式可以从任意的键值表达式中创建字典
d={x: x ** 2 for x in (2, 4, 6)} #{2: 4, 4: 16, 6: 36}
#字典一个Key只能对应一个值，所以下面的4会被5覆盖
D={a:b for a in "ABCDEF" for b in (4,5)} #{'A': 5, 'B': 5, 'C': 5, 'D': 5, 'E': 5, 'F': 5}


#当在字典中循环时，用 items() 方法可将关键字和对应的值同时取出
knights = {'gallahad': 'the pure', 'robin': 'the brave'}
for k, v in knights.items():
    print(k, v)

#当在序列中循环时，用 enumerate() 函数可以将索引位置和其对应的值同时取出
for i, v in enumerate(['tic', 'tac', 'toe']):
    print(i, v)

#当同时在两个或更多序列中循环时，可以用 zip() 函数将其内元素一一匹配。
questions = ['name', 'quest', 'favorite color']
answers = ['lancelot', 'the holy grail', 'blue']
for q, a in zip(questions, answers):
    print('What is your {0}?  It is {1}.'.format(q, a))

#当逆向循环一个序列时，先正向定位序列，然后调用 reversed() 函数
for i in reversed(range(1, 10, 2)):
    print(i)

#如果要按某个指定顺序循环一个序列，可以用 sorted() 函数，它可以在不改动原序列的基础上返回一个新的排好序的序列
basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
for f in sorted(set(basket)):
    print(f)

#布尔运算符 and 和 or 也被称为 短路 运算符
string1, string2, string3 = '', 'Trondheim', 'Hammer Dance'
non_null = string1 or string2 or string3 #'Trondheim'

#序列对象可以与相同类型的其他对象比较。它们使用 字典顺序 进行比较：首先比较两个序列的第一个元素，再比较每个序列的第二个元素，以此类推.
#字典顺序对字符串来说，是使用单字符的 Unicode 码的顺序
print(  (1, 2, ('aa', 'ab'))   <  (1, 2, ('abc', 'a'), 4)  )  #True






