# Author:Zhang Yuan
#断言，自定义中断
assert (1==1),"ABC"
#
words=["A","B","C","D","E"]
for word in words:
    print(word)
print(word)
for i in range(10):
    print(i)
#zip捆绑两个序列
words2=["a","b","c","d","e"]
print(list(zip(words,words2)))
for i,j in zip(words,words2):
    print(i,j)
for i,j in list(zip(words,words2)):
    print(i,j)
