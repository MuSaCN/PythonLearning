# Author:Zhang Yuan

# datelist=[[1,2,3],[4,5],[6]]
# def generator(date):
#     for i in date:
#         for j in i:
#             yield j
# #generator(datelist)生成器是以迭代的形式推进数据，而不是数据全部处理好一个一个调用。大数据时能省内存
# for ii in generator(datelist):
#     print(ii, generator(datelist))
#     for j in generator(datelist):
#         print("j",j)
#
# g=(i*i for i in range(10))#简单的生成器推导
# k=[i*i for i in range(10)]
# print(g,k)

nested=[[1,2,3],[4,5],[6],7,[8,9],[10,[11,[12]]]]
def flatten(nested):
    try:
        for s in nested:
            for e in flatten(s):
                yield e
    except TypeError:
        yield nested
for jj in flatten(nested):
    print(jj)
print("--------------------------------------")

nested1=[0,[1,2,3],"abc",["ABC",["DEF"]]]
def flatten1(nested):
    try:
        try:nested+""
        except TypeError:pass
        else:raise TypeError
        for s in nested:
            for e in flatten1(s):
                print("e={},s={},nested={}".format(e,s,nested))
                yield e
    except TypeError:
        yield nested
for kk in flatten1(nested1):
    print("print:",kk)

def repeater(value):
    while True:
        new=(yield value)
        if new is not None:
            print("not None")
            value=new
            print(value)
r=repeater(12)

print(next(r),r.__next__())
r.send("ABC")

#模拟生成器
def SimulateGenerator(multilist):
    result=[]
    try:
        try:multilist+""
        except:pass
        else:raise TypeError
        for sublist in multilist:
            for element in SimulateGenerator(sublist):
                result.append(element)
    except TypeError:
        result.append(multilist)
    return result
a=SimulateGenerator(nested1)
print(a)


nested2=[0,[1,2,3],"abc",["ABC",["DEF"]]]
def flatten2(nested):
    try:
        if isinstance(nested,str):
            raise TypeError
        for s in nested:
            for e in flatten2(s):
                print("e={},s={},nested={}".format(e,s,nested))
                yield e
    except TypeError:
        yield nested
for kk in flatten2(nested2):
    print("print:",kk)