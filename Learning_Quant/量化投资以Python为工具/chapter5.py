# Author:Zhang Yuan
print(3//4)
print( id(["a",3,True,"scd"][1])==id(3) )

print( (3==4) in [1,"345",3+4j,4 in [1,2,3]] )
#Python的比较链
#3 == 4 in [False] 相当于 (3 == 4) and (4 in [False])
print( 3==4 in [1,"345",3+4j,4 in [1,2,3]] )
print((3==4*4.5%2 is 0) in [3,4,"Tom","c" in "comic"])

print(min([1,4,5,0.2]) > 0.1)

import random
l=[random.normalvariate(0,1) for i in range(20)]
print(l)
print(max(l),min(l),sum(l))
l.sort()
print(l[0],l[-1],sum(l))

#python中固定的不可变的对象内存地址是一样的
a=3
b=3
print(a is b,a==b,id(a),id(b)) #True True 140716530557648 140716530557648
c=(3,)
d=(3,)
print(c is d,c==d,id(c),id(d)) #True True 2623132500824 2623132500824
class A:
    pass
e=A()
f=A()
print(e is f,e==f,id(e),id(f)) #False False 2623134376288 2623134314392

