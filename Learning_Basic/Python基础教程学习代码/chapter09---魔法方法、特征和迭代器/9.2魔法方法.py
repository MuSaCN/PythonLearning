# Author:Zhang Yuan

#检查key是否为int，同时要大于0
def check_index(key):
    if not isinstance(key,int): raise TypeError
    if key<0:raise IndexError

class ArithmeticSequence:
    def __init__(self,start=0,step=1):
        self.start=start
        self.step=step
        self.changed={}
    def __getitem__(self,key): #让实例可以直接访问A[index]
        check_index(key)
        try:
            return self.changed[key]
        except KeyError:
            return self.start+key*self.step
    def __setitem__(self, key, value): #让实例可以直接赋值A[index]=...
        check_index(key)
        self.changed[key]=value

ac=ArithmeticSequence(1,2)
print(ac[4])

class Tag:
    def __init__(self):
        self.change = {'python': 'This is python',
                       'php': 'PHP is a good language'}

    def __getitem__(self, item):
        print('调用getitem')
        return self.change[item]

    def __setitem__(self, key, value):
        print('调用setitem')
        self.change[key] = value

a = Tag()
print(a['php'])
a['php'] = 'PHP is not a good language'
print(a['php'])
print("-----------------------------------")

class CounterList(list):
    def __init__(self,*args):
        super().__init__(*args)
        self.count=0
    def __getitem__(self, index):
        self.count+=1
        return super().__getitem__(index)
C1=CounterList(range(10))
print(C1)
C1.reverse()
print(C1)
del C1[3:6]
print(C1)
print(C1.count)
print(C1[4],C1[2])
print(C1.count)

