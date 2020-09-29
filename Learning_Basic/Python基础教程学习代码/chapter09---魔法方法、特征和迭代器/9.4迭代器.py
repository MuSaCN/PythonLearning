# Author:Zhang Yuan

class Fibs:
    def __init__(self):
        self.a=0
        self.b=1
    def __next__(self):
        self.a,self.b=self.b,self.a+self.b
        return self.a
    def __iter__(self):
        return self
cf=Fibs()
date=[]
for f in cf:
    date.append(f)
    if f>100:
        break
print(date)

it=iter(range(10))
print(it.__next__())
print(it.__next__())
print(it.__next__())
print(next(it))
print(next(it))

class TestIterator:
    def __init__(self):
        self.value=0
    def __next__(self):
        self.value+=1
        if self.value>10: raise StopIteration
        return self.value
    def __iter__(self):
        return self
TI=TestIterator()
print(list(TI))





