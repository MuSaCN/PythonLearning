# Author:Zhang Yuan
def Fibs(num):
    "Calculate Fibonacci Series"
    base=[0,1]
    if num==0 or num==1:return None
    if num==1 :return [0]
    for i in range(num-2):
        base.append(base[i]+base[i+1])
        #更简单
        #base.append(base[-1]+base[-2])
    return base

print(Fibs(9))
print(Fibs.__doc__)
help(Fibs)


