# Author:Zhang Yuan
def function(*args,**kwargs):
    print(args)
    print(kwargs)
function(1,2,"abc",a=1,b=2,c=3)

def func1(x,y,z):
    print(x+y+z)
var1=(1,2,3)
func1(*var1)

def func2(name,age):
    print(name+"'s age is "+age)
var2={"name":"ZhangYuan","age":"32"}
func2(**var2)

def foo(x,y,z,m=0,n=0):
    print(x,y,z,m,n)
def call_foo(*args,**kwargs):
    print("Calling foo!")
    foo(*args,**kwargs)
var3=(1,4,2)
var4={"n":7,"m":8}
call_foo(*var3,**var4)
var5=("a","b","c",7,3)
call_foo(*var5)

x=1
print(vars())




