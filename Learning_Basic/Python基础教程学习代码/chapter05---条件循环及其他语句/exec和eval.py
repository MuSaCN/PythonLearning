# Author:Zhang Yuan
x=0
exec("x=1")
print(x)

Dict_Scope={}
exec("y=2",Dict_Scope)
print(Dict_Scope.keys())
exec("print(6*2)")
eval("print(2*2)")

Scope={}
Scope["x"]=2;
Scope["y"]=4;
exec("z=6",Scope)
print(Scope.keys())
eval("4*4")

