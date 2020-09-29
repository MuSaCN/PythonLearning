# Author:Zhang Yuan
#参数为可变参数时
def func_list(list):
    list[0]="ABC"
def func_dict(dict):
    dict.setdefault("A","TEST")

test_list=[5,3,2,1]
test_dict={"B":"123"}

func_list(test_list)
# 这相当于：list=test_list; list[0]="ABC"

func_dict(test_dict)
# 这相当于：dict=test_dict; dict.setdefault("A","TEST")

print(test_list,test_dict)

#如果是不可变参数string、数、tuple，虽然也指向内存，但是无法改变
#所以函数的不可变参数不改变外面


