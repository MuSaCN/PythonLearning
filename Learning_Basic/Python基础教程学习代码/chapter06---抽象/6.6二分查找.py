# Author:Zhang Yuan
def BiSearch(list,number,lower=0,upper=None):
    "二分查找，使用递归"
    #default setting
    if(upper==None):upper=len(list)-1
    #search mode
    if lower==upper:
        if(number!=list[lower]):
            print("No This Number In Your List,return near one")
        #assert number==list[lower]#查找的如果不是number，则崩溃
        return lower
    else:
        middle=(lower+upper)//2
        if number>list[middle]:
            return BiSearch(list,number,middle+1,upper)
        else:
            return BiSearch(list,number,lower,middle)
date=[5,47,9,3,6,12,65,88,20]
date.sort()
print("sorted date: ",date)
index=BiSearch(date,7)
print(index)