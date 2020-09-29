# Author:Zhang Yuan

#检测下一行的X坐标，与之前的一个状态是否冲突，Y坐标是固定的
def conflict(OnePreState,nextX):
    nextY=len(OnePreState)
    for i in range(nextY):
        if abs(OnePreState[i]-nextX) in (0,nextY-i):
            return True
    return False
#print(conflict((1,3,0),2))

#多种pre状态，与单独一个nextX，返回新状态
def conflictMore(PreState,nextX):
    CalState=[]
    for i in PreState:
        if not conflict(i,nextX):
            CalState.append(i+(nextX,))
    return CalState
morepre=[(1,3,0),(0,3,1),(3,1,0)]
# print(conflictMore(morepre,2))

#num个格子时，之前状态的推进
def MoreConflict(PreState,num):
    More=[]
    for i in range(num):
        More=More+conflictMore(PreState,i)
    return More

def queen(num):
    base=[(i,) for i in range(num)]
    for i in range(num-1):
        base=MoreConflict(base,num)
    return base
print(len(list(queen(8))))
print("----------------------------------------")

# def conflict(state,nextX):
#     nextY=len(state)
#     for i in range(nextY):
#         if abs(state[i]-nextX) in (0,nextY-i):
#             return True
#     return False
#
# def queens(num,state=()):
#     for pos in range(num):
#         bool_confict=conflict(state,pos)
#         if not bool_confict:
#             long=len(state)
#             if long==num-1:
#                 yield(pos,)
#             else:
#                 newstate=state+(pos,) #状态推进
#                 for result in queens(num,newstate):
#                     yield(pos,)+result
# print(list(queens(4,(1,))))

def prettyprint(solution):
    def line(pos,length=len(solution)):
        return "."*(pos)+"x"+"."*(length-pos-1)
    for pos in solution:
        print(line(pos))

import random
prettyprint(random.choice(list(queen(8))))

