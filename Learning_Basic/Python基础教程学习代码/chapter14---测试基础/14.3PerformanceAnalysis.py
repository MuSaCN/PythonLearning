# Author:Zhang Yuan
#运行时间分析
def add():
    count=0
    while count<=10000000:
        count+=1
    return
import cProfile
# cProfile.run("add()","AnalysisResult.txt")
#
# import pstats
# p=pstats.Stats("AnalysisResult.txt")
# print(p)
cProfile.run("add()")