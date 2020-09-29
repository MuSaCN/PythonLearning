# Author:Zhang Yuan
# class test:
#     name="ZhangYuan"
#     def show(self):
#         print(self.name)
# c1=test()
# c1.show()
# c1.name="ABCDEFG"
# c1.show()
# c2=test()
# c2.show()

# class test2:
#     count=0
#     def init(self):
#         self.count += 1
# cc1=test2()
# cc1.init()
# print(cc1.count)
# cc2=test2()
# cc2.init()
# print(cc2.count)

#注意以下可能犯错误的地方！！！！！！！！！！
class test3:
    count=0
    def init(self):
        test3.count += 1 #!!!注意此处不是self.count
#--------------------------------------------------#
cc3=test3()
cc3.init()
print(cc3.count)  #返回1
cc4=test3()
cc4.init()
print(cc3.count,cc4.count)  #返回2,2
cc4.count=6
cc4.init()
print(cc3.count,cc4.count)  #返回3,6
cc3.count=9
cc3.init()
print(cc3.count,cc4.count)  #返回9,6
