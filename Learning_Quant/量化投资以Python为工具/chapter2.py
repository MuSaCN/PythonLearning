# Author: ZhangYuan


answer1=[i for i in range(10) if i%2==1]
print(answer1)

answer2=[i for i in range(20) if i%2==0]
print(answer2)

#this is practice from the book.

class A:
    @classmethod
    def a(cls):
        print("A")

    class AB:
        def ab(self):
            print("AB")
        def cd(self):
            A.a()
            print("CD")


aaa=A()
aaa.a()
aaa.AB().ab()
aaa.AB().cd()
