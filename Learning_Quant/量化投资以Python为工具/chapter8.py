# Author:Zhang Yuan

#1
class bankcount:
    def __init__(self,input_name,input_balance):
        self.name=input_name
        self.__balance__=input_balance
    def SaveMoney(self,money):
        self.__balance__+=money
    def DrawMoney(self,money):
        if self.__balance__>=money:
            self.__balance__-=money
        else:
            print("your money is less than your Draw Money.")
    def GetBalance(self):
        return self.__balance__
    def ChangeBalance(self,money):
        self.__balance__=money
A=bankcount("Sam",1000)
A.SaveMoney(500)
A.DrawMoney(1200)
print(A.GetBalance())
A.ChangeBalance(10000)
print(A.GetBalance())

#2
class bankcount:
    def __init__(self,input_name,input_balance):
        self.name=input_name
        self.__balance__=input_balance
    def SaveMoney(self,money):
        self.__balance__+=money
    def DrawMoney(self,money):
        if self.__balance__>=money:
            self.__balance__-=money
        else:
            print("your money is less than your Draw Money.")
    def GetBalance(self):
        return self.__balance__
    def ChangeBalance(self,money):
        self.__balance__=money
    def Transfer(self,amount,TargetAcount):
        if amount>self.__balance__:
            print("your money is less than your Transfer Money.")
        else:
            self.__balance__-=amount
            TargetAcount.SaveMoney(amount)
A=bankcount("Sam",10000)
B=bankcount("Jony",1000)
A.Transfer(5000,B)
print(A.GetBalance(),B.GetBalance())

#3
class CreditcardCount(bankcount):
    def __init__(self,name,money,totalcredit):
        super().__init__(name,money)
        self.totalcredit=totalcredit
        self.overdraw=totalcredit
    def DrawMoney(self,amount):
        if self.__balance__<amount<= self.__balance__+self.overdraw:
            self.overdraw-=(amount-self.__balance__)
            self.__balance__=0
        elif amount<self.__balance__:
            self.__balance__-=amount
        else:
            print("Trading Error")

C=CreditcardCount("Sam",1000,1000)
C.DrawMoney(700)
print(C.GetBalance(),C.totalcredit,C.overdraw)
C.DrawMoney(500)
print(C.GetBalance(),C.totalcredit,C.overdraw)
C.DrawMoney(1500)
print(C.GetBalance(),C.totalcredit,C.overdraw)


