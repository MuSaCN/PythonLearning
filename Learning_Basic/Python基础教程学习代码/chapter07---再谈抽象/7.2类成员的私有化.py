# Author:Zhang Yuan
#加入双下划线__,让成员私有化
class test_private:
    __var="ABC"
    def __print1(self):
        print("This is print1 with __, %s"%(self.__var))
    def print2(self):
        self.__print1()
        print("This is print2 without __, %s"%(self.__var))
c_private=test_private()
c_private.print2()
#但是以这种方式，依然可以访问私有化成员
c_private._test_private__print1()

#加入单下划线_
class test_private2:
    _var="DEF"
    def _print1(self):
        print("This is print1 with __, %s"%(self._var))
    def print2(self):
        self._print1()
        print("This is print2 without __, %s"%(self._var))
c=test_private2()
c.print2()





