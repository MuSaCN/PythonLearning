# Author:Zhang Yuan
import time
def logger():
    time_format="%Y-%m-%d %X"
    time_current=time.strftime(time_format)
    with open("a.txt","a+") as f:
        f.write("%s end action\n" %time_current)

def test1():
    "......"
    print("In the Test1")
    logger()

def test2():
    "......"
    print("In the Test2")
    logger()

def test3():
    "......"
    print("In the Test3")
    logger()

test1()
test2()
test3()