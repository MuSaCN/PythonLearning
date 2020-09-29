# Author:Zhang Yuan

#raise Exception("This is Exception test")

try:
    x=int(20)
    y=int(2)
    print(x/y)
except Exception:
    print("The second number can not be zero")

class MuffledCalculator:
    muffled=False;
    def calc(self,expr):
        try:
            return eval(expr)
        except ZeroDivisionError:
            if self.muffled==True:
                print("Division by zero is illegal")
            else:
                raise
calculator=MuffledCalculator()
calculator.calc("10/2")
#calculator.calc("10/0")
calculator.muffled=True
calculator.calc("10/0")
print("--------------------------------------------")
# try:
#     1/0
# except ZeroDivisionError:
#     raise ValueError

# try:
#     x=int(20)
#     y=input("Input Number:")
#     print(x/y)
# except ZeroDivisionError:
#     print("The second number can not be zero")
# except TypeError:
#     print("that is not a number")

# try:
#     x=int(20)
#     y=input("Input Number:")
#     print(x/y)
# except (ZeroDivisionError,TypeError) as e:
#     print(e)

# try:
#     x=int(20)
#     y=input("Input Number:")
#     print(x/y)
# except:
#     print("something was wrong")

# try:
#     x=int(20)
#     y=int(input("Input Number:"))
#     print(x/y)
# except Exception as e:
#     print(e)

# try:
#     1/0
# except Exception as e:
#     print(e)
# else:
#     print("Run OK")
# finally:
#     print("Finished!")

def faulty():
    raise Exception("Something is wrong")
def ignore_faulty():
    faulty()
def handle_exception():
    try:
        faulty()
    except:
        print("Exception handled")
#ignore_faulty()
handle_exception()
