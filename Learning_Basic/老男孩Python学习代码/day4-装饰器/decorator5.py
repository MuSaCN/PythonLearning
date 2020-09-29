# Author:Zhang Yuan
import time
user,passwd="ZhangYuan","ABCDE"

def auth(auth_type):
    print("auth func:",auth_type)
    def outer_wrapper(func):
        def wrapper(*args, **kwargs):
            print("wrapper",*args,**kwargs)
            if auth_type=="local":
                username = input("Username:").strip()
                password = input("Password:").strip()
                if user == username and passwd == password:
                    print("授权通过")
                    res = func(*args, **kwargs)
                    return res
                else:
                    exit("Invalid username or password")
            elif auth_type=="ldap":
                print("ldap模式测试")
        return wrapper
    return outer_wrapper


def index():
    print("welcome to index page")

@auth(auth_type="local")
def home():
    print("welcome to home page")
    return "From Home"

@auth(auth_type="ldap")
def bbs():
    print("welcome to bbs page")

index()
print(home())
bbs()



