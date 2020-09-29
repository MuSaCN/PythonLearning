# Author:Zhang Yuan

def describe_person(person):
    print("name:",person["name"])
    print("age",person["age"])
    try:
        print("occupation",person["occupation"])
    except KeyError:
        pass

Person1={"name":"ZhangYuan","age":"32"}
describe_person(Person1)
Person2={"name":"ZhanXiang","age":"32","occupation":"IT"}
describe_person(Person2)

from warnings import warn
# warn("This is warn test")

from warnings import filterwarnings
filterwarnings("ignore")
warn("ABC")



