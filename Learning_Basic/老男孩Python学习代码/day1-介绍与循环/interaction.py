# Author:Zhang Yuan

name = input("name:");
age = int(input("age:"))

print(type(age), type(str(age)))

job = input("job:");

info = '''
------- info of %s ------
Name:%s
Age:%d
Job:%s
''' % (name, name, age, job)

print("info=", info)

info2 = '''
------- info of {_name} ------
Name:{_name}
Age:{_age}
Job:{_job}
'''.format(_name=name, _age=age, _job=job)

print("info2=", info2)

info3 = '''
------- info of {0} ------
Name:{0}
Age:{1}
Job:{2}
'''.format(name, age, job)

print("info3=", info3);



