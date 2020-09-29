#python中循环都要加冒号:，不同于C++
#python中的缩进有意义，不同于C++
#continue,break用法与C++相同
count=0

#while
while count<3:     #要加冒号:
    print(count)
    count+=1        #python中没有count++这个语法
else:                   #要加冒号:
    print("OK")

#for
for i in range(3):
    print(i)
else:
    print(i,"OK")