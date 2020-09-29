# Author:Zhang Yuan
#Tkinter是一个跨平台的Python GUI工具包
import tkinter
top=tkinter.Tk() #可以不写
btn=tkinter.Button()
#pack()控制从属控件在所属主体内部出现的位置
btn.pack()
btn["text"]="Click Me!"

def clicked():
    print("I was clicked!")
btn["command"]=clicked #不能是clicked()，要指向函数本身。

#下面的写法也可以
#btn.config(text="Click Again",command=clicked)

#如果没有指定主控件，则指向顶级主窗口
tkinter.Button(text="Click Again",command=clicked).pack()
tkinter.Label(text="I'm in the first window!").pack()

#建立主窗口外另一个窗口，并用label指向它
second=tkinter.Toplevel()
tkinter.Label(second,text="I'm in the second window!").pack()

for i in range(10):
    tkinter.Button(text=i).pack()

def callback(event):
    print(event.x,event.y)
#通过bind关联事件到top
top.bind("<Button-1>",callback)

tkinter.mainloop()
print("Finish")

