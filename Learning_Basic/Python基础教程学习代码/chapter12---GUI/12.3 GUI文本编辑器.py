# Author:Zhang Yuan

from tkinter import *
from tkinter.scrolledtext import ScrolledText
#---------------------
top=Tk()
top.title("simple editor")
contents=ScrolledText()
#pack()控制从属控件在所属主体内部出现的位置
contents.pack(side=BOTTOM,expand=True,fill=BOTH)
filename=Entry()
filename.pack(side=LEFT,expand=True,fill=X)
#----------------------
def load():
    with open(filename.get()) as file:
        contents.delete("1.0",END)
        contents.insert(INSERT,file.read())
def save():
    with open(filename.get(),"w") as file:
        file.write(contents.get("1.0",END))
#----------------------
Button(text="Open",command=load).pack(side=LEFT)
Button(text="Save",command=save).pack(side=LEFT)
mainloop()


