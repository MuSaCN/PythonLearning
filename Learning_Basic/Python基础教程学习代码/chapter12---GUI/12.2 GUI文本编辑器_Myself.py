# Author:Zhang Yuan
#Tkinter是一个跨平台的Python GUI工具包
import tkinter,tkinter.scrolledtext
#----------------------------------------
Top=tkinter.Tk()
Top.title("simple editor")
TextContents=tkinter.scrolledtext.ScrolledText()
#pack()控制从属控件在所属主体内部出现的位置
TextContents.pack(side=tkinter.BOTTOM,expand=True,fill=tkinter.BOTH)
FileName=tkinter.Entry()
FileName.pack(side=tkinter.LEFT,expand=True,fill=tkinter.X)
#----------------------------------------
def load(filename,contents):
    with open(filename.get()) as file:
        contents.delete("1.0",tkinter.END)
        contents.insert(tkinter.INSERT,file.read())
def save(filename,contents):
    with open(filename.get(),"w") as file:
        file.write(contents.get("1.0",tkinter.END))
#-----------------------------------------
def buttonLoad():
    load(FileName,TextContents)
def buttonSave():
    save(FileName,TextContents)
tkinter.Button(text="Open",command=buttonLoad).pack(side=tkinter.LEFT)
tkinter.Button(text="Save",command=buttonSave).pack(side=tkinter.LEFT)
#-----------------------------------------
tkinter.mainloop()