# Author:Zhang Yuan
#服务器与客户端建立方式不同
#网络连接的主要操作就是接受recv和发送send，python3里网络传输只能用byte
#recv(128)表示一次传输的最大容量，如果传输的内容过大，需要多次recv(128)才可以接受完。即使过大，每次不一定按照128进行最大传输
import socket
s=socket.socket() #建立一个套接字网络单位
#网络连接的host名称和端口需要一致才可以
host=socket.gethostname()
port=1234
s.bind((host,port)) #bind绑定，说明这是服务器
#最多监听的客户端数量
s.listen(5)
while True:
    #一个网络单位(服务器)可以有多个客户端连接
    c,addr=s.accept() #这是客户端连接时才执行，返回连接和地址
    data = c.recv(1024)  #服务器连接后接受到的客户端内容
    print("server get data:",data)
    print("Got connection from",addr)
    #服务器两次发送信息，客户端也需要两次接受
    c.send("Thank you for connecting".encode()) #str需要转成比特
    c.send(data) #发送接受到的信息，原本就是byte
    c.close()




