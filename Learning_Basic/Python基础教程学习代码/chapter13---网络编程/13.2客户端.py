# Author:Zhang Yuan

import socket
s=socket.socket() #建立一个套接字网络单位
# host=socket.gethostname()
# port=1234
host=socket.gethostname()
port=1234
#s网络单位进行connect连接，说明这是客户端
s.connect((host,port))
#向服务器发送信息，需要变成byte
s.send("This is client".encode())
#由于客户端发送了两次信息，需要两次接受
print(s.recv(1024).decode()) #把比特数据转成str
print(s.recv(1024).decode()) #把比特数据转成str
print(s.recv(1024).decode()) #把比特数据转成str
print(s.recv(1024).decode()) #把比特数据转成str