# Author:Zhang Yuan

import socket
client = socket.socket()
#client.connect(('192.168.16.200',9999))
client.connect(('localhost',9999))
while True:
    cmd = input(">>:").strip()
    if len(cmd) == 0: continue
    client.send(cmd.encode("utf-8"))
    ##第一次接受内容命令结果的长度，byte数据
    cmd_res_size = client.recv(128)
    print("命令结果大小:",cmd_res_size)
    received_size = 0
    received_data = b''
    TotalSize=int(cmd_res_size.decode()) #byte数据转成整数
    #开始接受内容
    while received_size < TotalSize :
        data = client.recv(128)
        # 每次收到的有可能小于1024，所以必须用len判断，返回的是byte长度
        received_size += len(data)
        print("接受内容的长度：",len(data))
        #print("接受的内容：",data) #分段打印有时候打印不出来，因为分段格式有时不完整
        received_data += data
    else:
        print("cmd res receive done...",received_size)
        print(received_data.decode())
client.close()
