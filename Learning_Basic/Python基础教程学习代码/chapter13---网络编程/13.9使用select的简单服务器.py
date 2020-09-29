# Author:Zhang Yuan

import socket,select
s=socket.socket()
host=socket.gethostname()
port=1234
s.bind((host,port))
s.listen(5)
Inputs=[s]
while True:
    rs,ws,es=select.select(Inputs,[],[])
    for r in rs:
        if r is s:
            conn,addr=s.accept()
            print("Got Connection From",addr)
            Inputs.append(conn)
    else:
        try:
            data=r.recv(1024)
            disconnected=not data
        except socket.error:
            disconnected=True
        if disconnected:
            print(r.getpeername(),"disconnected")
            Inputs.remove(r)
        else:
            print(data)






