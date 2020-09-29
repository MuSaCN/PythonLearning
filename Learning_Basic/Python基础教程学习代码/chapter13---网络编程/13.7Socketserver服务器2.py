# Author:Zhang Yuan
import socketserver
#可以支持多客户端同时处理
class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            try:
                self.data = self.request.recv(1024).strip()
                print("{} wrote:".format(self.client_address[0]))
                print(self.data)
                self.request.send(self.data.upper())
            except ConnectionResetError as e:
                print("err",e)
                break
if __name__ == "__main__":
    HOST, PORT = "localhost", 9999
    # Create the server, binding to localhost on port 9999
    server = socketserver.ThreadingTCPServer((HOST, PORT), MyTCPHandler) #支持并发
    #server = socketserver.TCPServer((HOST, PORT), MyTCPHandler) #这个不支持多并发，只是简单服务器
    server.serve_forever()