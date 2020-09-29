# Author:Zhang Yuan
#基于Socketserver的极简服务器
#可以支持多客户端同时处理
from socketserver import TCPServer,StreamRequestHandler
class Handler(StreamRequestHandler):
    def handle(self):
        addr=self.request.getpeername()
        print("Got Connection From",addr)
        self.wfile.write("Thanks for connecting")
serve=TCPServer(('',1234),Handler)
serve.serve_forever()


