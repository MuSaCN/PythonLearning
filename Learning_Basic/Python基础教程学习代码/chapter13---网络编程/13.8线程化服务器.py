# Author:Zhang Yuan
#线程化服务器
import socketserver
class Server(socketserver.ThreadingMixIn,socketserver.TCPServer):
    pass
class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        addr=self.request.getpeername()
        print("Got connection from",addr)
        self.wfile.write("Thank you for connecting.")
server=Server(("",1234),Handler)
server.serve_forever()





