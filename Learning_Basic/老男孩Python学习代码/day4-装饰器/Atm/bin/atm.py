# Author:Zhang Yuan
print(__file__) #相对路劲
import os,sys
Place=os.path.abspath(__file__) #绝对路劲
print(Place)
UpPlace=os.path.dirname(Place)
print(UpPlace)
BASE_DIR=os.path.dirname(UpPlace)
print(BASE_DIR)

sys.path.append(BASE_DIR)
from conf import settings
from core import main
