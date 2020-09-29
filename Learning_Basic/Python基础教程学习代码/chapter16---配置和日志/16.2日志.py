# Author:Zhang Yuan
import logging
logging.basicConfig(level=logging.INFO,filename="mylog.log")
logging.info("Start")
logging.info("Try to divide 1 by 0")
print(1/0)
logging.info("OK")
