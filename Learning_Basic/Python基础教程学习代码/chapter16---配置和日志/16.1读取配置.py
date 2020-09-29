# Author:Zhang Yuan

#配置模块configparser
import configparser
File="config.ini"
config=configparser.ConfigParser()
#读取配置文件
config.read(File)
print(config["numbers"].get("pi"))
print(dict(config["numbers"]))



