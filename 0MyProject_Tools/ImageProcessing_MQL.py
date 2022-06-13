# Author:Zhang Yuan
# ------------------使用说明------------------------------------------------
# 放到桌面的 abc文件夹 下
# 文件命名规则：
#   logo原始文件命名规则：abc_logo.png --> abc_logo_*_DEMO.png / abc_logo_*_Paid.png
#   screenshot原始文件(无_logo标识符)命名规则：abc.png --> abc_640_480.png
# logo图片添加文字：
#   IndicatorName = [*] / myImage.textOnImage() 为logo核心文本输入内容，每次使用都不同，需要修改。

#%%
from MyPackage.MyPath import MyClass_Path
from MyPackage.MyMql import MyClass_ImageMql

mypath = MyClass_Path()  # 路径类
myIMql = MyClass_ImageMql() # MQL产品图片处理类

# ---设置桌面为默认的image目录
filepath = mypath.get_desktop_path()+"\\abc"

# ---需要添加到logo图片的指标字样
# IndicatorName = ["AC","Alligator","AO","BWMFI","Fractals","Gator","ATR","BearsPower","BullsPower","CCI","Chaikin","DeMarker","Force","MACD","Momentum","OsMA","RSI","RSV","Stochastic","TriX","WPR","AD","MFI","OBV","Volumes","ADX","ADXWilder","AMA","Bands","DEMA","Envelopes","FrAMA","Ichimoku","MA","SAR","StdDev","TEMA","VIDyA"]  # ***每次需修改***
IndicatorName = ["MT5","MT4"]
IndicatorName = ["MT5"]

myIMql.__init__(filepath)
myIMql.screen_shot()
myIMql.logo(indicatorName=IndicatorName,y=150,size=28)

