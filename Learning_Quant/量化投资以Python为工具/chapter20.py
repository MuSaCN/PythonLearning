# Author:Zhang Yuan
import MyPackage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

#------------------------------------------------------------
__mypath__ = MyPackage.MyClass_Path.MyClass_Path("\\量化投资以Python为工具")  #路径类
myfile = MyPackage.MyClass_File.MyClass_File()  #文件操作类
myplt = MyPackage.MyClass_Plot.MyClass_Plot()  #直接绘图类(单个图窗)
myfig = MyPackage.MyClass_Plot.MyClass_Figure()  #对象式绘图类(可多个图窗)
mypltpro = MyPackage.MyClass_PlotPro.MyClass_PlotPro()  #Plot高级图系列
myfigpro = MyPackage.MyClass_PlotPro.MyClass_FigurePro()  #Figure高级图系列
mynp = MyPackage.MyClass_Array.MyClass_NumPy()  #多维数组类(整合Numpy)
mypd = MyPackage.MyClass_Array.MyClass_Pandas()  #矩阵数组类(整合Pandas)
mypdpro = MyPackage.MyClass_ArrayPro.MyClass_PandasPro()  #高级矩阵数组类
mytime = MyPackage.MyClass_Time.MyClass_Time()  #时间类
myDA = MyPackage.MyClass_DataAnalysis.MyClass_DataAnalysis()  #数据分析类
#MyPackage.MyClass_ToDefault.DefaultMatplotlibBackend()       #恢复默认设置(仅main主界面)
#------------------------------------------------------------
Path="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\数据及源代码\\020"
Path2="C:\\Users\\i2011\\OneDrive\\Book_Code&Data\\量化投资以python为工具\\习题解答"

# 8
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import statsmodels.formula.api as smf

rf = 1.036 ** (1 / 360) - 1

nyyh = web.DataReader('601288.SS', 'yahoo',dt.datetime(2014, 1, 1), dt.datetime(2014, 12, 31))
returns = (nyyh.Close - nyyh.Close.shift(1)) / nyyh.Close.shift(1)

indexcd = pd.read_csv(Path2+'/Part3/003/TRD_Index.csv', index_col='Trddt')
mktcd = indexcd[indexcd.Indexcd == 902]
mktret = pd.Series(mktcd.Retindex.values, index=pd.to_datetime(mktcd.index))
mktret.name = 'mktret'
mktret = mktret['2014-01-02':'2014']

dat = pd.concat([mktret, returns], 1)
dat = dat - rf


myDA.CAMP(dat.Close,dat.Close,Rf=0,DrawScatter=True)

model = smf.ols('Close~Close', data=dat).fit()
print(model.summary())

# 9.
from statsmodels.api import add_constant

rf = 1.036 ** (1 / 360) - 1

lsw = web.DataReader('300104.SZ', 'yahoo',dt.datetime(2014, 1, 1), dt.datetime(2014, 12, 31))

returns = (lsw.Close - lsw.Close.shift(1)) / lsw.Close.shift(1)

indexcd = pd.read_csv('Data/Part3/003/TRD_Index.csv', index_col='Trddt')
mktcd = indexcd[indexcd.Indexcd == 902]
mktret = pd.Series(mktcd.Retindex.values, index=pd.to_datetime(mktcd.index))
mktret.name = 'mktret'
mktret = mktret['2014-01-02':'2014']

dat = pd.concat([mktret, returns], 1)
dat = dat - rf

model = smf.ols('Close~mktret', data=dat).fit()
print(model.summary())

ret_2015 = pd.Series(mktcd.Retindex.values, index=pd.to_datetime(mktcd.index))['2015-01']
ret_2015.name = 'mktret'
ret_2015 = (ret_2015 - rf)
prediction = model.predict(add_constant(ret_2015), transform=False)

#10
import pypyodbc
import pandas as pd
import datetime as dt
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.api import add_constant

for line in open('Data/Part3/003/industryCodes.txt', encoding='utf-8'):
    listData = line.split(',')
    fileName = listData[0]
    listData[-1] = listData[-1][:6]
    if fileName == '银行':
        break

indexcd = pd.read_table('Data/Part3/003/TRD_Index.txt', sep='\t', index_col='Trddt')
mktcd = indexcd[indexcd.Indexcd == 902]
mktret = pd.Series(mktcd.Retindex.values, index=pd.to_datetime(mktcd.index))
mktret.name = 'mktret'


def GetStockAlpha(symbol, mktret):
    if symbol[0] == '0':
        i = 0
        while (symbol[i] == '0'):
            i += 1
        symbol = symbol[i:]
    accessName = 'Data/Part3/003/Stock.accdb'
    # 先提前安装Microsoft Access Database Engine
    # 下载地址为https://www.microsoft.com/zh-tw/download/details.aspx?id=13255
    conn = pypyodbc.connect(r'Driver=Microsoft Access Driver (*.mdb, *.accdb);DBQ=%s;' % accessName)
    cursor = conn.cursor()
    cursor.execute('select Stkcd, Trddt, Clsprc from stock where Stkcd=%s;' % symbol)
    data = cursor.fetchall()
    prices = []
    dates = []
    for entry in data:
        prices.append(entry[-1])
        time = str(entry[1])
        date = pd.to_datetime(time)
        dates.append(date)
    price = pd.Series(np.array(prices), index=dates)['2013']
    returns = (price - price.shift(1)) / price.shift(1)
    returns.name = 'stock'

    rf = 1.036 ** (1 / 360) - 1
    dat = pd.concat([mktret['2013-01-06':'2013'], returns['2013-01-06':'2013']], 1)
    dat = dat - rf
    model = smf.ols('stock~mktret', data=dat).fit()
    return (model.params[0])


alphas = {}
for symbol in listData[1:]:
    try:
        alphas[symbol] = GetStockAlpha(symbol, mktret)
    except Exception as e:
        print(Exception, ":", e)

selectAlpha = [('-00', -100000), ('-00', -100000), ('-00', -100000)]
for symbol in alphas.keys():
    if alphas[symbol] > selectAlpha[0][1]:
        selectAlpha[0] = (symbol, alphas[symbol])
        selectAlpha.sort(key=lambda d: d[1])
selectAlpha









