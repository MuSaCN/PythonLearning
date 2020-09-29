# =============================================================================
# 7.1.1 Tushare库的基本介绍 by 王宇韬&肖金鑫
# =============================================================================

# 1 获得日线行情数据
import tushare as ts
df = ts.get_hist_data('000002', start='2018-01-01', end='2019-01-31')
print(df)

# 2 获得分钟级别的数据
df = ts.get_hist_data('000002', ktype='5')
print(df)

# 3 获得实时行情数据
df = ts.get_realtime_quotes('000002')
print(df)
# 如果觉得列数过多，可以通过DataFrame选取列的方法选取相应的列，代码如下
df = df[['code', 'name', 'price', 'bid', 'ask', 'volume', 'amount', 'time']]
print(df)
# 获得多个股票代码的实时数据
df = ts.get_realtime_quotes(['000002', '000980', '000981'])
print(df)

# 4 获得分笔数据
df = ts.get_tick_data('000002', date='2018-12-12', src='tt')
print(df)
# 获取当日分笔信息
# df = ts.get_today_ticks('000002')  # 注意在非交易日无法用该代码
# print(df)

# 5 获得指数信息
df = ts.get_index()
print(df)
