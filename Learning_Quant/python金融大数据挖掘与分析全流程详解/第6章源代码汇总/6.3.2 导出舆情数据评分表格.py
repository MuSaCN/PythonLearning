# =============================================================================
# 6.3.2 导出舆情数据评分 by 王宇韬&肖金鑫
# =============================================================================

# 假设已经获得了上一小节的分数字典，这里用一小部分数据来做演示
import pandas as pd
score_list = {'2018-09-01': 100, '2018-09-02': 100, '2018-09-03': 70, '2018-09-04': 75, '2018-12-01': 100}
data = pd.DataFrame.from_dict(score_list, orient='index', columns=['score'])
print(data)

# 此时获得了一个二维表格结构，不过此时的行索引为日期，如果想把行索引变成数字序号，日期作为单独的一列进行存储的话，这里有两种常规的处理方式：
# 方式一：通过重置索引的方式将行索引转换为列
data = pd.DataFrame.from_dict(score_list, orient='index', columns=['score'])
data.index.name = 'date'  # 将行索引那一列命名为date
data.reset_index()  # 重置索引
print(data)

# 方式二：将字典转换成列表，然后生成DataFrame
data = pd.DataFrame(list(score_list.items()), columns=['date', 'score'])
print(data)
data.to_excel('score_测试数据.xlsx', index=False)

'''下面的内容为真正实战所用的代码，先被我注释掉了，如果想研究的话，可以全选，然后取消注释'''
# import pymysql
# import datetime
# import pandas as pd
#
# # 生成时间列表
# date_list = pd.date_range('2018-09-01','2018-12-01') # 时间区间
# print(date_list)
# date_list = list(date_list)
#
# # 日期类型转换（Timestamp->str）
# for i in range(len(date_list)):
#     date_list[i] = datetime.datetime.strftime(date_list[i],'%Y-%m-%d')  # 类型转换
#
# # 读取数据库
# db = pymysql.connect(host='localhost', port=3306, user='root', password='', database='pachong', charset='utf8')  # 连接数据库
# cur = db.cursor()  # 获取会话指针，用来调用SQL语句
# company = '万科集团'
# sql = 'SELECT * FROM article WHERE company = %s AND date = %s'  # 编写SQL
#
# # 遍历date_list中的日期，获取每日分数并存储到字典score_list中
# score_list = {}    # 定义分数的字典，用以存储每日分数
# for d in date_list:
#     cur.execute(sql, (company, d))
#     data = cur.fetchall  # 提取所有数据并赋值给data变量
#     score = 100
#     for i in range(len(data)):
#         score += data[i][5]  # 合并该公司当天每条新闻分数
#     score_list[d] = score
# db.commit()  # 更新表单，如果对数据表没有修改，可以不写这行 
# cur.close()  # 关闭会话指针
# db.close()  # 关闭数据库连接
#
# # 将dic转换为DataFrame表格
# data = pd.DataFrame.from_dict(score_list, orient='index')
# data.index.name = 'date'  # 将行索引那一列命名为date
# data.reset_index()  # 重置索引
#
# # 保存数据
# data.to_excel('score_测试数据.xlsx', index=False)  # 存储到本地的score.xlsx
