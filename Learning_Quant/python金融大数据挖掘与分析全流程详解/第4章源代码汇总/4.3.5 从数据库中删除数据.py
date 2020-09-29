# =============================================================================
# 4.3.5 连接数据库并删除数据 by 王宇韬
# =============================================================================

import pymysql
db = pymysql.connect(host='localhost', port=3306, user='root', password='', database='pachong', charset='utf8')

company = '阿里巴巴'

cur = db.cursor()  # 获取会话指针，用来调用SQL语句
sql = 'DELETE FROM test WHERE company = %s'  # 编写SQL语句
cur.execute(sql, company)  # 执行SQL语句
db.commit()  # 因为改变了表结构，这一行必须要加
cur.close()  # 关闭会话指针
db.close()  # 关闭数据库链接
