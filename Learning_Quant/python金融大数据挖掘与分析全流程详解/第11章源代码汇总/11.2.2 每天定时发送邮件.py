# =============================================================================
# 11.2.2 每天定时发送邮件 by 华能信托-王宇韬
# =============================================================================

import smtplib
from email.mime.text import MIMEText
import schedule
import pymysql
import time

user = '你自己的qq号@qq.com'
pwd = '你自己的SMTP授权码'
to = '你自己设置的收件人邮箱'  # 可以设置多个收件人，英文逗号隔开，如：'***@qq.com, ***@163.com'


def send_email():
    # 1.连接数据库 提取所有今天的"阿里巴巴"的新闻信息
    db = pymysql.connect(host='localhost', port=3306, user='root', password='', database='pachong', charset='utf8')
    company = '阿里巴巴'
    today = time.strftime("%Y-%m-%d")  # 这边采用标准格式的日期格式

    cur = db.cursor()  # 获取会话指针，用来调用SQL语句
    sql = 'SELECT * FROM test WHERE company = %s AND date = %s'  # 编写SQL语句
    cur.execute(sql, (company,today))  # 执行SQL语句
    data = cur.fetchall()  # 提取所有数据，并赋值给data变量
    print(data)
    db.commit()  # 这个其实可以不写，因为没有改变表结构
    cur.close()  # 关闭会话指针
    db.close()  # 关闭数据库链接

    # 2.设置一个可以添加正文和附件的msg
    mail_msg = []
    mail_msg.append('<p style="margin:0 auto">尊敬的小主，您好，以下是今天的舆情监控报告，望查阅：</p>')  # style="margin:0 auto"用来调节行间距
    mail_msg.append('<p style="margin:0 auto"><b>一、阿里巴巴舆情报告</b></p>')  # 加上<b>表示加粗
    for i in range(len(data)):
        href = '<p style="margin:0 auto"><a href="' + data[i][2] + '">' + str(i + 1) + '.' + data[i][1] + '</a></p>'
        mail_msg.append(href)

    mail_msg.append('<br>')  # <br>表示换行
    mail_msg.append('<p style="margin:0 auto">祝好</p>')
    mail_msg.append('<p style="margin:0 auto">华小智</p>')
    mail_msg = '\n'.join(mail_msg)
    print(mail_msg)

    # 3.添加正文内容
    msg = MIMEText(mail_msg, 'html', 'utf-8')

    # 4.设置邮件主题、发件人、收件人
    msg["Subject"] = "华小智舆情监控报告"
    msg["From"] = user
    msg["To"] = to

    # 5.发送邮件
    s = smtplib.SMTP_SSL('smtp.qq.com', 465)  # 选择qq邮箱服务，默认端口为465
    s.login(user, pwd)  # 登录qq邮箱
    s.send_message(msg)  # 发送邮件
    s.quit()  # 退出邮箱服务
    print('Success!')


send_email()  # 这个是用来演示的
# 建立一个schedule任务，每天下午5点钟执行指令
schedule.every().day.at("17:00").do(send_email)
while True:
    schedule.run_pending()
    time.sleep(10)

