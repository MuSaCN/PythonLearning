# =============================================================================
# 11.1.1 QQ邮箱发送邮件 by 华能信托-王宇韬
# =============================================================================

import smtplib  # 引入两个控制邮箱发送邮件的库
from email.mime.text import MIMEText

user = '你自己的qq号@qq.com'  # 发件人邮箱
pwd = '你自己的SMTP授权码'  # 邮箱的SMTP密码
to = '你自己设置的收件人邮箱'  # 可以设置多个收件人，英文逗号隔开，如：'***@qq.com, ***@163.com'

# 1.邮件正文内容
msg = MIMEText('测试邮件正文内容')

# 2.设置邮件主题、发件人、收件人
msg['Subject'] = '测试邮件主题!'  # 邮件的标题
msg['From'] = user  # 设置发件人
msg['To'] = to  # 设置收件人

# 3.发送邮件
s = smtplib.SMTP_SSL('smtp.qq.com', 465)  # 选择qq邮箱服务，默认端口为465
s.login(user, pwd)  # 登录qq邮箱
s.send_message(msg)  # 发送邮件
s.quit()  # 退出邮箱服务
print('Success!')