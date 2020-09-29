# =============================================================================
# 11.1.2 网易163邮箱发送邮件 by 华能信托-王宇韬
# =============================================================================

import smtplib
from email.mime.text import MIMEText
user = '你自己的163邮箱@163.com'  # 发件人，这里为163邮箱了
pwd = 'huaxiaozhi123'  # 163邮箱的SMTP授权码
to = '收件人邮箱'  # 可以设置多个收件人，英文逗号隔开，如：'***@qq.com, ***@163.com'

# 1.邮件正文内容
msg = MIMEText('测试邮件正文内容')

# 2.设置邮件主题、发件人、收件人
msg['Subject'] = '测试邮件主题!'
msg['From'] = user
msg['To'] = to

# 3.发送邮件
s = smtplib.SMTP_SSL('smtp.163.com', 465)  # 选择163邮箱服务，默认端口为465
s.login(user, pwd)  # 登录163邮箱
s.send_message(msg)  # 发送邮件
s.quit()
print('Success!')