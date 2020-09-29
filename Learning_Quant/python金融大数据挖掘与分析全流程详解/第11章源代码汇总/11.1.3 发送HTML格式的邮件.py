# =============================================================================
# 11.1.3 发送HTML格式的邮件 by 华能信托-王宇韬
# =============================================================================

import smtplib
from email.mime.text import MIMEText
user = '你自己的qq号@qq.com'
pwd = '你自己的SMTP授权码'
to = '你自己设置的收件人邮箱'  # 可以设置多个收件人，英文逗号隔开，如：'***@qq.com, ***@163.com'

# 1.编写邮件正文内容
mail_msg = '''
<p>这个是一个常规段落</p>
<p><a href="https://www.baidu.com">这是一个包含链接的段落</a></p>
'''
msg = MIMEText(mail_msg, 'html', 'utf-8')

# 2.设置邮件主题、发件人、收件人
msg['Subject'] = '测试邮件主题!'
msg['From'] = user
msg['To'] = to

# 3.发送邮件
s = smtplib.SMTP_SSL('smtp.qq.com', 465)  # 选择qq邮箱服务，默认端口为465
s.login(user, pwd)  # 登录qq邮箱
s.send_message(msg)  # 发送邮件
s.quit()  # 退出邮箱服务
print('Success!')
