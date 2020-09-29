# =============================================================================
# 11.1.4 发送邮件附件 by 华能信托-王宇韬
# =============================================================================

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
user = '你自己的qq号@qq.com'
pwd = '你自己的SMTP授权码'
to = '你自己设置的收件人邮箱'  # 可以设置多个收件人，英文逗号隔开，如：'***@qq.com, ***@163.com'

# 1.设置一个可以添加正文和附件的msg
msg = MIMEMultipart()

# 2.先添加正文内容，设置HTML格式的邮件正文内容
mail_msg = '''
<p>这个是一个常规段落</p>
<p><a href="https://www.baidu.com">这是一个包含链接的段落</a></p>
'''
msg.attach(MIMEText(mail_msg, 'html', 'utf-8'))

# 3.再添加附件，这里的文件名可以有中文，但下面第三行的filename不可以为中文
att1 = MIMEText(open('E:\\test.docx', 'rb').read(), 'base64', 'utf-8')
att1["Content-Type"] = 'application/octet-stream'
# 下面的filename是在邮件中显示的名字及后缀名, 这边的文件名可以和之前不同，但不可以为中文！！
att1["Content-Disposition"] = 'attachment; filename="test.docx"'
msg.attach(att1)




# 4.设置邮件主题、发件人、收件人
msg['Subject'] = '测试邮件主题!'
msg['From'] = user
msg['To'] = to

# 5.发送邮件
s = smtplib.SMTP_SSL('smtp.qq.com', 465)
s.login(user, pwd)
s.send_message(msg)  # 发送邮件
s.quit()
print('Success!')