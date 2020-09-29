# Author:Zhang Yuan
"第三库自动记录和升级"
# ipython或cmd可直接运行
# 检查安装哪些第三方库和版本
'pip list'
# 生成到指定目录
'pip freeze > "C:\\Users\\i2011\\OneDrive\\Work-Python_backups\\site-packages_record.txt"'
# 根据指定目录安装第三方库
'pip install -r "C:\\Users\\i2011\\OneDrive\\Work-Python_backups\\site-packages_record.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple'

# ---cmd方式自动执行
# 批量生成
'''
@echo off
set /p op="Note that non-host operations overwrite the file: yes/no"
if "%op%" == "yes" (
pip list
pip freeze > "C:\\Users\\i2011\\OneDrive\\Work-Python_backups\\site-packages_record.txt"
) 
taskkill /f /im cmd.exe
exit
'''
# 批量安装
'''
@echo off
pip list
pip install -r "C:\\Users\\i2011\\OneDrive\\Work-Python_backups\\site-packages_record.txt"  -i https://pypi.tuna.tsinghua.edu.cn/simple
taskkill /f /im cmd.exe
exit
'''
# anaconda批量安装
'''
@echo off
set condaRoot=C:\Users\i2011\Anaconda3
call %condaRoot%\Scripts\activate.bat
pip install -r "C:\\Users\\i2011\\OneDrive\\Work-Python_backups\\site-packages_record.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
set /p op="Update finished, Check message and exit: any input"
taskkill /f /im cmd.exe
exit
'''
