import tensorflow as tf

# 创建4个张量
a = tf.constant(1.)
b = tf.constant(2.)
c = tf.constant(3.)
w = tf.constant(4.)
"y = a * w**2 + b * w + c".split("=")
# 设置需要计算梯度的函数和变量，方便求解同时也提升效率。
# tf.GradientTape()自动微分的记录操作。
with tf.GradientTape() as tape:# 构建梯度环境
	tape.watch([w]) # 将w加入梯度跟踪列表
	# (必须写在里面)构建计算过程
	y = a * w**2 + b * w + c
# 默认情况下GradientTape的资源在调用gradient函数后就被释放，再次调用就无法计算了
[dy_dw] = tape.gradient(y, [w])
print(dy_dw)

