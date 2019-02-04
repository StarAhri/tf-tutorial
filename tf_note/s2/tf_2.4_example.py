import tensorflow as tf
import numpy as np

# 用 numpy 生成100个随机点
x_data=np.random.rand(100)
y_data=x_data*0.1+0.2

# 构造一个线性模型
b=tf.Variable(0.)
k=tf.Variable(0.)
y=k*x_data+b

# 损失函数
loss=tf.reduce_mean(tf.square(y_data-y))
# y_data 和 x_data 都是确定的值
# b k 是两个变量，优化的时候会使用梯度下降法优化b和k，使这两个值不断的变化使loss越来越小

# 梯度下降法作为优化器
optimizer=tf.train.GradientDescentOptimizer(0.2)

# 使用优化器 optimizer 最小化损失函数（代价函数 loss）
# 表示我们训练的目的
train=optimizer.minimize(loss)

# 初始化变量
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 每次迭代都会 run(train)  train 会最小化loss
    for step in range(201):
        sess.run(train)
        if step%20==0:
            print(step,sess.run([k,b]))
