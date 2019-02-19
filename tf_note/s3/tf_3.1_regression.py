
"""
非线性回归的示例
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# -0.5 到 0.5 的 200 个元素的等差数列
# np.newaxis 增加一个维度 数据变为200行 一列
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]

# 生成符合正态分布的随机值
noise=np.random.normal(0,0.02,x_data.shape)
# y=x^2 并不是线性回归！
y_data=np.square(x_data)+noise

# 定义两个placeholder
# [None,1] 表示行不确定，列是一列。根据我们定义的样本定义的。
# x 要传入神经网络
# y 要传入损失函数
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

# 定义神经网络的中间层

# 定义权值
# 十个节点的中间层 [1,10]  1 表示的 x 输入 10 表示的节点的个数
Weights_L1=tf.Variable(tf.random_normal([1,10]))
biases_L1=tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1=tf.matmul(x,Weights_L1)+biases_L1

# 激活函数
L1=tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络的输出层
# 将十个中间层的节点规约为单个值
Weights_L2=tf.Variable(tf.random_normal([10,1]))
biases_L2=tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2=tf.matmul(L1,Weights_L2)+biases_L2

# 激活函数
prediction=tf.nn.tanh(Wx_plus_b_L2)

# 二次代价函数
# reduce_mean 平均值
loss=tf.reduce_mean(tf.square(y-prediction))

# 使用梯度下降法 是 loss 的值最小
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    # 获得预测值
    # prediction_value 的格式是和 x_data 一样的
    prediction_value=sess.run(prediction,feed_dict={x:x_data})

    print(x_data)
    print(prediction_value)

    plt.figure()
    # 画散点图 原始数据
    plt.scatter(x_data,y_data)
    # 画线 训练数据
    plt.plot(x_data,prediction_value,'r-')
    plt.show()


