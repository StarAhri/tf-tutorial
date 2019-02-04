import tensorflow as tf
import numpy as np

# create data
x_data= np.random.rand(100).astype(np.float32)
y_data= x_data*0.1+0.3

### create tensorflow structure start ###

# 定义tf中的两个变量
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))  # random_uniform 随机均匀分布  Weights 是tf中一个从-1到1的随机的一个数
biases = tf.Variable(tf.zeros([1]))  # biases 是tf中一个为0的数

y=Weights*x_data+biases

loss=tf.reduce_mean(tf.square(y-y_data))    # 预测 y 和实际的 y 的差别
optimizer=tf.train.GradientDescentOptimizer(0.5)   # 选择optimizer 梯度下降法  参数是学习效率 要求小于1 目前是0.5
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()    # 初始化 Weights 和 biases 两个变量


### create tensorflow structure end ###

sess=tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train,)   # 每run一次 TensorFlow 会执行一次训练
    if step%20 ==0:
        print(step,sess.run(Weights),sess.run(biases))   # 输出 训练次数 以及训练的两个参数的值 Weights biases

# 2018.12.20 training
#   Weights       biases
# 0 [-0.05854616] [0.5487952]
# 20 [0.05032595] [0.3274698]
# 40 [0.08911934] [0.30601704]
# 60 [0.09761667] [0.301318]
# 80 [0.09947796] [0.3002887]
# 100 [0.09988565] [0.30006325]
# 120 [0.09997494] [0.30001387]
# 140 [0.0999945] [0.30000305]
# 160 [0.0999988] [0.30000067]
# 180 [0.09999974] [0.30000016]
# 每训练一次 Weights 和 biases 会更接近真实值



