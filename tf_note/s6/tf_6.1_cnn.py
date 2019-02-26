"""
传统神经网络存在的问题：
权值太多，计算量太大
权值太多，需要大量的样本进行训练

如果图片是100*100，那么输入层就需要有10000个神经元，当输入层和中间层各有10000个神经元，那么权值就会有1亿+
一般情况下，模型的复杂程度是根据样本的量来进行设计的。否则，容易出现过拟合和欠拟合的现象。
样本的大小应该在神经元参数的5-30倍

局部感受野
Local receptive field
"""



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

# 每个批次的大小
batch_size=100
# 批次的个数
n_batch=mnist.train.num_examples//batch_size


# 初始化权值 truncated_normal
def  weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)


# 初始化偏置 constant
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# 卷积层
def conv2d(x,W):

    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

# 池化层
def max_pool_2x2(x):

    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])


x_image=tf.reshape(x,[-1,28,28,1])




