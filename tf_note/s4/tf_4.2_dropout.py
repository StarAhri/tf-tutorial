

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

# 每次批次的大小
batch_size=100
# 批次的数量
n_batch=mnist.train.num_examples//batch_size

# 定义两个placeholder
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
"""
设置 dropout 参数
"""
keep_prob=tf.placeholder(tf.float32)

# 创建神经网络

"""
truncated_normal 相对于 random_normal 是不会出现 2*stddev 以外的数据
全部初始化为0 并不是比较好的方式
"""
W1=tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
B1=tf.Variable(tf.zeros([2000])+0.1)
"""
双曲正切做激活函数
"""
L1=tf.nn.tanh(tf.matmul(x,W1)+B1)
"""
tf.nn.dropout 

使用dropout 模型的收敛速度回变慢

L1 某层神经元的输出
keep_prob 工作神经元占比 1 为100% 0.5 为 50%

"""
L1_drop=tf.nn.dropout(L1,keep_prob)

"""
多层隐藏层
"""
W2=tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
B2=tf.Variable(tf.zeros([2000])+0.1)
L2=tf.nn.tanh(tf.matmul(L1_drop,W2)+B2)
L2_drop=tf.nn.dropout(L2,keep_prob)

W3=tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
B3=tf.Variable(tf.zeros([1000])+0.1)
L3=tf.nn.tanh(tf.matmul(L2_drop,W3)+B3)
L3_drop=tf.nn.dropout(L3,keep_prob)


# 权值
W4=tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
# 偏置值
B4=tf.Variable(tf.zeros([10])+0.1)
# softmax 作为激活函数
prediction=tf.nn.softmax(tf.matmul(L3_drop,W4)+B4)

# 损失函数
#loss=tf.reduce_mean(tf.square(y-prediction))

# 重新定义loss
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)


# 初始化变量
init=tf.global_variables_initializer()

# prediction 预测的值
# y 数据原始标签值
# correct_prediction 是一个布尔型列表
# argmax 返回一维张量中最大的值所在的位置
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

# 求准确率  tf.cast 将布尔型转换成 float32 再求平均值
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    #训练21次  每个批次分别训练
    for epoch in range(31):
        for batch in range(n_batch):
            # mnist.train.next_batch 获得100个图片及其标签
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})

        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        train_acc=sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})

        print("Iter"+str(epoch)+"Test Accuracy"+str(test_acc)+"Train Accuracy"+str(train_acc))







