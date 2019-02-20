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

# 创建神经网络

# 权值
W=tf.Variable(tf.zeros([784,10]))
# 偏置值
b=tf.Variable(tf.zeros([10]))
# softmax 作为激活函数
prediction=tf.nn.softmax(tf.matmul(x,W)+b)

# 损失函数
loss=tf.reduce_mean(tf.square(y-prediction))
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
    for epoch in range(21):
        for batch in range(n_batch):
            # mnist.train.next_batch 获得100个图片及其标签
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})

        print("Iter"+str(epoch)+"Test Accuracy"+str(acc))







