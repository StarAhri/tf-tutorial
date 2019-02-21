import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每次批次的大小
batch_size = 100
# 批次的数量
n_batch = mnist.train.num_examples // batch_size



def variable_summaries(var):
    """
    Tensor summaries for exporting information about a model.
    summary 可以模型的一些参数导出。


    """
    with tf.name_scope("summaries"):
        mean=tf.reduce_mean(var)
        tf.summary.scalar("mean",mean)

        with tf.name_scope("stddev"):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar("stddev",stddev)
        tf.summary.scalar("max",tf.reduce_max(var))
        tf.summary.scalar("min",tf.reduce_min(var))
        tf.summary.histogram("histogram",var)


"""
使用 tensorboard 我们要给相应的层写在 命名空间 下面
并且相关的变量和参数要有名字

tensorboard -> graphs 查看神经网络结构

如果不定义命名空间 那么 tensorboard 会使用默认的变量名
并且各个层无法从主图中拆分
"""
with tf.name_scope("input"):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope("layer"):
    # 创建神经网络
    # 权值
    with tf.name_scope("wights"):
        W = tf.Variable(tf.zeros([784, 10]))
    # 偏置值
    with tf.name_scope("biases"):
        b = tf.Variable(tf.zeros([10]))
    with tf.name_scope("wx_plus_b"):
        wx_plus_b = tf.matmul(x, W) + b
    # softmax 作为激活函数
    with  tf.name_scope("softmax"):
        prediction = tf.nn.softmax(wx_plus_b)

# 损失函数
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(y - prediction))
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# prediction 预测的值
# y 数据原始标签值
# correct_prediction 是一个布尔型列表
# argmax 返回一维张量中最大的值所在的位置

with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

    # 求准确率  tf.cast 将布尔型转换成 float32 再求平均值
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)

    """
    设置 tensorboard 的路径
    不指定绝对路径，就是相对于当前目录的路径
    Terminal 中输入 tensorboard --logdir= logs/ 就会打开tensorboard

    """
    writer = tf.summary.FileWriter("logs/", sess.graph)

    # 训练21次  每个批次分别训练
    for epoch in range(1):
        for batch in range(n_batch):
            # mnist.train.next_batch 获得100个图片及其标签
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})

        print("Iter" + str(epoch) + "Test Accuracy" + str(acc))







