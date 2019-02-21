import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

# 每次批次的大小
batch_size=100
# 批次的数量
n_batch=mnist.train.num_examples//batch_size

# 定义两个placeholder
"""
# 因为上面批次我们定义了100  这里x 和 y 的 None 在训练中实际上是100
# X 是一个长度为784的一维向量，表示手写数字的向量，里面的值介于0-1 之间 0代表纯白 1代表纯黑
# 784 是图片的像素个数 

# Y 是一个长度为10的一维向量，表示图像的标签，MINST 数据集的标签是介于0-9之间的数字
# 标签被转化为 one-hot vectors 形式 ，即除了对应数字位的是1，其余位全部为0 的长度为10的一维向量
# 
"""
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

# 创建神经网络

# 权值
W=tf.Variable(tf.zeros([784,10]))
# 偏置值
b=tf.Variable(tf.zeros([10]))
# softmax 作为激活函数
"""
softmax 常被用来作为经典分类问题的最后一层
来给出被归为某类的概率
因为我们如果不加 softmax ，我们神经网络最后一层会得到一个 长度为10的 一维向量，里面保存的值也是云里雾里
我们强制通过 softmax 将里面的值转化为概率的形式，再通过 Optimizer 使我们的模型的输出结果向标签对齐

softmax 输出结果 是一个长度为10的一维向量，里面值代表图像被推断为某个值的概率，10值之和为1

"""

"""
prediction 每次循环的结果是一个100*10的二维矩阵
"""
prediction=tf.nn.softmax(tf.matmul(x,W)+b)

# 损失函数
"""
这里的损失函数是二次代价函数 会存在训练收敛速度的问题

y 是一个100*10的二维矩阵，图像标签数字对应的位被置1，其他的为零
prediction 也是一个100*10的二维矩阵，对应位置表示相应数字的概率

tf.square 会求出每个元素的平方
矩阵差的平方 reduce_mean 会输出二维元素的均值，类型是兼容的
"""
loss=tf.reduce_mean(tf.square(y-prediction))
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)


# 初始化变量
init=tf.global_variables_initializer()

# prediction 预测的值
# y 数据原始标签值

"""
argmax 返回一维张量中最大的值的索引 1 表示是的 axis
    y 和 prediction 是 100*10 的二维矩阵
    argmax 返回的长度为100的向量
tf.equal 会判断两个张量对应元素是否相等
    所以 correct_prediction 是一个布尔型长度为100的向量
"""
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

"""
求准确率
tf.cast 强制类型转换 将布尔型转换成 float32 
tf.reduce_mean 求均值 因为 true 就是1  false 就是0 

"""
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    #训练21次  每个批次分别训练
    for epoch in range(21):
        for batch in range(n_batch):
            # mnist.train.next_batch 获得100个图片及其标签
            """
            batch_xs batch_ys 分别是 100*784 100*10 的矩阵
            """
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})

        print("Iter"+str(epoch)+"Test Accuracy"+str(acc))







