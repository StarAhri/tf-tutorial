"""
RNN LSTM  循环神经网络，递归神经网络

用于 语音识别，自然语言处理，机器翻译

各个数据不是独立的，相互之间有关联，比如一段话，一段文字。
和 BP 神经网络输入多了一个反馈回路，反馈回路用于累计神经网络的计算结果。

梯度消失问题，之前的信息会随着传播不断衰减。


LSTM Long Short Term Memory 长短期记忆

Input gate 对输入数据进行筛选
Forget gate  确定衰减率
output gate 对输出数据进行筛选
LSTM 可以通过对三个 gate 的控制，来解决梯度消失的问题

"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""
依然是图片分类
"""

# 载入数据集
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)



"""
在 BP 神经网络中我们是将784个像素作为一个一维数据传入
在 RNN 中我们要分行传入
"""
# 输入一行，一行有28个数据
n_inputs=28
# 输入28行
max_time=28
# 隐藏层神经元数
lstm_size=100
# 图片分为0-9 共10类
n_classes=10
# 每次训练50个样本
batch_size=50
# 训练总批次
n_batch=mnist.train.num_examples//batch_size


# 同之前定义
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

# 初始化权值和偏置值
weights=tf.Variable(tf.truncated_normal([lstm_size,n_classes],stddev=0.1))
biases=tf.Variable(tf.constant(0.1,shape=[n_classes]))

# RNN 网络
def RNN(x,weights,biases):
    """

    :param x:
    :param weights:
    :param biases:
    :return:
    """
    # inputs=tf.reshape(x,[-1,max_time,n_inputs]
    """
    将 X 的格式进行转换 x 原来是[50,784]
    我们需要将 x 转化为 [50,28,28] 的
    -1 指的是弹性维度，在这个维度是弹性的，以保证正元素的总量不变
    """
    inputs=tf.reshape(x,[-1,max_time,n_inputs])

    # contrib 下的模块是惰性 import 模块
    # 只有真正使用的时候才会导入 IDE也不会有任何提示
    """
    定义 LSTM 的基本的 CELL,LSTM 中间的隐藏层
    
    

    """
    lstm_cell=tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(lstm_size)

    outputs,final_state=tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
    results=tf.nn.softmax(tf.matmul(final_state[1],weights)+biases)

    return results

# 通过 RNN 计算
prediction=RNN(x,weights,biases)
cross_entory=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entory)
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter "+str(epoch)+",Testing Accuracy= "+str(acc))







