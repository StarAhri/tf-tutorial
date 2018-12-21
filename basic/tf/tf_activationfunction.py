import tensorflow as tf
import numpy as np


# 整个神经网络可以概括为 y=Wx y=AF(Wx) AF就是激励函数
# AF 本质是一个非线性方程  将原来的线性结果改变为非线性结果
# AF 一般有 relu sigmoid tanh 函数 AF 要求必须是可微分的  在反向传递的时候需要


# 加入一个神经层
def add_layer(inputs, in_size, out_size, activation_function=None):
    """

    :param inputs:
    :param in_size:
    :param out_size:
    :param activation_function:  激励函数
    :return:
    """
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 定义矩阵大写
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # biases 全部为0.1 不推荐为0
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# linspace 创建线性等分量（等差数列），起始值为-1，终止值为1，数列个数为300
# np.newaxis 增加一个维度，原维度的数据将变为新维度的第一列的数据
# 原数据是一个长度为300的数组  np.newaxis 后为 300行1 列的数组
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 定义噪点 模拟真实数据
noise = np.random.normal(0, 0.05, x_data.shape)  # 均值为0 标准差0.05 与x_data 格式相同
# 定义输出
y_data = np.square(x_data) - 0.5 + noise

# tf.float32 定义 placeholder 的类型。 [None,1] 定义 shape  def placeholder(dtype, shape=None, name=None)
#
xs = tf.placeholder(tf.float32,[None, 1])
ys = tf.placeholder(tf.float32,[None, 1])

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 学习效率 要小于1

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

    if i % 50 ==0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))


sess.close()
