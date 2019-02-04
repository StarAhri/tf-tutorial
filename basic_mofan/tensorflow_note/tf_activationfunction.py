import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt


# 整个神经网络可以概括为 y=Wx y=AF(Wx) AF就是激励函数
# AF 本质是一个非线性方程  将原来的线性结果改变为非线性结果
# AF 一般有 relu sigmoid tanh 函数 AF 要求必须是可微分的  在反向传递的时候需要
# 加入一个神经层
def add_layer(inputs, in_size, out_size, activation_function=None):
    """
    设置神经网络层
    :param inputs: 输入数据
    :param in_size:  输入数据大小
    :param out_size: 输出数据格式
    :param activation_function:  激励函数
    :return: output
    """
    with tf.name_scope('layer'):
        # 定义 tf_note 变量
        # Weights 该层网络的参数
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 定义矩阵大写

            # 数据可视化
            tf.his
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # biases 全部为0.1 不推荐为0

        # 定义该层网络
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            # 激励函数
            outputs = activation_function(Wx_plus_b)

            # relu 是线性整流函数  x<=0 y=0 x>0 y=x
        return outputs

# 构造数据

# linspace 创建线性等分量（等差数列），起始值为-1，终止值为1，数列个数为300
# np.newaxis 增加一个维度，原维度的数据将变为新维度的第一列的数据
# 原数据是一个长度为300的数组  np.newaxis 后为 300行1 列的数组
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 定义噪点 模拟真实数据
noise = np.random.normal(0, 0.05, x_data.shape)  # 均值为0 标准差0.05 与x_data 格式相同
# 定义输出
y_data = np.square(x_data) - 0.5 + noise

# tf_note.float32 定义 placeholder 的类型。 [None,1] 定义 shape
# def placeholder(dtype, shape=None, name=None)
with tf.name_scope("input"):
    xs = tf.placeholder(tf.float32,[None, 1],name='x_input')
    ys = tf.placeholder(tf.float32,[None, 1],name='y_input')

# 定义神经网络输入层和隐藏层第一层  一个输入 10个神经元
# 10个神经元的输出 经过 relu 激励函数校正 xs 为输入 xs 为 placeholder
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# 定义神经网络输出层  10个神经元的输入  1个输出 l1 为输入  l1 为生一层的输出
prediction = add_layer(l1, 10, 1, activation_function=None)

# 定义损失函数  ys为真实输出值  prediction 为神经网络的输出值
# 神经网络的输出值 第一次的输出是纯随机的 后面会逐渐校正
# reduce 会记录训练的数据 计算总的均值  这样的loss是总体样本的loss 而非ys-prediction个体的
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))

with tf.name_scope("train"):

    # 定义 optimizer  优化模型，采用学习效率为0.1的梯度下降优化模型 让损失函数的值最小化
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 学习效率 要小于1


# 初始化所有的tf变量
init = tf.initialize_all_variables()
sess = tf.Session()

# tensorboard --logdir='logs/'  神经网络结构可视化
writer=tf.summary.FileWriter('logs/',sess.graph)

sess.run(init)


# ---------- 与tf无关  用于绘图----------
# 利用 plot 显示真实数据
# 创建一个画布
fig=plt.figure()
# 创建图形格式
ax=fig.add_subplot(1,1,1)
# scatter 用于绘制散点图
ax.scatter(x_data,y_data)
# 开启交互模式，可以连续绘制
# plt.isinteractive() 确认是否开启交互模式
# plt.ioff()  关闭交互模式
plt.ion()

# 展示图片
plt.show()



# 训练1000次
for i in range(1000):

    # 训练一次  启动 train_step 会把你定义的整个的训练过程链式的启动起来
    # feed_dict 是 placeholder 的传入值
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

    # 每训练50次 展示一次训练结果
    if i % 50 ==0:


        # 输出总体的损失函数，所有的 tf_note 值得查看都要在 sess.run 中查看
        # 为什么在此还要传入 feed_dict  惰性实现有关？？
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))

        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        # 输出预测值，为了绘制下面的曲线
        prediction_value=sess.run(prediction,feed_dict={xs:x_data})

        # 绘制一条由预测值连接成的红色的线
        lines=ax.plot(x_data,prediction_value,'r-',lw=5)
        # ax.lines.remove(lines[0])
        # 暂停0.1 秒  为什么？？
        plt.pause(0.1)


sess.close()
