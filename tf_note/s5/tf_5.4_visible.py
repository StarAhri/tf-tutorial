

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

# mnist
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

# 运行次数
max_step=1001
# 图片数量
image_num=3000

# 文件路径
DIR="./"

# 会话
sess=tf.Session()

# 载入图片
"""
tf.stack
Packs the list of tensors in `values` into a tensor with rank one higher than
each tensor in `values`, by packing them along the `axis` dimension.
有点像 Python 的 zip 函数

  x = tf.constant([1, 4])
  y = tf.constant([2, 5])
  z = tf.constant([3, 6])
  tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)  3行2列
  tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]                    2行3列
  

"""

"""
tf.stack(mnist.test.imgaes[:image_num])  将 mnist 的测试集的前3000张图片打包成一个 3000行 784列的张量
trainable=False 此变量不会被训练
"""
embedding=tf.Variable(tf.stack(mnist.test.imgaes[:image_num]),trainable=False,name="embedding")

# SUMMARY

def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean=tf.reduce_mean(var)
        tf.summary.scalar("mean",mean)  #平均值
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean))) #标准差
        tf.summary.scalar("stddev",stddev)
        tf.summary.scalar("max",tf.reduce_max(var)) # 最大值
        tf.summary.scalar("min",tf.reduce_min(var)) # 最小值
        tf.summary.histogram("histogram",var) # 直方图

# input 命名空间
with tf.name_scope("input"):
    # None 表示可以是任意长度
    x=tf.placeholder(tf.float32,[None,784],name='x-input')
    # 正确的标签
    y=tf.placeholder(tf.float32,[None,10],name='y_input')

with tf.name_scope("input_reshape"):
    """
    tf.reshape 会将输入 x 变形
    """
    image_shaped_input=tf.reshape(x,[-1,28,28,1])

    """
    把 image 图片传进去，这样summary 就可以显示了
    """
    tf.summary.image("input",image_shaped_input,10)



for  i in range(max_step):
    batch_xs,batch_ys=mnist.train.next_batch(100)