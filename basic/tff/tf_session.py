import tensorflow as tf


matrix1=tf.constant([[3,3]])  # 定义 tf_note 中的常量  一行，两列 值均为3 的矩阵
matrix2=tf.constant([[2],[2]])  # 两行 一列的 值均为2 的矩阵

product =tf.matmul(matrix1,matrix2 )  # matrix multiply  矩阵乘法

# session 的使用方法一
sess= tf.Session()   #Session 是一个类 要大写
result=sess.run(product)   # 每run 一次 tf_note 会执行一次预定义的结构

print(result)
sess.close()

# session 的使用方法二   python 上下文
with tf.Session() as sess:
    result2=sess.run(product)
    print(result2)

