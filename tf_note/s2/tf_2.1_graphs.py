import tensorflow as tf

# 创建常量 tensor
m1=tf.constant([[3,3]])
m2=tf.constant([[2],[3]])

print(m1)
print(m2)
# Tensor("Const:0", shape=(1, 2), dtype=int32)
# Tensor("Const_1:0", shape=(2, 1), dtype=int32)

# 创建一个矩阵乘法op，把m1和m2传入
product=tf.matmul(m1,m2)
print(product)
# Tensor("MatMul:0", shape=(1, 1), dtype=int32)

# 定义一个会话 Session ，Session会启动默认的图
sess=tf.Session()

# 调用sesss的run方法来执行矩阵乘法op
result=sess.run(product)
print(result)
sess.close()

# 通过上下文来启动Session
with tf.Session() as sess:
    result=sess.run(product)
    print(result)

