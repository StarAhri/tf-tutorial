import tensorflow as tf

# tf.Variable 创建变量 stddev 标准差
# seed 随机种子 设置seed可以保证每次运行得到的结果是相同的
w1=tf.Variable(tf.random_normal((2,3),stddev=1,seed=1))
w2=tf.Variable(tf.random_normal((3,1),stddev=1,seed=1))

sess=tf.Session()
sess.run(w1.initializer)
sess.close()
x=tf.constant([[0.7,0.9]])

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

sess=tf.Session()

sess.run(w1.initializer)
sess.run(w2.initializer)

print(sess.run(y))
sess.close()