import tensorflow as tf

x=tf.Variable([1,2])
a=tf.Variable([3,3])

sub=tf.subtract(x,a)
add=tf.add(x,sub)

# 定义变量初始化操作
init=tf.global_variables_initializer()

with tf.Session() as sess:
    # 开始进行变量初始化
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

# 创建一个变量，初始化为0，并且给这个变量命名
state=tf.Variable(0,name='counter')
# 创建一个operator，作用是使state加1
new_value=tf.add(state,1)
# 赋值 operator ，将 new_value 赋值给 state
# 在 tf中 不能用= 来进行赋值，必须调用assign方法
update=tf.assign(state,new_value)

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # 打印变量的值
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))

