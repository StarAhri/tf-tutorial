
import tensorflow as tf

# 定义变量
state=tf.Variable(0,name="counter")  # 给定变量初始值 0 并定义变量名 name
# print(state.name)
one = tf.constant(1) # 定义常量 1

# 定义动作
new_value=tf.add(state,one )   # 变量加常量还是变量
update=tf.assign(state,new_value)  # 把 new_value 赋值给 state


init=tf.initialize_all_variables()  # 所有的 tf变量 必须要初始化

with tf.Session() as sess:
    sess.run(init)     # 初始 tf 变量
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
        # print(state)   直接 print state 是没有用的  输出变量必须要在 sess.run() 里面