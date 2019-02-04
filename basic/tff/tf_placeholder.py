import tensorflow as tf

input1 = tf.placeholder(tf.float32)  # placeholder 要指定type
input2 = tf.placeholder(tf.float32)

# input2 = tf_note.placeholder(tf_note.float32,[]) #可以通过[] 指定input的结构

output=tf.multiply(input1,input2)

with tf.Session() as sess:

    res=sess.run(output,feed_dict={input1:[7.],input2:[2.]})
    print(res)

# placeholder 和 feed_dict 是绑定的



