
import tensorflow as tf
import numpy as np
y=tf.constant([1,2,3,4,5])
e=tf.argmax(y,1)
print(e)