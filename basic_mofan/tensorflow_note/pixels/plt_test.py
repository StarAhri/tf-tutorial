import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# https://bbs.csdn.net/topics/392418592
x=np.linspace(-1,1,300)
print(x)
print("-----"*10)
noise = np.random.normal(0, 0.05, x.shape)
y = np.square(x) - 0.5 + noise
print(noise)
print("-----"*10)
print(y)

# fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x,y)
plt.show()
plt.ion()


