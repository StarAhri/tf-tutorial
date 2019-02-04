import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt



X=np.linspace(-1,1,200)
print(X)
np.random.shuffle(X)
print(X)
Y=0.5*X +2 + np.random.normal(0,0.05,(200,))
print(Y)



model=Sequential()
# Dense 全连接层
model.add(Dense(output_dim=1,input_dim=1))
# 上层的输出为下层的输入
model.add(Dense(output_dim=1))

# choose loss function and optimizing method
model.compile(loss='mse',optimizer='sgd')



print("------Training--------")

# for step in range(301):
