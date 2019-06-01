import numpy as np
np.random.seed(444)

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

x = np.array([[5,0],
             [3,4],
             [0,5],
             [-4,-3],
              [0, -5]])

y = np.array([[25],[25],[25],[25],[25]])

model = Sequential()

model.add(Dense(2, input_dim=2))
model.add(Activation('softmax'))
model.add(Dense(1))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(x, y, batch_size=1, epochs=5000)

if __name__ == '__main__':
    print(model.predict(x))