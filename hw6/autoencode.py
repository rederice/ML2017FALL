from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback
import numpy as np

x_train = np.load("image.npy")

input_img = Input(shape=(784,))
encoded = Dense(392, activation='relu')(input_img)
encoded = Dense(196, activation='relu')(encoded)
encoded = Dense(98, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(98, activation='relu')(encoded)
decoded = Dense(196, activation='relu')(decoded)
decoded = Dense(392, activation='relu')(decoded)
decoded = Dense(784, activation='relu')(decoded)
autoencoder = Model(input_img, decoded)
callback = [ModelCheckpoint('model/auto.h5',monitor='mse', save_best_only=False)]
autoencoder.compile(optimizer='adam', loss='mse')

x_train = x_train.astype('float32') / 255.
# mean = np.mean(x_train, axis=0)
# std = np.std(x_train, axis=0)
# x_train = (x_train-mean)/(std+1e-19)

autoencoder.fit(x_train, x_train, epochs=300, batch_size=128, callbacks=callback)