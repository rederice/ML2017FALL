import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.advanced_activations import *
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.utils import to_categorical
from keras import optimizers
import sys

data = pd.read_csv(sys.argv[1])
X=[]
for i in range(int(data.size/2)):
	X.append(np.array(data['feature'][i].split()))
X_all = np.array(X).reshape(-1,48,48,1).astype('float')
X_all /= 255
Y_all = data['label'].as_matrix().astype('int')

batch_size = 128
epochs = 500
classes=7
input_shape=(48,48,1)
mean,std = np.mean(X_all,axis=0), np.std(X_all,axis=0)
np.save('normalize.npy', [mean, std])
X_all = (X_all - mean)/(std + 1e-19)
X_train,X_valid = X_all[5000:],X_all[:5000]
Y_train,Y_valid = Y_all[5000:],Y_all[:5000]

X_train = np.concatenate((X_train, X_train[:, :, ::-1]), axis=0)
Y_train = np.concatenate((Y_train, Y_train), axis=0)

Y_train = to_categorical(Y_train, classes)
Y_valid = to_categorical(Y_valid, classes)

datagen = ImageDataGenerator(
            zoom_range=[0.8, 1.2],
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)

model = Sequential()
model.add(Conv2D(64,kernel_size=(5,5),input_shape=input_shape,padding='same',activation="selu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128,kernel_size=(3,3),input_shape=input_shape,padding='same',activation="selu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.35))

model.add(Conv2D(256,kernel_size=(3,3),input_shape=input_shape,padding='same',activation="selu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256,kernel_size=(3,3),input_shape=input_shape,padding='same',activation="selu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(512, activation='selu',kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='selu',kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax',kernel_initializer='glorot_normal'))
# ADAM = keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

callbacks=[]
callbacks.append(LambdaCallback(on_epoch_end=lambda batch, logs: print('\nEpoch[%d] Train-loss=%f Train-accuracy=%f Validation-loss=%f Validation-accuracy=%f' %(batch,logs['loss'], logs['acc'],logs['val_loss'],logs['val_acc']))))
callbacks.append(ModelCheckpoint('model/{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc',save_best_only=True,period=1))

model.fit_generator(
	datagen.flow(X_train,Y_train,batch_size=batch_size),
	verbose=0,
	steps_per_epoch=len(X_train)//batch_size,
	epochs=epochs,
	validation_data=(X_valid,Y_valid),
	callbacks=callbacks)
model.save("model.h5")