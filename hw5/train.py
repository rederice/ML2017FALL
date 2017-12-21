import numpy as np
import pandas as pd
from keras.layers import Input, Embedding, Dense, Dropout, Reshape, Flatten, BatchNormalization
from keras.layers.merge import concatenate, dot, add
from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LambdaCallback
from sklearn.utils import shuffle
dim = 150
def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1., 5.)
    return K.sqrt(K.mean(K.square((y_true - y_pred))))

train = pd.read_csv("train.csv")
train = shuffle(train)

print("-----user and movie counting-----")
n_users = train['UserID'].unique().max()
n_movies = train['MovieID'].unique().max()
X_user = train['UserID']-1
X_movie = train['MovieID']-1
Y_rating = train['Rating']

# mean, std = np.mean(Y_rating), np.std(Y_rating)
# np.save("normalize.npy", [mean, std])
# Y_rating = (Y_rating-mean)/(std+1e-19)
# np.save("normalize.npy", [mean, std])

print("-----building up model-----")
u_input = Input(shape=[1])
u = Embedding(n_users, dim)(u_input)
u = Reshape((dim,))(u)
u = Dropout(0.3)(u)
# u = BatchNormalization()(u)

m_input = Input(shape=[1])
m = Embedding(n_movies, dim)(m_input)
m = Reshape((dim,))(m)
m = Dropout(0.3)(m)
# m = BatchNormalization()(m)

u_bias = Embedding(n_users, 1)(u_input)
u_bias = Flatten()(u_bias)
m_bias = Embedding(n_movies, 1)(m_input)
m_bias = Flatten()(m_bias)

u_m = dot([u, m], -1)
u_m = add([u_m, u_bias, m_bias])

log_name = "model/"+str(dim)+"_log.txt"
flog = open(log_name,"w")
model = Model(inputs=[u_input, m_input], outputs=u_m)
model.summary()
callback = [ModelCheckpoint('model/dnn_output.h5',monitor='val_rmse', save_best_only=True),
            EarlyStopping(monitor='val_rmse', patience=1),
            LambdaCallback(on_epoch_end=lambda batch, logs: print(
        'Epoch[%d] Validation-rmse=%f Validation-loss=%f'
         %(batch, logs['val_rmse'], logs['val_loss']), file=flog))]
model.compile(loss='mse', optimizer='adam', metrics=[rmse])

model.fit([X_user, X_movie], Y_rating, batch_size=1024, epochs=666, validation_split=0.1, callbacks=callback)
