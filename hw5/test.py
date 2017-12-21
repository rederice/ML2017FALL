import numpy as np
import pandas as pd
from keras.layers import Input, Embedding, Dense, Dropout, Reshape, Flatten, BatchNormalization
from keras.layers.merge import concatenate, dot, add
from keras import backend as K
from keras.models import Model, load_model
import sys
# dim = 150
# norm = np.load("normalize.npy")
def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1., 5.)
    return K.sqrt(K.mean(K.square((y_true - y_pred))))

test = pd.read_csv(sys.argv[1])
model = load_model(sys.argv[3], custom_objects={'rmse': rmse})
pred = model.predict([np.array(test['UserID']-1),np.array(test['MovieID']-1)])

# pred = pred * norm[1] + norm[0]
with open(sys.argv[2],"w") as f:
	print("TestDataID,Rating",file=f)
	for i in range(pred.shape[0]):
		print("{},{}".format(i+1, pred[i][0]),file=f)