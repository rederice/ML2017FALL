import numpy as np
import pandas as pd
from keras.models import load_model
import sys

width = height = 48
data = pd.read_csv(sys.argv[1])
X=[]
for i in range(int(data.size/2)):
	X.append(np.array(data['feature'][i].split()))
X_all = np.array(X).reshape(-1,48,48,1).astype('float')
X_all /= 255

norm = np.load("normalize.npy")
X_all = (X_all-norm[0])/(norm[1]+1e-19)

model = load_model("hw3_model.h5")
pred = model.predict(X_all)
pred = np.argmax(pred, axis=-1)

with open(sys.argv[2], 'w') as f:
        print('id,label', file=f)
        print('\n'.join(['{},{}'.format(i, p) for (i, p) in enumerate(pred)]), file=f)