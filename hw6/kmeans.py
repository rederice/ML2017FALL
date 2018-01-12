import numpy as np
from sklearn.cluster import KMeans
from keras.models import load_model, Model
import sys
model = load_model("model.h5")
model.layers.pop()
model.layers.pop()
model.layers.pop()
model.layers.pop()
model.outputs = [model.layers[-1].output]
model.summary()
x_train = np.load(sys.argv[1])
x_train = x_train.astype('float32') / 255.
# mean = np.mean(x_train, axis=0)
# std = np.std(x_train, axis=0)
# x_train = (x_train-mean)/(std+1e-19)

reduce_dim = model.predict(x_train, verbose=1)
# np.save("reduce_dim", reduce_dim)

test = []
f = open(sys.argv[2], "r")
for line in f:
	tmp = line.split(',')
	if tmp[0] == 'ID':
		continue
	else:
		test.append([int(tmp[1]), int(tmp[2])])

label = KMeans(n_clusters=2, verbose=1).fit_predict(reduce_dim)

kmeans = np.load("label.npy")

with open(sys.argv[3],"w") as f:
	print("ID,Ans",file=f)
	for i in range(len(test)):
		if kmeans[test[i][0]] == kmeans[test[i][1]]:
			print("{},1".format(i),file=f)
		else:
			print("{},0".format(i),file=f)