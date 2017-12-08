import numpy as np
from gensim.models import Word2Vec
from gensim.utils import tokenize
from keras.models import Sequential, load_model
from keras.layers import LSTM, Embedding, Dropout, Dense, Activation
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
import sys
MAX_SEQUENCE_LENGTH = 40

mm = Word2Vec.load("word_dict.h5")

word_dict = {"_PAD": 0}
vocab = [(k, mm.wv[k]) for k, v in mm.wv.vocab.items()]
for i in range(len(vocab)):
	w = vocab[i][0]
	word_dict[w] = i+1

x_test=[]
with open(sys.argv[1],encoding='utf-8') as f:
	for line in f.readlines():
		tmp = line.split(",",1)
		if tmp[0] == "id":
			continue
		st = ""
		for i in range(len(tmp[1])):
			if i <= len(tmp[1])-3:
				if tmp[1][i] != tmp[1][i+1] or tmp[1][i] != tmp[1][i+2]:
					st += tmp[1][i]
			else:
				st += tmp[1][i]
		x_test.append(st.strip('\r\n').split(' '))
for c in range(len(x_test)):
	x_test[c] += ["_PAD"] * (MAX_SEQUENCE_LENGTH-len(x_test[c]))

shape = (len(x_test), MAX_SEQUENCE_LENGTH)
oao = np.zeros(shape)
for i in range(len(x_test)):
	for j in range(len(x_test[i])):
		if x_test[i][j] in word_dict:
			oao[i][j] = int(word_dict[x_test[i][j]])

# LOAD MODEL BY NAME
model = load_model("model.h5")
print("-----ready to predict-----")
pred = model.predict(oao,batch_size=256)
# SAVE OUTPUT OF NAME
with open(sys.argv[2], 'w') as f:
    print('id,label', file=f)
    for i in range(pred.shape[0]):
    	if pred[i] >= 0.5:
    		print("%d,%d"%(i,1),file=f)
    	else:
    		print("%d,%d"%(i,0),file=f)