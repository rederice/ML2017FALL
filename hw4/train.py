import numpy as np
from gensim.models import Word2Vec
from gensim.utils import tokenize
from keras.models import Sequential
from keras.layers import GRU, Embedding, Dropout, Dense, Activation, Bidirectional
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
import sys
MAX_SEQUENCE_LENGTH = 40
train = []
Y_all = []
www = []
s = []

with open(sys.argv[1],encoding='utf-8') as f:
	for line in f.readlines():
		tmp = line.split(" +++$+++ ",1)
		Y_all.append(int(tmp[0]))
		st = ""
		for i in range(len(tmp[1])):
			if i <= len(tmp[1])-3:
				if (tmp[1][i] != tmp[1][i+1]) or (tmp[1][i] != tmp[1][i+2]):
					st += tmp[1][i]
			else:
				st += tmp[1][i]
		s.append(st.strip('\r\n').split(' '))
		www.append(st.strip('\r\n').split(' '))
for c in range(len(s)):
	s[c] += ["_PAD"] * (MAX_SEQUENCE_LENGTH-len(s[c]))

x_test=[]
with open("testing_data.txt",encoding='utf-8') as f:
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
		www.append(st.strip('\r\n').split(' '))
for c in range(len(x_test)):
	x_test[c] += ["_PAD"] * (MAX_SEQUENCE_LENGTH-len(x_test[c]))

with open(sys.argv[2],encoding="utf-8") as f:
	for line in f.readlines():
		st = ""
		for i in range(len(line)):
			if i <= len(line)-3:
				if line[i] != line[i+1] or line[i] != line[i+2]:
					st += line[i]
			else:
				st += line[i]
		www.append(st.strip('\r\n').split(' '))
		
for c in range(len(www)):
	www[c] += ["_PAD"] * (MAX_SEQUENCE_LENGTH-len(www[c]))

print("load data ...finished")

s,Y_all = shuffle(s,Y_all)
# preprocess to word2vec
mm = Word2Vec(www, size=100, min_count=10)
mm.save("word_dict.h5")
print("word2vec ...finished")
# mm = Word2Vec.load("word2vec.h5")

word_dict = {"_PAD": 0}
vocab = [(k, mm.wv[k]) for k, v in mm.wv.vocab.items()]
embedded_matrix = np.zeros((len(mm.wv.vocab.items())+1, mm.vector_size))
for i in range(len(vocab)):
	w = vocab[i][0]
	word_dict[w] = i+1
	embedded_matrix[i+1] = vocab[i][1]
print("embedding matrix ...finished")

shape = (len(s), MAX_SEQUENCE_LENGTH)
X = np.zeros(shape)
for i in range(len(s)):
	for j in range(len(s[i])):
		if s[i][j] in word_dict:
			X[i][j] = int(word_dict[s[i][j]])
print("text encoded ...finished")

model = Sequential()
model.add(Embedding(len(vocab) + 1,
                    100,
                    weights=[embedded_matrix],
                    input_length=MAX_SEQUENCE_LENGTH))
model.add(Bidirectional(GRU(256)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

callbacks = []
callbacks.append(ModelCheckpoint('{epoch:01d}-{val_acc:.5f}-{val_loss:.5f}.h5', monitor='val_acc',save_best_only=True,period=1))

model.fit(X, np.array(Y_all),
          batch_size=512,
          epochs=10,
          callbacks=callbacks,
          validation_split=0.1)
