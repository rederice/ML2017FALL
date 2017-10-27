import numpy as np
import pandas as pd
import xgboost as xgb
import sys
def read_data(file):
	data = pd.read_csv(file)
	return data.as_matrix().astype('float')

def sigmoid(z):
	return 1 / (1.0 + np.exp(-z))

def outputFile(file):
	with open(sys.argv[4], 'w') as fout:
		print('id,label', file=fout)
		for i in range(file.shape[0]):
			if file[i] >= 0.5:
				print("%d,%d" %(i+1, 1),file=fout)
			else:
				print("%d,%d" %(i+1, 0),file=fout)

def main():
	feat = [0,1,3,4,5]
	x_train = read_data(sys.argv[1])
	y_train = read_data(sys.argv[2])
	x_test = read_data(sys.argv[3])

	x_train = np.concatenate((x_train[:,:59], np.log(x_train[:,feat]+1)), axis=1)
	x_test = np.concatenate((x_test[:,:59], np.log(x_test[:,feat]+1)), axis=1)

	mean = np.mean(np.concatenate((x_train,x_test),axis=0),axis=0)
	std = np.std(np.concatenate((x_train,x_test),axis=0),axis=0)
	x_train = (x_train-mean)/std
	x_test = (x_test-mean)/std

	model = xgb.XGBClassifier(learning_rate=0.07, n_estimators=1300, silent=True)
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	outputFile(y_pred)

if __name__ == '__main__':
	main()