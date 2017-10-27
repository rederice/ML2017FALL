import numpy as np
import pandas as pd
import sys
def read_data(file):
	data = pd.read_csv(file)
	return data.as_matrix().astype('float')

def divide_class(label, x, y):
	divide = []
	for i in range(y.shape[0]):
		if y[i] == label:
			divide.append(x[i])
	return np.array(divide)

def sigmoid(z):
	return 1 / (1.0 + np.exp(-z))

def main():
	x_train = read_data(sys.argv[1])
	y_train = read_data(sys.argv[2])
	x_test = read_data(sys.argv[3])

	mean = np.mean(x_train,axis=0)
	std = np.std(x_train,axis=0)
	x_train = (x_train-mean)/std
	x_test = (x_test-mean)/std
	
	x_over = divide_class(1, x_train, y_train)
	x_under = divide_class(0, x_train, y_train)

	over_mean = np.mean(x_over, axis=0)
	under_mean = np.mean(x_under, axis=0)
	
	over_sigma = (x_over - over_mean).T.dot(x_over - over_mean)
	under_sigma = (x_under - under_mean).T.dot(x_under - under_mean)
	sigma = (over_sigma + under_sigma)/x_train.shape[0]

	w = ((over_mean - under_mean).T).dot( np.linalg.inv(sigma) )
	b = (under_mean.T.dot(np.linalg.inv(sigma))).dot(under_mean)/2 \
		- (over_mean.T.dot(np.linalg.inv(sigma))).dot(over_mean)/2 \
		+ np.log(x_over.shape[0] / x_under.shape[0])
	z = w.dot(x_test.T) + b
	p = sigmoid(z)

	with open(sys.argv[4], 'w') as fout:
		print('id,label', file=fout)
		for i in range(p.shape[0]):
			if p[i] >= 0.5:
				print("%d,%d" %(i+1, 1),file=fout)
			else:
				print("%d,%d" %(i+1, 0),file=fout)

if __name__ == '__main__':
	main()