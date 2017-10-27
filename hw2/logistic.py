import numpy as np
import pandas as pd
import sys
def read_data(file):
	data = pd.read_csv(file)
	return data.as_matrix().astype('float')

def sigmoid(z):
	# z = np.clip(z, 1e3, -1e-3)
	return np.clip(1 / (1.0 + np.exp(-z)), 1e-9, (1-1e-9))

def main():
	x_train = read_data(sys.argv[1])
	y_train = read_data(sys.argv[2])
	x_test = read_data(sys.argv[3])
	feat_pow2 = [0,3,4,5]
	x_train = np.concatenate((x_train[:,:59],x_train[:,feat_pow2]**1.5,x_train[:,feat_pow2]**2, \
		x_train[:,feat_pow2]**2.5,x_train[:,feat_pow2]**3,x_train[:,feat_pow2]**3.5, \
		x_train[:,feat_pow2]**4,x_train[:,feat_pow2]**4.5,x_train[:,feat_pow2]**5, \
		x_train[:,feat_pow2]**5.5,x_train[:,feat_pow2]**6,x_train[:,feat_pow2]**6.5, \
		x_train[:,feat_pow2]**7,x_train[:,feat_pow2]**7.5, \
		x_train[:,feat_pow2]**8,x_train[:,feat_pow2]**8.5, \
		x_train[:,feat_pow2]**9,np.log(x_train[:,feat_pow2]+1)), axis=1)
	x_test = np.concatenate((x_test[:,:59],x_test[:,feat_pow2]**1.5,x_test[:,feat_pow2]**2, \
		x_test[:,feat_pow2]**2.5,x_test[:,feat_pow2]**3,x_test[:,feat_pow2]**3.5, \
		x_test[:,feat_pow2]**4,x_test[:,feat_pow2]**4.5,x_test[:,feat_pow2]**5, \
		x_test[:,feat_pow2]**5.5,x_test[:,feat_pow2]**6,x_test[:,feat_pow2]**6.5, \
		x_test[:,feat_pow2]**7,x_test[:,feat_pow2]**7.5, \
		x_test[:,feat_pow2]**8,x_test[:,feat_pow2]**8.5, \
		x_test[:,feat_pow2]**9,np.log(x_test[:,feat_pow2]+1)), axis=1)

	# normalize
	mean = np.mean(np.concatenate((x_train,x_test),axis=0),axis=0)
	std = np.std(np.concatenate((x_train,x_test),axis=0),axis=0)
	x_train = (x_train-mean)/std
	x_test = (x_test-mean)/std
	
	feat_num = x_train.shape[1]
	data_num = x_train.shape[0]
	weight = np.ones((feat_num, 1))
	bias = 0.0
	w_lr = np.zeros((feat_num, 1))
	b_lr = 0.0

	lamda = 0
	lr = 0.1
	iteration = 4000
	cnt=0
	ploss = 1e9
	for cnt in range(iteration):
		f = sigmoid(bias + x_train.dot(weight))
		loss = -(y_train.T.dot(np.log(f+1e-20)) + (1-y_train).T.dot(np.log(1-f+1e-20)))
		# print(loss)
		if loss > ploss:
			break
		error = y_train - f # [data_num, 1]
		w_grad = (x_train.T.dot(error) + lamda*weight) / data_num
		b_grad = np.sum(error) / data_num
		w_lr += w_grad**2
		b_lr += b_grad**2
		weight -= lr/np.sqrt(w_lr)*(-w_grad)
		bias -= lr/np.sqrt(b_lr)*(-b_grad)

		ploss = loss

	print("iter=",cnt,"; loss=",ploss)
	ans = x_test.dot(weight)+bias
	
	# output
	with open(sys.argv[4], 'w') as fout:
		print('id,label', file=fout)
		for i in range(ans.shape[0]):
			if ans[i] >= 0.5:
				print("%d,%d" %(i+1, 1),file=fout)
			else:
				print("%d,%d" %(i+1, 0),file=fout)

if __name__ == '__main__':
	main()