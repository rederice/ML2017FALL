import pandas as pd
import numpy as np
import sys

test_path = sys.argv[1]
output_path = sys.argv[2]
read_model = int(sys.argv[3]) # 1 => read best_model
factor = ['AMB', 'CH4', 'CO', 'NMHC', 'NO', 'NO2',
        'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH',
        'SO2', 'THC', 'WD_HR', 'WIND_DIR', 'WIND_SPEED', 'WS_HR']

data1 = pd.read_csv(test_path, header=None, encoding='big5').as_matrix()
testdata = data1[:,2:]
testdata[testdata == 'NR'] = 0.0
testdata = testdata.astype('float') # [4320, 9]

if read_model != 1:
	data = pd.read_csv("train.csv",encoding='big5').as_matrix() #convert to numpy
	traindata = data[:,3:]
	traindata[traindata == 'NR'] = 0.0
	traindata = traindata.astype('float') # [4320, 24]
	train_x = []
	train_y = []
	for i in range(0, traindata.shape[0], 18*20): # 12 months
		days = np.vsplit(traindata[i:i+18*20], 20) # [18, 24]*20
		merge = np.concatenate(days,axis=1) # [18, 480]
		for j in range(0, merge.shape[1]-9):
			# [0~8(1st 9hr,attr1),9~17(attr2),...,~161(attr18),162~170(2nd 9hr...)] 
			# total : [months, attrs * continuous_hrs]
			train_x.append(merge[:, j:j+9].flatten()) 
			train_y.append([merge[9, j+9]])
	train_allhours = np.array(train_x)
	train_res = np.array(train_y)

test_x = []
days = np.vsplit(testdata, testdata.shape[0]/18) # [18, 9]*240
for i in days:
	test_x.append(i.flatten())
test = np.array(test_x)

# dict for attrs -> hours range column
attr_colrange = {}
for i, attr in enumerate(factor):
    attr_colrange[attr] = list(range(9*i, 9*i+9))
select = ['PM2.5','PM10','O3']
select_colrange = []
for attr in select:
	select_colrange += attr_colrange[attr]
# PM2.5 => 81~89
if read_model != 1:
	train_calc = train_allhours[:, select_colrange]
	train_calc = np.concatenate((train_calc, train_calc**2),axis=1)

test = test[:, select_colrange]
test = np.concatenate((test, test**2),axis=1)

lr = 1.0
lamda = 0.0
iteration = 100000
if read_model != 1:
	data_num = train_calc.shape[0]
	param_num = train_calc.shape[1]
	print(data_num,param_num)
else:
	data_num = 5652
	param_num = 54
weight = np.ones((param_num, 1)) # coefficients
b = 0.0
w_lr = np.zeros((param_num, 1)) # accumulate denominator of Adagrad
b_lr = 0.0

if read_model == 1:
	#read from model txt
	arr = np.genfromtxt("best_model", delimiter="")
	nor_min = arr[0:param_num]
	nor_max = arr[param_num:2*param_num]
	b = arr[2*param_num]
	weight = arr[2*param_num+1:]
	weight = weight.T
	# print(weight)

if read_model != 1:
	# normalize
	nor_min = np.min(train_calc, axis=0)
	nor_max = np.max(train_calc, axis=0)
	train_calc = (train_calc - nor_min)/(nor_max - nor_min)

	ploss = 1e9
	for cnt in range(iteration):
		#train_calc = np.reshape(train_calc, (-1, param_num))
		# err = y-(b+wx)
		error = train_res - (train_calc.dot(weight)+b)
		# differential
		w_grad = -1.0 * (train_calc.T.dot(error) + lamda*weight) / data_num
		b_grad = -1.0 * np.sum(error) / data_num
		# move on graph
		w_lr += w_grad**2
		b_lr += b_grad**2
		weight = weight - lr / np.sqrt(w_lr) * w_grad
		b = b - lr / np.sqrt(b_lr) * b_grad

		loss = (np.sqrt(np.mean(error ** 2)))
		if ploss < loss:
			break
		ploss=loss

	print(ploss)

ans = (test - nor_min)/(nor_max - nor_min)
ans = ans.reshape((-1, param_num))
ans = ans.dot(weight)+b

file = open(output_path, 'w')
print("id,value", file = file)
for i in range(ans.shape[0]):
	print("id_%d,%f" % (i, ans[i]), file = file)
file.close