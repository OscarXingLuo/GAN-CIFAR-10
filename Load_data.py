#Read data
import numpy as np 
import random
from matplotlib import pyplot as plt
import math
#172.20.2.4
#lustre2
def unpickle(file):
	import pickle
	with open(file,'rb') as fo:
		dict = pickle.load(fo, encoding = 'bytes')
	return dict

def one_hot(Y,dataset = "CIFAR-10"):
	if dataset == "CIFAR-10":
		y = np.zeros((Y.shape[0],10))
		for index, label in enumerate(Y):
			y[index, int(label)] = 1
		return y
	else:
		print("No such dataset as '"+dataset+"'")
	return y

def batching(X, Y, batch_size):
	idx_list = []
	for i in range(batch_size):
		idx_list.append(int(random.randint(0,X.shape[0]-1)))
	X_batch = []
	Y_batch = np.zeros((batch_size,1),dtype = int)

	for i,idx in enumerate(idx_list):
		X_batch.append(X[idx,:,:,:])
		Y_batch[i,:]=Y[idx]
	X_batch = np.stack(X_batch)
	return X_batch, Y_batch

def visualize(image_matrix):
	plt.imshow(image_matrix)
	plt.show()

def CIFAR_10():
	ds1 = unpickle('Datasets/cifar-10-batches-py/data_batch_1')
	ds2 = unpickle('Datasets/cifar-10-batches-py/data_batch_2')
	ds3 = unpickle('Datasets/cifar-10-batches-py/data_batch_3')
	ds4 = unpickle('Datasets/cifar-10-batches-py/data_batch_4')
	ds5 = unpickle('Datasets/cifar-10-batches-py/data_batch_5')
	ds6 = unpickle('Datasets/cifar-10-batches-py/test_batch')

	X_training = np.concatenate((ds1[b'data'], ds2[b'data'], ds3[b'data'], ds4[b'data']))
	X_training = np.array(X_training, dtype = float)
	X_training = np.transpose(np.reshape(X_training,[40000,3,32,32]),(0,2,3,1))/255

	Y_training = np.concatenate((ds1[b'labels'], ds2[b'labels'], ds3[b'labels'], ds4[b'labels']))
	Y_training = np.array(Y_training, dtype= float)
	y_training = one_hot(Y_training, dataset = "CIFAR-10")

	X_validation = np.array(ds5[b'data'])
	X_validation = np.array(X_validation,dtype = float)
	X_validation = np.transpose(np.reshape(X_validation, [10000, 32,32,3]),(0,2,3,1))/255

	Y_validation = np.array(ds5[b'labels'])
	Y_validation = np.array(Y_validation,dtype = float)
	y_validation = one_hot(Y_validation, dataset="CIFAR-10")

	X_test = np.array(ds6[b'data'])
	X_test = np.array(X_test, dtype=float)
	X_test = np.transpose(np.reshape(X_validation, [10000,32,32,3]),(0,2,3,1))/255

	Y_test = np.array(ds6[b'labels'])
	Y_test = np.array(Y_test, dtype=float)
	#Y_temp_test = np.zeros((10000,1), dtype = int)
	#for i in range(Y_test.shape[0]):
		#Y_temp_test[i,0]=Y_test[i]
	#Y_test = Y_temp_test
	y_test = one_hot(Y_test, dataset = "CIFAR-10")

	return (X_training, Y_training, y_training, X_validation, Y_validation, y_validation,
		X_test, Y_test, y_test)


#(X_training, Y_training, y_training, X_validation, Y_validation, y_validation,
#			X_test, Y_test, y_test) = CIFAR_10()

#print(X_training[0,:,:,:].shape)
#//lustre.dlp.com/lustre2