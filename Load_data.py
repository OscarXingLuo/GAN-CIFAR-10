import tensorflow as tensorflow
import numpy as np
from matplotlib import pyplot as pyplot
import random
from preprocessing import *

def load_CIFAR_10():

	X_training = np.load('Data/X_training.npy')
	Y_training = np.load('Data/Y_training.npy')
	y_training = np.load('Data/_y_training.npy')
	
	X_validation = np.load('Data/X_validation.npy')
	Y_validation = np.load('Data/Y_validation.npy')
	y_validation = np.load('Data/_y_validation.npy')
	
	X_test = np.load('Data/X_test.npy')
	Y_test = np.load('Data/Y_test.npy')
	y_test = np.load('Data/_y_test.npy')
	
	return X_training, Y_training, y_training, X_validation, Y_validation, y_validation, X_test, Y_test, y_test