import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import random
from Load_data import *

	
def avg_pool_2x2(x):
	return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def save(img, name):
	plt.imshow(img)
	plt.savefig(name)


def conv2d(in_data, layer_number, settings, r, type_name = 'd'):
	W_name = type_name+'_wconv'+layer_number
	b_name = type_name+'_bconv'+layer_number
	W = tf.get_variable(W_name, settings[W_name], initializer = tf.truncated_normal_initializer(stddev=0.02))
	b = tf.get_variable(b_name, settings[W_name][-1], initializer = tf.constant_initializer(0))
	return tf.nn.conv2d(input = in_data, filter = W, strides = [1,r,r,1], padding = "SAME")+b


def deconv2d(in_data, layer_number, settings, r, out_shape, type_name = 'g'):
	W_name = type_name+'_wdeconv'+layer_number
	b_name = type_name+'_bdeconv'+layer_number
	W = tf.get_variable(W_name, [settings['filter_size'], settings['filter_size'], out_shape[-1], 
		int(in_data.get_shape()[-1])], initializer = tf.truncated_normal_initializer(stddev=0.02))
	b = tf.get_variable(b_name, [out_shape[-1]], initializer = tf.constant_initializer(0))
	return tf.nn.conv2d_transpose(in_data, W, output_shape=out_shape, strides = [1,r,r,1], padding = "SAME")+b

def fclayer(in_data, layer_number, settings, type_name = 'd'):
	W_name = type_name+'_wfc'+layer_number
	b_name = type_name+'_bfc'+layer_number
	W = tf.get_variable(W_name, settings[W_name], initializer = tf.truncated_normal_initializer(stddev=0.02))
	b = tf.get_variable(b_name, settings[W_name][-1], initializer = tf.constant_initializer(0))
	return tf.matmul(in_data,W)+b

def leakyrelu(in_layer, name):
	return tf.nn.leaky_relu(in_layer, alpha = 0.2, name = name)

def visualize(image_matrix):
	plt.imshow(image_matrix)
	plt.show()
