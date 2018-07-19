import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import random
from Load_data import *

	
def avg_pool_2x2(x):
	return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


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

def discriminator(input_image, settings, dataset, reuse = False):
	
	with tf.variable_scope('discriminator') as scope:
		if (reuse):
			tf.get_variable_scope().reuse_variables()
		if dataset == 'CIFAR-10':
			#simple
			#h_conv1 = leakyrelu(conv2d(input_image, '1',settings, 2, type_name='d'),'first_layer')
			#h_conv2 = leakyrelu(conv2d(h_conv1, '2',settings, 2, type_name='d'),'second_layer')
			#max_pooling = tf.nn.max_pool(h_conv2, settings['pool_ksize'], settings['pool_stride'], padding = "SAME")
			#avg_pool = avg_pool_2x2(max_pooling)
			#flat = tf.reshape(avg_pool, [-1, 16*16*16])
			#fc = leakyrelu(fclayer(flat,'3', settings, type_name='d'),'third_layer')
			#final = leakyrelu(fclayer(fc,'4',settings, type_name='d'),'last_layer')
	
			#Complex
			h_conv1 = tf.contrib.layers.batch_norm(conv2d(input_image, '1', settings, 1, type_name='d'), 
				center = True, scale = True, is_training = True, scope = "d_bn1")
			h_1 = leakyrelu(h_conv1, 'layer1')
			#h_1 = avg_pool_2x2(h_conv1)
			h_conv2 = tf.contrib.layers.batch_norm(conv2d(h_1, '2', settings, 1, type_name='d'), 
				center = True, scale = True, is_training = True, scope = "d_bn2")
			h_2 = leakyrelu(h_conv2, 'layer2')

			h_conv3 = tf.contrib.layers.batch_norm(conv2d(h_2, '3', settings, 2, type_name='d'), 
				center = True, scale = True, is_training = True, scope = "d_bn3")
			h_3 = leakyrelu(h_conv3, 'layer3')

			drop_out1 = tf.nn.dropout(h_3, settings['keep_prob'])
			h_conv4 = tf.contrib.layers.batch_norm(conv2d(drop_out1, '4', settings, 1, type_name='d'), 
				center = True, scale = True, is_training = True, scope = "d_bn4")
			h_4 = leakyrelu(h_conv4, 'layer4')

			h_conv5 = tf.contrib.layers.batch_norm(conv2d(h_4, '5', settings, 1, type_name='d'),
				center = True, scale = True, is_training = True, scope = "d_bn5")
			h_5 = leakyrelu(h_conv5, 'layer5')

			h_conv6 = tf.contrib.layers.batch_norm(conv2d(h_5, '6', settings, 2, type_name='d'),
				center = True, scale = True, is_training = True, scope = "d_bn6")
			h_6 = leakyrelu(h_conv6, 'layer6')

			drop_out2 = tf.nn.dropout(h_6, settings['keep_prob'])
			h_conv7 = tf.contrib.layers.batch_norm(conv2d(drop_out2, '7', settings, 1, type_name='d'),
				center = True, scale = True, is_training = True, scope = "d_bn7")
			h_1 = leakyrelu(h_conv1, 'layer7')

			avg_pool = avg_pool_2x2(h_conv7)
			flatten = tf.layers.flatten(avg_pool)
			fc1 = tf.contrib.layers.batch_norm(fclayer(flatten, '8', settings, type_name = 'd'),
				center = True, scale = True, is_training = True, scope = "d_bn8")
			h_1 = leakyrelu(fc1, 'layer8')

			fc2 = tf.contrib.layers.batch_norm(fclayer(fc1, '9', settings, type_name = 'd'),
				center = True, scale = True, is_training = True, scope = "d_bn9")
			h_1 = leakyrelu(fc2, 'layer9')

			return fc2


def generator(z, batch_size, z_dim, settings, reuse = False):
	with tf.variable_scope('generator') as scope:
		if (reuse):
			tf.get_variable_scope().reuse_variables()
		g_dim = settings['g_dim']
		c_dim = settings['c_dim']
		s = settings['s']
		s, s2, s4, s8, s16 = settings['s'], settings['s2'], settings['s4'], settings['s8'], settings['s16']

		output_shape1 = [batch_size, s8, s8, g_dim*4]
		output_shape2 = [batch_size, s4, s4, g_dim*2]
		output_shape3 = [batch_size, s2, s2, g_dim]
		output_shape4 = [batch_size, s, s, 3]

		input_noise = tf.reshape(z, [batch_size, s16, s16, 25])		
		input_vector = tf.nn.relu(input_noise)

		h_deconv1 = tf.contrib.layers.batch_norm(deconv2d(input_vector, '1', settings, 2, output_shape1, type_name = 'g'), center = True, scale = True, is_training = True, scope = "g_bn1")
		h_1 = tf.nn.relu(h_deconv1)
		h_deconv2 = tf.contrib.layers.batch_norm(deconv2d(h_1, '2', settings, 2, output_shape2, type_name = 'g'), center = True, scale = True, is_training = True, scope = "g_bn2")
		h_2 = tf.nn.relu(h_deconv2)
		h_deconv3 = tf.contrib.layers.batch_norm(deconv2d(h_2, '3', settings, 2, output_shape3, type_name = 'g'), center = True, scale = True, is_training = True, scope = "g_bn3")
		h_3 = tf.nn.relu(h_deconv3)
		h_deconv4 = tf.contrib.layers.batch_norm(deconv2d(h_3, '4', settings, 2, output_shape4, type_name = 'g'), center = True, scale = True, is_training = True, scope = "g_bn4")
		h_4 = tf.nn.tanh(h_deconv4)

		return h_deconv4

#__main__

(X_training, Y_training, y_training, X_validation, Y_validation, y_validation,
			X_test, Y_test, y_test) = CIFAR_10()

batch_size = 16

d_settings = {
	
	'd_wconv1': [5,5,3,8],
	'd_wconv2': [5,5,8,16],
	'd_wconv3': [5,5,16,32],
	'd_wconv4': [5,5,32,64],
	'd_wconv5': [5,5,64,128],
	'd_wconv6': [5,5,128,256],
	'd_wconv7': [5,5,256,512],

	'keep_prob': 0.75,
	'pool_ksize': [1,2,2,1],
	'pool_stride': [1,1,1,1],
	'd_wfc3': [16*16*16,32],	
	'd_wfc4': [32,1],
	'd_wfc8':[8192,32],
	'd_wfc9': [32,1]
}

g_settings = {

	'g_dim': 64,
	'c_dim': 3,
	's': 32,
	's2': 16,
	's4': 8,
	's8': 4,
	's16': 2,
	'filter_size': 5,
}

sess=tf.Session()

x_placeholder=tf.placeholder("float", shape=[None, 32,32,3])
z_dimensions = 100
z_placeholder = tf.placeholder("float", shape=[None,z_dimensions])

Dx = discriminator(x_placeholder, d_settings, dataset='CIFAR-10')
Gz = generator(z_placeholder, batch_size, z_dimensions, g_settings)
Dg = discriminator(Gz, d_settings, dataset = 'CIFAR-10', reuse = 'True')

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))
d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.zeros_like(Dx))) + \
tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

adam = tf.train.AdamOptimizer()
trainerD = adam.minimize(d_loss, var_list=d_vars)
trainerG = adam.minimize(g_loss, var_list=g_vars)

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)
iterations = 1000
dLoss_list = []
gLoss_list = []
for i in range(iterations):
	print(i)
	X_batch, Y_batch = batching(X_training,Y_training, batch_size)
	z_batch = np.random.normal(-1, 1, size=[batch_size, z_dimensions])
	_,dLoss = sess.run([trainerD,d_loss], feed_dict={z_placeholder: z_batch, x_placeholder: X_batch})
	dLoss_list.append(dLoss)
	_,gLoss = sess.run([trainerG,g_loss], feed_dict={z_placeholder:z_batch})
	gLoss_list.append(gLoss)


#x = np.arange(0,iterations,1)
#plt.plot(x,dLoss_list, x,gLoss_list)
sample_image = generator(z_placeholder, 1, z_dimensions, g_settings, reuse = True)
z_batch = np.random.normal(-1, 1, size=[1, z_dimensions])
temp = (sess.run(sample_image, feed_dict = {z_placeholder: z_batch}))
classify_once = discriminator(x_placeholder, d_settings,)
img = temp.squeeze()
plt.imshow(img)
plt.show()

#sess = tf.Session()
#image_placeholder = tf.placeholder()












