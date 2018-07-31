import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import random
from Load_data import *
from ops import *


def discriminator(input_image, settings, dataset, reuse = False):
	
	with tf.variable_scope('discriminator') as scope:
		if (reuse):
			tf.get_variable_scope().reuse_variables()

		if settings['Model'] == 'Simple':
			#simple
			h_conv1 = leakyrelu(conv2d(input_image, '1',settings, 2, type_name='d'),'first_layer')
			print(h_conv1)
			h_conv2 = leakyrelu(conv2d(h_conv1, '2',settings, 2, type_name='d'),'second_layer')
			print(h_conv2)
			h_conv3 = leakyrelu(conv2d(h_conv2, '3',settings, 2, type_name='d'),'third_layer')
			print(h_conv3)
			flat = tf.reshape(h_conv3, [settings['batch_size'],-1])
			print(flat)
			final = leakyrelu(fclayer(flat,'4',settings, type_name='d'),'last_layer')
			print(final)
			return final
		if settings['Model'] == 'Complex':
			
			#Complex
			h_conv1 = tf.contrib.layers.batch_norm(conv2d(input_image, '1', settings, 1, type_name='d'), 
				center = True, scale = True, is_training = True, scope = "d_bn1")
			h_1 = leakyrelu(h_conv1, 'layer1')
			print(h_1)
			h_1 = avg_pool_2x2(h_conv1)
			h_conv2 = tf.contrib.layers.batch_norm(conv2d(h_1, '2', settings, 1, type_name='d'), 
				center = True, scale = True, is_training = True, scope = "d_bn2")
			h_2 = leakyrelu(h_conv2, 'layer2')
			print(h_2)
			h_conv3 = tf.contrib.layers.batch_norm(conv2d(h_2, '3', settings, 2, type_name='d'), 
				center = True, scale = True, is_training = True, scope = "d_bn3")
			h_3 = leakyrelu(h_conv3, 'layer3')
			print(h_3)

			drop_out1 = tf.nn.dropout(h_3, settings['keep_prob'])
			h_conv4 = tf.contrib.layers.batch_norm(conv2d(drop_out1, '4', settings, 1, type_name='d'), 
				center = True, scale = True, is_training = True, scope = "d_bn4")
			h_4 = leakyrelu(h_conv4, 'layer4')
			print(h_4)

			h_conv5 = tf.contrib.layers.batch_norm(conv2d(h_4, '5', settings, 1, type_name='d'),
				center = True, scale = True, is_training = True, scope = "d_bn5")
			h_5 = leakyrelu(h_conv5, 'layer5')
			print(h_5)

			h_conv6 = tf.contrib.layers.batch_norm(conv2d(h_5, '6', settings, 2, type_name='d'),
				center = True, scale = True, is_training = True, scope = "d_bn6")
			h_6 = leakyrelu(h_conv6, 'layer6')
			print(h_6)

			drop_out2 = tf.nn.dropout(h_6, settings['keep_prob'])
			h_conv7 = tf.contrib.layers.batch_norm(conv2d(drop_out2, '7', settings, 1, type_name='d'),
				center = True, scale = True, is_training = True, scope = "d_bn7")
			h_7 = leakyrelu(h_conv7, 'layer7')
			print(h_7)

			flatten = tf.layers.flatten(h_7)
			fc1 = tf.contrib.layers.batch_norm(fclayer(flatten, '8', settings, type_name = 'd'),
				center = True, scale = True, is_training = True, scope = "d_bn8")
			h_8 = leakyrelu(fc1, 'layer8')
			print(h_8)

			fc2 = tf.contrib.layers.batch_norm(fclayer(h_8, '9', settings, type_name = 'd'),
				center = True, scale = True, is_training = True, scope = "d_bn9")
			h_9 = leakyrelu(fc2, 'layer9')
			print(h_9)

			return h_9	


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
		print(h_1)

		h_deconv2 = tf.contrib.layers.batch_norm(deconv2d(h_1, '2', settings, 2, output_shape2, type_name = 'g'), center = True, scale = True, is_training = True, scope = "g_bn2")
		h_2 = tf.nn.relu(h_deconv2)
		print(h_2)

		h_deconv3 = tf.contrib.layers.batch_norm(deconv2d(h_2, '3', settings, 2, output_shape3, type_name = 'g'), center = True, scale = True, is_training = True, scope = "g_bn3")
		h_3 = tf.nn.relu(h_deconv3)
		print(h_3)

		h_deconv4 = tf.contrib.layers.batch_norm(deconv2d(h_3, '4', settings, 2, output_shape4, type_name = 'g'), center = True, scale = True, is_training = True, scope = "g_bn4")
		h_4 = tf.nn.tanh(h_deconv4)
		print(h_4)

		return h_4

#__main__

(X_training, Y_training, y_training, X_validation, Y_validation, y_validation,
			X_test, Y_test, y_test) = load_CIFAR_10()

print("X_training")
print(X_training.shape)
print("Y_training")
print(Y_training.shape)
print("y_training")
print(y_training.shape)
print("X_validation")
print(X_validation.shape)
print("Y_validation")
print(Y_validation.shape)
print("y_validation")
print(y_validation.shape)
print("X_test")
print(X_test.shape)
print("Y_test")
print(Y_test.shape)
print("y_test")
print(y_test.shape)

batch_size = 64

d_settings_Simple = {
	'Model': 'Simple',
	'd_wconv1': [5,5,3,32],
	'd_wconv2': [5,5,32,64],
	'd_wconv3': [5,5,64,128],
	'batch_size': batch_size,
	'keep_prob': 0.75,
	'pool_ksize': [1,2,2,1],
	'pool_stride': [1,1,1,1],
	'd_wfc4': [4*4*128,1],

}


d_settings_Complex = {
	'Model': 'Complex',
	'd_wconv1': [5,5,3,32],
	'd_wconv2': [5,5,32,64],
	'd_wconv3': [5,5,64,128],
	'd_wconv4': [5,5,128,256],
	'd_wconv5': [5,5,256,512],
	'd_wconv6': [5,5,512,1024],
	'd_wconv7': [5,5,1024,2048],
	'batch_size': batch_size,
	'keep_prob': 0.75,
	'pool_ksize': [1,2,2,1],
	'pool_stride': [1,1,1,1],
	'd_wfc3': [16*16*16,32],	
	'd_wfc4': [4*4*256,1],
	'd_wfc8':[32768,32],
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

Dx = discriminator(x_placeholder, d_settings_Simple, dataset='CIFAR-10')
Gz = generator(z_placeholder, batch_size, z_dimensions, g_settings)
Dg = discriminator(Gz, d_settings_Simple, dataset = 'CIFAR-10',reuse = 'True')

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
	X_batch = batching(X_training, batch_size)
	z_batch = np.random.normal(-1, 1, size=[batch_size, z_dimensions])
	dLoss = sess.run(d_loss, feed_dict={z_placeholder: z_batch, x_placeholder: X_batch})
	dLoss_list.append(dLoss)
	if dLoss > 1:
		sess.run(trainerD, feed_dict={z_placeholder: z_batch, x_placeholder: X_batch})
	gLoss = sess.run(g_loss, feed_dict={z_placeholder:z_batch})
	gLoss_list.append(gLoss)

	if gLoss > 1:
		sess.run(trainerG, feed_dict={z_placeholder:z_batch})

	#if i%int(iterations/5)==0:
	#	sample_image = generator(z_placeholder, 10, z_dimensions,g_settings, reuse = True)
	#	z_batch = np.random.normal(-1, 1, size=[10, z_dimensions])
	#	temp = (sess.run(sample_image, feed_dict={z_placeholder: z_batch}))
	#	for idx,img in enumerate(temp):
	#		my_i = img.squeeze()
	#		save(my_i,str(i)+str(idx))
xlist = np.arange(iterations)

plt.plot(xlist, dLoss_list, 'b')
plt.plot(xlist, gLoss_list, 'r')
plt.show()

#x = np.arange(0,iterations,1)
#plt.plot(x,dLoss_list, x,gLoss_list)

#sess = tf.Session()
#image_placeholder = tf.placeholder()












