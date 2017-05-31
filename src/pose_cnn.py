# Author: Anthony Ma, Gus Liu
# Email: akma327@stanford.edu, gusliu@stanford.edu
# Date: 05/29/17
# cnn_pose.py

from utils import *



xdim, ydim, zdim = 640, 480, 3
output_len = 32

num_filters1 = 32
num_filters2 = 64

fc_size = 1024


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(float(1)/output_len, shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')





def cnn():

	### Get data
	x_train, y_train, x_test, y_test, image_paths_train, image_paths_test = get_data(100, 20)


	### Setup CNN

	x = tf.placeholder(tf.float32, shape=[None, xdim*ydim*zdim])
	y_ = tf.placeholder(tf.float32, shape=[None, output_len])

	W_conv1 = weight_variable([5,5,zdim, num_filters1])
	b_conv1 = bias_variable([num_filters1])

	x_image = tf.reshape(x, [-1, xdim, ydim, zdim])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5,5,num_filters1, num_filters2])
	b_conv2 = bias_variable([num_filters2])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([160*120*num_filters2, fc_size])
	b_fc1 = bias_variable([fc_size])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 160*120*num_filters2])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([fc_size, output_len])
	b_fc2 = bias_variable([output_len])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


	### Training
	mse = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))
	train_step = tf.train.AdamOptimizer(1e-2).minimize(mse)
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	NUM_TRAIN_STEPS = 1000
	BATCH_SIZE = 50
	for i in range(NUM_TRAIN_STEPS):
		for j in range(0, len(x_train), BATCH_SIZE):
			batch_x = (x_train[j:j+BATCH_SIZE]).astype(float)
			batch_y = (y_train[j:j+BATCH_SIZE]).astype(float)

			feed_dict = {x: batch_x, y_: batch_y, keep_prob: 0.5}
			print('Loss = ' + str(sess.run(mse, feed_dict)))
			if j == 0:
				predictions = sess.run([y_conv], feed_dict)
				print predictions, predictions[0][0].shape
				plot_image_and_points(image_paths_train[j], predictions[0][0], i)




if __name__ == "__main__":
	cnn()