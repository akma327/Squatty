# Author: Anthony Ma, Gus Liu
# Email: akma327@stanford.edu, gusliu@stanford.edu
# Date: 05/29/17
# deep_cnn_pose.py

from utils import *



xdim, ydim, zdim = 640, 480, 3
output_len = 32

num_filters1 = 16
num_filters2 = 32

fc_size = 512


def weight_variable(shape, name):
	initial = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
	return initial

def bias_variable(shape, name):
	initial = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
	return initial

def conv2d(x,W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')





def cnn():

	### Get data
	# x_train, y_train, x_test, y_test, image_paths_train, image_paths_test = get_data(600, 300)

	print("Loading in training and testing annotations")
	image_paths_train, y_train = get_data(TRAIN_ANNOTATION_FILE)
	image_paths_test, y_test = get_data(TEST_ANNOTATION_FILE)
	

	### Setup CNN
	print("Setting up CNN architecture ...")
	x = tf.placeholder(tf.float32, shape=[None, xdim*ydim*zdim])
	y_ = tf.placeholder(tf.float32, shape=[None, output_len])

	W_conv1 = weight_variable([5,5,zdim, num_filters1], "W1")
	b_conv1 = bias_variable([num_filters1], "b1")

	x_image = tf.reshape(x, [-1, xdim, ydim, zdim])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5,5,num_filters1, num_filters2], "W2")
	b_conv2 = bias_variable([num_filters2], "b2")

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_conv3 = weight_variable([5,5,num_filters1, num_filters2], "W3")
	b_conv3 = bias_variable([num_filters2], "b3")

	h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool3 = max_pool_2x2(h_conv2)

	W_conv4 = weight_variable([5,5,num_filters1, num_filters2], "W4")
	b_conv4 = bias_variable([num_filters2], "b4")

	h_conv4 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool4 = max_pool_2x2(h_conv2)

	W_conv5 = weight_variable([5,5,num_filters1, num_filters2], "W5")
	b_conv5 = bias_variable([num_filters2], "b5")

	h_conv5 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool5 = max_pool_2x2(h_conv2)



	W_fc1 = weight_variable([160*120*num_filters2, fc_size], "W3")
	b_fc1 = bias_variable([fc_size], "b3")

	h_pool2_flat = tf.reshape(h_pool2, [-1, 160*120*num_filters2])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([fc_size, output_len], "W4")
	b_fc2 = bias_variable([output_len], "b4")

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


	### Training
	print("Begin training ...")
	mse = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(mse)
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	NUM_TRAIN_STEPS = 1000
	BATCH_SIZE = 25
	#saver = tf.train.Saver()
	for i in range(NUM_TRAIN_STEPS):
		print("Epoch: " + str(i))

		for j in range(0, len(image_paths_train), BATCH_SIZE):
			batch_x_train = []
			for train_im in image_paths_train[j: j+BATCH_SIZE]:
				im_path = IMAGE_DIR + "/" + train_im 
				img_pixels = mpimg.imread(im_path).reshape((1, xdim*ydim*zdim))[0]
				batch_x_train.append(img_pixels)

			batch_x_train = np.array(batch_x_train).astype(float)
			# batch_x_train = (x_train[j:j+BATCH_SIZE]).astype(float)
			batch_y_train = (y_train[j:j+BATCH_SIZE]).astype(float)

			#print("test", batch_x_train, batch_y_train)

			feed_dict_train = {x: batch_x_train, y_: batch_y_train, keep_prob: 0.5}
			sess.run(train_step, feed_dict_train)
			print('Training loss = ' + str(sess.run(mse, feed_dict_train)))
			if j == 0:
				predictions = sess.run([y_conv], feed_dict_train)
				print predictions, predictions[0][0].shape
				plot_image_and_points(IMAGE_DIR + "/" + image_paths_train[j], predictions[0][0], i, 0)

		### Testing 
		
		testing_losses = []
		for m in range(0, len(image_paths_test), BATCH_SIZE):
			batch_x_test = []
			for test_im in image_paths_test[m:m+BATCH_SIZE]:
				im_path = IMAGE_DIR + "/" + test_im
				img_pixels = mpimg.imread(im_path).reshape((1, xdim*ydim*zdim))[0]
				batch_x_test.append(img_pixels)

			batch_x_test = np.array(batch_x_test).astype(float)
			batch_y_test = (y_test[m: m+BATCH_SIZE])
			feed_dict_test = {x:batch_x_test, y_:batch_y_test, keep_prob:1.0}
			testing_losses.append(sess.run(mse, feed_dict_test))
			if(m == 0):
				predictions = sess.run([y_conv], feed_dict_test)
				for k in range(10):
					plot_image_and_points(IMAGE_DIR + "/" + image_paths_test[m*BATCH_SIZE + k], predictions[0][k], i, k, False)
					

		# test_loss = sum(testing_losses)/len(testing_losses)
		# print('Test Loss = ' + str(test_loss))

		# feed_dict_test = {x: x_test[:BATCH_SIZE], y_: y_test[:BATCH_SIZE], keep_prob: 1.0}
		# print('Test loss = ' + str(sess.run(mse, feed_dict_test)))
		# predictions = sess.run([y_conv], feed_dict_test)
		# for k in range(10):
		# 	plot_image_and_points(IMAGE_DIR + "/" + image_paths_test[k], predictions[0][k], i, k, False)
		#save_path = saver.save(sess, "cnn_1000.ckpt")


if __name__ == "__main__":
	cnn()
