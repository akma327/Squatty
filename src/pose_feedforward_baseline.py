# Author: Anthony Ma, Gus Liu
# Email: akma327@stanford.edu, gusliu@stanford.edu
# Date: 05/29/17
# pose_feedforward_baseline.py


from utils import *



USAGE_STR = """
# Purpose
# Implementation of single layer feedforward neural network baseline for pose estimation

# Usage
# python pose_feedforward_baseline.py 

"""


def feedforward_nn():
	### x_train has dimensions (1000, 640*480*3)
	### y_train has dimensions (1000, 32)
	x_train, y_train, x_test, y_test = get_data(1000, 200)
	print("x_train", x_train)

	sess = tf.Session()
	x = tf.placeholder(tf.float32, shape = [None, xdim*ydim*zdim])
	y_ = tf.placeholder(tf.float32, shape = [None, output_len])
	W = tf.Variable(tf.zeros((xdim*ydim*zdim, output_len)))
	b = tf.Variable(tf.zeros((output_len)))
	y = tf.matmul(x,W) + b
	mse = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))

	LEARNING_RATE = 0.1
	TRAIN_STEPS = 25
	init = tf.global_variables_initializer()
	sess.run(init)

	training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(mse)

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	for i in range(TRAIN_STEPS+1):
		for j in range(len(x_train)):
		  sess.run(training, feed_dict={x: x_train[j:j+1], y_: y_train[j:j+1]})
		  if i%2 == 0:
				print('Training Step:' + str(i) + '  Accuracy =  ' + str(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})) + '  Loss = ' + str(sess.run(mse, {x: x_train, y_: y_train})))


if __name__ == "__main__":
	feedforward_nn()