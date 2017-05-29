# Author: Anthony Ma, Gus Liu
# Email: akma327@stanford.edu, gusliu@stanford.edu
# Date: 05/29/17
# pose_feedforward_baseline.py

import os 
import sys
import numpy as np 
import tensorflow as tf 
import glob 
import ast 
import random
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg



USAGE_STR = """
# Purpose
# Implementation of single layer feedforward neural network baseline for pose estimation

# Usage
# python pose_feedforward_baseline.py 

"""

IMAGE_DIR = "/afs/ir.stanford.edu/users/g/u/gusliu/cs231a/final_project/data/images_scaled"
ANNOTATION_FILE="/afs/ir.stanford.edu/users/g/u/gusliu/cs231a/final_project/data/images/joint_annotation_data.txt"


xdim, ydim, zdim = 640, 480, 3
output_len = 32

def get_data(num_train_pts, num_test_pts):
	"""
		Parse annotation to map image.jpg file to 1 x 32 vector of all x,y joint coordinates
		Output: {img.jpg: [x1, x2, ... x16, y1, y2, ... y16]}
	"""

	x, y = [], []
	f = open(ANNOTATION_FILE, 'r')
	for line in f:
		linfo = line.strip().split("\t")
		image_name = linfo[0]
		coord = map(float, linfo[1:])
		im_path = IMAGE_DIR + "/" + image_name
		img_pixels = mpimg.imread(im_path).reshape((1, xdim*ydim*zdim))

		x.append(img_pixels)
		y.append(coord)


	tot_num_points = num_train_pts + num_test_pts
	if(num_test_pts > len(x)):
		print("Data set does not have " + str(tot_num_points) + " data points.")
		exit(1)

	indices = range(len(x))
	random.shuffle(indices)
	train_indices, test_indices = indices[:num_train_pts], indices[num_train_pts:tot_num_points]
	x_train, y_train = x[train_indices], y[train_indices]
	x_test, y_test = x[test_indices], y[test_indices]

	return x_train, y_train, x_test, y_test


def feedforward_nn():
	### x_train has dimensions (1000, 640*480*3)
	### y_train has dimensions (1000, 32)
	x_train, y_train, x_test, y_test = get_data(1000, 200)

	sess = tf.Session()
	x = tf.placeholder(tf.float32, shape = [None, xdim*ydim*zdim])
	y_ = tf.placeholder(tf.float32, shape = [None, output_len])
	W = tf.Variable(tf.zeros[xdim*ydim*zdim, output_len])
	b = tf.Variable(tf.zeros[output_len])
	y = tf.nn.softmax(tf.matmul(x,W) + b)
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

	LEARNING_RATE = 0.1
	TRAIN_STEPS = 2500
	init = tf.global_variables_initializer()
	sess.run(init)

	training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

	correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	for i in range(TRAIN_STEPS+1):
	    sess.run(training, feed_dict={x: x_train, y_: y_train})
	    if i%100 == 0:
	        print('Training Step:' + str(i) + '  Accuracy =  ' + str(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})) + '  Loss = ' + str(sess.run(cross_entropy, {x: x_train, y_: y_train})))
				


# def load_img(images):
# 	"""
# 		Most common image size 480 x 640
# 	"""
# 	minx, miny = 1000000, 1000000
# 	img_size = {}
# 	for i, im in enumerate(images):
# 		if(i %100 == 0): print(i, len(images))
# 		if(i > 1000): break
# 		if(os.path.isfile(im)):
# 			# if(i > 2): break 
# 			try:
# 				img = mpimg.imread(im)
# 				xdim, ydim, zdim = img.shape
# 				imsize = (xdim, ydim)
# 				print(im, xdim, ydim)
# 				if(imsize not in img_size):
# 					img_size[imsize] = 0
# 				img_size[imsize] += 1
# 				if(xdim < minx): minx = xdim
# 				if(ydim > miny): miny = ydim
# 			except:
# 				print("Invalid file")
# 	print("Smallest Image: ", xdim, ydim)
# 	return img_size



# def driver():
# 	x,y = get_data()
# 	img_size = load_img(x)
# 	print(img_size)



if __name__ == "__main__":
	feedforward_nn()