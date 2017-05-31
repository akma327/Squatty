# utils.py

import os 
import sys
import numpy as np 
import tensorflow as tf 
import glob 
import ast 
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg


IMAGE_DIR = "/home/gusliu/cs231a/data/images_scaled"
ANNOTATION_FILE="/home/gusliu/cs231a/data/images_scaled/joint_annotation_data_scaled.txt"
TRAINING_PLOT_DIR = "/home/gusliu/cs231a/data/plots"
TEST_PLOT_DIR = "/home/gusliu/cs231a/data/test_plots"

xdim, ydim, zdim = 640, 480, 3
output_len = 32

def get_data(num_train_pts, num_test_pts):
	"""
		Parse annotation to map image.jpg file to 1 x 32 vector of all x,y joint coordinates
		Output: {img.jpg: [x1, x2, ... x16, y1, y2, ... y16]}
	"""
	tot_num_points = num_train_pts + num_test_pts
	x, y, image_paths = [], [], []
	f = open(ANNOTATION_FILE, 'r')
	print("Loading in data ...")
	i = 0
	for line in f:
		i += 1
		if(i > 6010): break
		linfo = line.strip().split("\t")
		image_name = linfo[0]
		coord = map(float, linfo[1:])
		im_path = IMAGE_DIR + "/" + image_name
		img_pixels = mpimg.imread(im_path).reshape((1, xdim*ydim*zdim))[0]
		x.append(img_pixels)
		y.append(list(coord))
		image_paths.append(im_path)

	
	if(num_test_pts > len(x)):
		print("Data set does not have " + str(tot_num_points) + " data points.")
		exit(1)


	x, y = np.array(x).astype('float'), np.array(y).astype('float')
	image_paths = np.array(image_paths)
	indices = range(len(x))
	random.shuffle(indices)
	train_indices, test_indices = indices[:num_train_pts], indices[num_train_pts:tot_num_points]
	x_train, y_train = x[train_indices], y[train_indices]
	x_test, y_test = x[test_indices], y[test_indices]
	image_paths_train, image_paths_test = image_paths[train_indices], image_paths[test_indices]

	return x_train, y_train, x_test, y_test, image_paths_train, image_paths_test


def plot_image_and_points(img_path, points, iteration, training=True):
  img = mpimg.imread(img_path)
  x = points[:len(points) / 2]
  y = points[len(points) / 2:]

  plt.imshow(img)
  plt.scatter(x, y)
  if training:
  	plt.savefig(TRAINING_PLOT_DIR + "/" + str(iteration) + ".png")
  else:
  	plt.savefig(TEST_PLOT_DIR + "/" + str(iteration) + ".png")
  plt.close()

