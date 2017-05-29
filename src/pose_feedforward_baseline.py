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
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg



USAGE_STR = """
# Purpose
# Implementation of single layer feedforward neural network baseline for pose estimation

# Usage
# python pose_feedforward_baseline.py 

"""

IMAGE_DIR = "/afs/ir.stanford.edu/users/g/u/gusliu/cs231a/final_project/data/images"
ANNOTATION_FILE="/afs/ir.stanford.edu/users/g/u/gusliu/cs231a/final_project/data/images/joint_annotation_data.txt"


def get_data():
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
		x.append(IMAGE_DIR + "/" + image_name)
		y.append(coord)

	return x,y 

def load_img(images):
	"""
		Most common image size 480 x 640
	"""
	minx, miny = 1000000, 1000000
	img_size = {}
	for i, im in enumerate(images):
		if(i %100 == 0): print(i, len(images))
		if(i > 1000): break
		if(os.path.isfile(im)):
			# if(i > 2): break 
			try:
				img = mpimg.imread(im)
				xdim, ydim, zdim = img.shape
				imsize = (xdim, ydim)
				print(im, xdim, ydim)
				if(imsize not in img_size):
					img_size[imsize] = 0
				img_size[imsize] += 1
				if(xdim < minx): minx = xdim
				if(ydim > miny): miny = ydim
			except:
				print("Invalid file")
	print("Smallest Image: ", xdim, ydim)
	return img_size


def driver():
	x,y = get_data()
	img_size = load_img(x)
	print(img_size)



if __name__ == "__main__":
	driver()