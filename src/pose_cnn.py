# Author: Anthony Ma, Gus Liu
# Email: akma327@stanford.edu, gusliu@stanford.edu
# Date: 05/29/17
# cnn_pose.py

import os 
import sys
import numpy as np 
import tensorflow as tf 
import glob 
import ast 
import random
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from utils import *



xdim, ydim, zdim = 640, 480, 3
output_len = 32


def cnn():
	x_train, y_train, x_test, y_test = get_data(100, 20)

	print(x_train)


if __name__ == "__main__":
	cnn()