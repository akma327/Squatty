import tensorflow as tf
import numpy as np

class Pose_CNN:
  def __init__(self):
    self.batch_size = 100
    self.img_height = 224
    self.img_width = 224
    self.num_points = 16
    self.nchannels = 3
    self.images = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.img_height, self.img_width, self.nchannels))
    self.labels = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.num_points * 2))

  def build(self):
