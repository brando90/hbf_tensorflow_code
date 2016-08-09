import numpy as np

import sklearn as sk
from sklearn.metrics.pairwise import euclidean_distances

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


task_name = 'task_MNIST_flat_auto_encoder'
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train, Y_train = mnist.train.images, mnist.train.labels # N x D
X_cv, Y_cv = mnist.validation.images, mnist.validation.labels
X_test, Y_test = mnist.test.images, mnist.test.labels

# data shape is "[batch, in_height, in_width, in_channels]",
# X_train = N x D
N, D = X_train.shape
x = tf.Variable( X_train ).reshape(N,1,D,1)
filter_size, K = 10, 12 # number of units
W = tf.Variable( tf.truncated_normal(shape=[1,filter_size,1,K], stddev=0.1) )
conv = tf.nn.conv2d(input=x, filter=W,strides=[1, 1, 1, 1], padding="SAME", name="conv")

with tf.Session() as sess:
    sess.run(fetches=conv, feed_dict={x:X_train})
