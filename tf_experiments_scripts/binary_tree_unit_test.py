import numpy as np

import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
#
# task_name = 'task_MNIST_flat_auto_encoder'
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# X_train, Y_train = mnist.train.images, mnist.train.labels # N x D
# X_cv, Y_cv = mnist.validation.images, mnist.validation.labels
# X_test, Y_test = mnist.test.images, mnist.test.labels

# data shape is "[batch, in_height, in_width, in_channels]",
# X_train = N x D
X_train = tf.constant( np.array([[1,2,3,4],[4,5,6,7],[8,9,10,11]]) )
N, D = X_train.shape
# think of it as N images with height 1 and width D.
X_train = X_train.reshape(N,1,D,1)
x = tf.placeholder(tf.float32, shape=[None,1,D,1], name='x-input')
#x = tf.Variable( X_train , name='x-input')
# filter shape is "[filter_height, filter_width, in_channels, out_channels]"
filter_size, nb_filters = 10, 12 # filter_size , number of hidden units/units
# think of it as having nb_filters number of filters, each of size filter_size
W = tf.Variable( tf.truncated_normal(shape=[1,filter_size,1,nb_filters], stddev=0.1) )
stride_convd1 = 2 # controls the stride for 1D convolution
conv = tf.nn.conv2d(input=x, filter=W, strides=[1, stride_convd1, 1, 1], padding="SAME", name="conv")


##
W1 = tf.variable( tf.constant( np.array([1,2,0,0],[3,4,0,0]) ) )
y1 = tf.matmul(W1,x)
W2 = tf.variable( tf.constant( np.array([0,0,1,2],[0,0,3,4]) ) )
y2 = tf.matmul(W2,x)
C1 = tf.constnat( np.array([4,3]) )
C2 = tf.constnat( np.array([2,1]) )
y = tf.matmul(C1,y1) + tf.matmul(C2,y2)
with tf.Session() as sess:
    sess.run( tf.initialize_all_variables() )
    print sess.run(fetches=conv, feed_dict={x:X_train})
    print sess.run(fetches=y, feed_dict={x:X_train})
