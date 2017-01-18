import numpy as np

import tensorflow as tf

import my_tf_pkg as mtf

# from tensorflow.examples.tutorials.mnist import input_data
#
# task_name = 'task_MNIST_flat_auto_encoder'
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# X_train, Y_train = mnist.train.images, mnist.train.labels # N x D
# X_cv, Y_cv = mnist.validation.images, mnist.validation.labels
# X_test, Y_test = mnist.test.images, mnist.test.labels

# data shape is "[batch, in_height, in_width, in_channels]",
# X_train = N x D
task_name = 'task_f_4d'
(X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = mtf.get_data(task_name)
X_train_org = X_train
#X_train_org = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11]])
#X_train_org = np.array([[0,1,2,3]])
#X_train_org = np.array([[0,1,2,3],[4,5,6,7]])
N, D = X_train_org.shape
print X_train_org.shape
X_train_1d = X_train_org.reshape(N,1,D,1)
#X_train = tf.constant( X_train_org )
# think of it as N images with height 1 and width D.
xx = tf.placeholder(tf.float32, shape=[None,1,D,1], name='xx-input')
#x = tf.Variable( X_train , name='x-input')
# filter shape is "[filter_height, filter_width, in_channels, out_channels]"
filter_size, nb_filters = 2, 3 # filter_size , number of hidden units/units
# think of it as having nb_filters number of filters, each of size filter_size
#W_filters = np.array([[1, 3, 5],[2, 4, 6]]).reshape(1,filter_size,1,nb_filters)
init_W = tf.truncated_normal(shape=[1,filter_size,1,nb_filters], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
W_filters = tf.get_variable(name='W', dtype=tf.float64, initializer=init_W, regularizer=None, trainable=True)
#W = tf.Variable( tf.truncated_normal(shape=[1,filter_size,1,nb_filters], stddev=0.1) )
#W = tf.Variable( tf.constant(W_filters, dtype=tf.float32) )
W = W_filters
stride_convd1 = 2 # controls the stride for 1D convolution
conv = tf.nn.conv2d(input=xx, filter=W, strides=[1, 1, stride_convd1, 1], padding="SAME", name="conv")
conv_flat = tf.reshape(conv, [-1,filter_size*nb_filters])
#C = tf.constant( (np.array([[1,1]]).T) , dtype=tf.float32 ) #
#tf.reshape( conv , [])
#y_tf = tf.matmul(conv, C)

##
x = tf.placeholder(tf.float32, shape=[None,D], name='x-input') # N x 4
W1 = tf.Variable( tf.constant( np.array([[1,2,0,0],[3,4,0,0],[5,6,0,0]]).T, dtype=tf.float32 ) ) # 2 x 4
y1 = tf.matmul(x,W1) # N x 2 = N x 4 x 4 x 2
W2 = tf.Variable( tf.constant( np.array([[0,0,1,2],[0,0,3,4],[0,0,5,6]]).T, dtype=tf.float32 ))
y2 = tf.matmul(x,W2) # N x 2 = N x 4 x 4 x 2
# C1 = tf.constant( np.array([[4,3]]).T, dtype=tf.float32 ) # 1 x 2
# C2 = tf.constant( np.array([[2,1]]).T, dtype=tf.float32 )
#
# p1 = tf.matmul(y1,C1)
# p2 = tf.matmul(y2,C2)
#y = p1 + p2
with tf.Session() as sess:
    sess.run( tf.initialize_all_variables() )
    print 'manual conv'
    print sess.run(fetches=y1, feed_dict={x:X_train_org})
    print sess.run(fetches=y2, feed_dict={x:X_train_org})
    #print sess.run(fetches=y, feed_dict={x:X_train_org})
    print 'tf conv'
    output_conv = sess.run(fetches=conv, feed_dict={xx:X_train_1d})
    output_conv_flat = sess.run(fetches=conv_flat, feed_dict={xx:X_train_1d})
    print output_conv
    print output_conv_flat
    #print sess.run(fetches=y_tf, feed_dict={xx:X_train_1d})
