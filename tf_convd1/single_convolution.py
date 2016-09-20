import numpy as np

import tensorflow as tf

import my_tf_pkg as mtf

[kernel_height, kernel_width] = [1,5]
F = 8
x = tf.placeholder(tf.float32, shape=[None,1,D,1], name='x-input') #[M, 1, D, 1]
conv = tf.contrib.layers.convolution2d(inputs=x,
    kernel_size=[kernel_height, kernel_width],
    num_outputs=[F],
    padding='VALID'
    rate=1,
    activation_fn=tf.nn.relu,
    trainable=True
)



with tf.Session() as sess:
    print( sess.run(fetches=y1, feed_dict={x:}) )
