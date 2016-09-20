import numpy as np

import tensorflow as tf

import my_tf_pkg as mtf

normalizer_fn = None
normalizer_fn = tf.contrib.layers.batch_norm

#weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
weights_initializer = tf.constant_initializer(value=1.0, dtype=tf.float32)
#biases_initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
biases_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)

D = 5
kernel_height = 1
kernel_width = 3
F = 4
x = tf.placeholder(tf.float32, shape=[None,1,D,1], name='x-input') #[M, 1, D, 1]
conv = tf.contrib.layers.convolution2d(inputs=x,
    num_outputs=F, # 4
    kernel_size=[kernel_height, kernel_width], # [1,3]
    stride=[1,1],
    padding='VALID',
    rate=1,
    activation_fn=tf.nn.relu,
    normalizer_fn=normalizer_fn,
    normalizer_params=None,
    weights_initializer=weights_initializer,
    biases_initializer=biases_initializer,
    trainable=True,
    scope='cnn'
)

# syntheitc data
M = 2
X_data = np.array( [np.arange(0,5),np.arange(5,10)] )
print(X_data)
X_data = X_data.reshape(M,1,D,1)
with tf.Session() as sess:
    sess.run( tf.initialize_all_variables() )
    print( sess.run(fetches=conv, feed_dict={x:X_data}) )
