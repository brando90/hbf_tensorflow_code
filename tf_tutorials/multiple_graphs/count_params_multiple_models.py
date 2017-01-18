import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def get_mdl1(x):
    # variables for parameters
    W = tf.Variable(tf.truncated_normal([784, 10], mean=0.0, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[10]))
    Wx_b = tf.matmul(W,x) + b
    mdl = tf.nn.relu(Wx_b)
    return mdl

def get_mdl2(x):
    return mdl

# placeholder for data
x = tf.placeholder(tf.float32, [None, 784])

mdl1 = get_mdl1(x)
mdl2 = get_mdl2(x)

X_data = mnist.train.images
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    bt_output = sess.run(fetches=bt_mdl, feed_dict={x:X_data})
