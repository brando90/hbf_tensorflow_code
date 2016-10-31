import namespaces as ns

import tensorflow as tf
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def get_mdl(x):
    # placeholder for data
    # variables for parameters
    W = tf.Variable(tf.truncated_normal([784, 10], mean=0.0, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[10]))
    Wx_b = tf.matmul(x, W) + b
    y = tf.nn.softmax(Wx_b)
    return y

def main():
    #arg = ns.Namespace()
    nb_array_jobs = 2
    for index in range(nb_array_jobs):
        x = tf.placeholder(tf.float32, [None, 784])
        y = get_mdl(x)
        y_ = tf.placeholder(tf.float32, [None, 10])
        #
        cross_entropy = tf.reduce_mean( -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]) )
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) ## list of booleans indicating correct predictions
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            # we'll run the training step 1000 times
            for i in range(100):
              batch_xs, batch_ys = mnist.train.next_batch(50)
              sess.run(fetches=train_step, feed_dict={x: batch_xs, y_: batch_ys})
            # list of booleans indicating correct predictions
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            #prints accuracy of model
            print(sess.run(fetches=accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
    main()
