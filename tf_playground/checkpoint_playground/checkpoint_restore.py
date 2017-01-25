import tensorflow as tf
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp_MNIST_data/", one_hot=True)

#date = str( datetime.date.now() )

# placeholder for data
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# get model
W = tf.Variable(tf.truncated_normal([784, 10], mean=0.0, stddev=0.1),name='w')
b = tf.Variable(tf.constant(0.1, shape=[10]),name='b')
y = tf.nn.softmax(tf.matmul(x, W) + b)
#
saver = tf.train.Saver()
with tf.Session() as sess:
    # get model
    saver.restore(sess=sess, save_path='./tmp/my-model-400')
    # loss and accuracy
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # list of booleans indicating correct predictions
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # evaluate
    print(sess.run(fetches=accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
