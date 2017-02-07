import tensorflow as tf
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# placeholder for data
x = tf.placeholder(tf.float32, [None, 784])
# variables for parameters
W = tf.Variable(tf.truncated_normal([784, 10], mean=0.0, stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[10]))
# create model
#Wx_b = tf.matmul(x, W) + b
mdl = x
for i in range(5):

    mdl = tf.nn.relu( tf.matmul(mdl, W) + b )
y = tf.nn.softmax(Wx_b) # model
### training
# new placeholder to input the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])
#cost function
cross_entropy = tf.reduce_mean( -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]) )
# single training step opt.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#init op
# init = tf.initialize_all_variables()
# # launch model in a session
# sess = tf.Session()
# #run opt to initialize
# sess.run(init)
# #sess.run(tf.initialize_all_variables())
#
# # we'll run the training step 1000 times
# for i in range(1000):
#   batch_xs, batch_ys = mnist.train.next_batch(100)
#   sess.run(fetches=train_step, feed_dict={x: batch_xs, y_: batch_ys})
#   #class_distribution = sess.run(fetches=y, feed_dict={x: batch_xs, y_: batch_ys})
# accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def get_first_pred(sess,mnist):
    preds = sess.run(fetches=correct_prediction, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    return preds[0]

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    #sess.run(tf.initialize_all_variables())

    # we'll run the training step 1000 times
    for i in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(fetches=train_step, feed_dict={x: batch_xs, y_: batch_ys})
      #class_distribution = sess.run(fetches=y, feed_dict={x: batch_xs, y_: batch_ys})
    # list of booleans indicating correct predictions
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    print(get_first_pred(sess,mnist))
    print(sess.run(fetches=accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
