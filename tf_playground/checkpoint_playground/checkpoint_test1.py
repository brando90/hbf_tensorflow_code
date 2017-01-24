import tensorflow as tf
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp_MNIST_data/", one_hot=True)

#date = str( datetime.date.now() )

# placeholder for data
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# get model
W = tf.Variable(tf.truncated_normal([784, 10], mean=0.0, stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
# loss and accuracy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # list of booleans indicating correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
saver = tf.train.Saver()
#saver = tf.train.Saver({'W':W})
# train and evalaute
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1001):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(fetches=train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # check_point mdl
        if i % 200 == 0:
            # Append the step number to the checkpoint name:
            saver.save(sess=sess,save_path='./tmp/my-model',global_step=i)
    # evaluate
    print(sess.run(fetches=accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
