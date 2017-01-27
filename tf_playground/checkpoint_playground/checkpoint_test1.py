import tensorflow as tf
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp_MNIST_data/", one_hot=True)

import pdb

def get_mdl(x):
    # get model
    W = tf.Variable(tf.truncated_normal([784, 10], mean=0.0, stddev=0.1),name='w')
    b = tf.Variable(tf.constant(0.1, shape=[10]),name='b')
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    return y

# build graph
graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    y = get_mdl(x)
    # loss and accuracy
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # list of booleans indicating correct predictions
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # optimizer
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # step for optimizer
    step = tf.Variable(0, name='step')
    #step_assign = step.assign(i)
    # save everything that was saved in the session
    saver = tf.train.Saver()

# train and evalaute
with tf.Session(graph=graph) as sess:
    # placeholder for data
    sess.run(tf.global_variables_initializer())
    for i in range(2001):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(fetches=train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # check_point mdl
        if i % 100 == 0:
            step_assign = step.assign(i)
            sess.run(step_assign)
            print('step: ',i)
            print(sess.run(fetches=accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
            # Append the step number to the checkpoint name:
            #saver.save(sess=sess,save_path='./tmp/my-model',global_step=i)
            saver.save(sess=sess,save_path='./tmp/mdl_ckpt')
    # evaluate
    print(step.eval())
    print( [ op.name for op in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)] )
    print(sess.run(fetches=accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
