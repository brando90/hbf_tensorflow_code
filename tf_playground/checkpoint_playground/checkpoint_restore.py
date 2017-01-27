import tensorflow as tf
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp_MNIST_data/", one_hot=True)

import pdb

def debug1():
    print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    print( [ op.name for op in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)] )

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
    # accuracy
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # list of booleans indicating correct predictions
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # step for optimizer
    step = tf.Variable(0, name='step')
    # save everything that was saved in the session
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    saver.restore(sess=sess,save_path='./tmp/mdl_ckpt')
    #saver.restore(sess=sess, save_path='./tmp_all_ckpt/experiment_task_test1/job_mdl_nn10/hp_stid_2/mdl_ckpt')
    # evaluate
    print(step.eval())
    print(sess.run(fetches=accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
