import tensorflow as tf
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp_MNIST_data/", one_hot=True)

import namespaces as ns
import pdb

arg = ns.Namespace()

def get_mdl():
    # get model
    W = tf.Variable(tf.truncated_normal([784, 10], mean=0.0, stddev=0.1),name='w')
    b = tf.Variable(tf.constant(0.1, shape=[10]),name='b')
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    return y

def check_for_most_recent_mdl(save_path):
    for (dirpath, dirnames, filenames) in os.walk(top=save_path,topdown=True):

def main_ckpt(arg):

    # if (there is a ckpt for this experiment continue training) otherwise (start from scratch)
    if check_point_for_experiment(arg.experiment_name):
        # continue training, since there is a ckpt for this experiment
        stid = get_hp_largest_stid() # it can be -1
        if no_more_recent_hp(stid): # stid == -1 means need to start hp from scratch
            # start straining hp from scratch
        else:
            # continue from most recent iteration of that hp
    else:
        #start from scratch, since there wasn't a ckpt for this experiment

    #
    # check for most recent model
    checkpoints_folder = 'task_name/model1'
    checkpoint_filename = 'mdl_ckpt' # its more of a prefix as in mdl_ckpt/data or mdl_ckpt.index or mdl_ckpt.meta
    save_path = './%s/my-model'%checkpoints_folder
    check_for_most_recent_mdl(save_path)
    #do jobs
    SLURM_ARRAY_TASK_IDS = list(range(int(arg.nb_array_jobs)))
    for job_array_index in SLURM_ARRAY_TASK_IDS:
        scope_name = 'stid_'+str(job_array_index)
        with tf.variable_scope(scope_name):
            arg.slurm_array_task_id = job_array_index
            main_nn(arg)


def main_nn(arg):
    y = get_mdl()
    # placeholder for data
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    # loss and accuracy
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # list of booleans indicating correct predictions
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # optimizer
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    #
    saver = tf.train.Saver()
    # train and evalaute
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1001):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(fetches=train_step, feed_dict={x: batch_xs, y_: batch_ys})
            # check_point mdl
            if i % 200 == 0:
                # Append the step number to the checkpoint name:
                saver.save(sess=sess,save_path=save_path,global_step=i)
        # evaluate
        print(sess.run(fetches=accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
