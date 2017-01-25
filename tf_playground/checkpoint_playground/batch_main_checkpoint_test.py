import tensorflow as tf
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp_MNIST_data/", one_hot=True)

import namespaces as ns
import pdb

import np

def run_hyperparam_search(arg):
    #do jobs
    SLURM_ARRAY_TASK_IDS = list(range(int(arg.nb_array_jobs)))
    for job_array_index in SLURM_ARRAY_TASK_IDS:
        scope_name = 'stid_'+str(job_array_index)
        with tf.variable_scope(scope_name):
            arg.slurm_array_task_id = job_array_index
            main_nn(arg)

def ckpts_exist_for_job_mdl(path_to_task_exp):
    '''
    checks if the current job has had any checkpoints saved.
    looks in ./all_ckpts/expt_task/mdl_job and see if the directory exists to see
    if any ckpts has been created.

    path_to_task_exp = ./all_ckpts/exp_task_name
    '''
    return os.path.isdir(path_to_task_exp)

def get_hp_largest_stid(path_to_folder_with_hps_jobs):
    # For each directory in the tree rooted at directory top (including top itself), it yields a 3-tuple (dirpath, dirnames, filenames).
    for (dirpath, dirnames, filenames) in os.walk(top=path_to_folder_with_hps_jobs,topdown=True):
        if dirpath == arg.job_name: # only processes (dirpath == mdl_nn10) else: (nothing)
            # dirnames = [...,hp_stid_N,...] or []
            largest_stid = get_largest(hp_dirnames=dirnames)
            return largest_stid
    # if it gets here it means there where no hp's that have been ran. So start from scratch
    return -1

def get_largest(hp_dirnames):
    if len(hp_dirnames) == 0:
        return -1
    else:
        # since the hp_dirnames are in formapt hp_stid_N, try to extract the number
        largest_stid = np.max([ int(hp_dirname[2]) for hp_dirname in hp_dirnames ])
        return largest_stid
#

def get_args_for_experiment():
    arg = ns.Namespace()
    #
    arg.root_cktps_folder = './tmp_all_ckpt'
    arg.experiment_name = 'task_test1'
    arg.job_name = 'mdl_nn10'
    arg.checkpoint_prefix = 'mdl_ckpt'
    arg.save_path = './%s/%s/%s/%s'%(arg.root_cktps_folder, arg.experiment_name, arg.job_name, arg.checkpoint_prefix) # ./all_ckpts/exp_task_name/mdl_nn10/hp_stid_N/ckptK
    return arg

def main_ckpt(arg):
    '''
    '''
    arg = get_args_for_experiment()
    # if (there is a ckpt for this experiment continue training) otherwise (start from scratch)
    if ckpts_exist_for_job_mdl(arg.root_cktps_folder+'/'+arg.experiment_name+'/'+arg.job_name): #/all_ckpts/exp_task_name
        # continue training, since there is a ckpt for this experiment
        path_to_folder_with_hps_jobs = './%s/%s/%s/'%(arg.root_cktps_folder, arg.experiment_name, arg.job_name)
        stid = get_hp_largest_stid(path_to_folder_with_hps_jobs) # it can be -1
        if no_hp_exists(stid): # (stid == -1) means need to start hp from scratch
            # start straining hp from scratch
            arg.start_stid = 1
            arg.nb_array_jobs = arg.nb_array_jobs
        else:
            # continue from most recent iteration of that hp
            arg.start_stid = stid
            save_path_to_restored_mdl = get_path_mdl_to_restore() # /task_exp_name/mdl_nn10/hp_stid_N/ckptK
    else:
        #start from scratch, since there wasn't a ckpt for this experiment

##

def get_mdl():
    # get model
    W = tf.Variable(tf.truncated_normal([784, 10], mean=0.0, stddev=0.1),name='w')
    b = tf.Variable(tf.constant(0.1, shape=[10]),name='b')
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    return y

def main_nn(arg):
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
