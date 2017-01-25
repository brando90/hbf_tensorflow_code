import os

import namespaces as ns
import pdb

import numpy as np
import tensorflow as tf
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp_MNIST_data/", one_hot=True)


def make_and_check_dir(path):
    '''
        tries to make dir/file, if it exists already does nothing else creates it.
    '''
    try:
        os.makedirs(path)
    except OSError:
        # uncomment to make it raise an error when path is not a directory
        #if not os.path.isdir(path):
        #    raise
        pass
#

'''
Note the idea of structure for a ckpt files is as follow:

E.g.
./all_ckpts/exp_task_name/mdl_NN10/{hp_stid_1/, ..., hp_stid_200/}/mdl_ckpt_file

so ./all_ckpts/exp_task_name/mdl_NN10/ holds the ckpts for job/mdl mdl_NN10. This
directory holds all the directories for each hyper param. As in mdl_NN10/{hp_stid_1/, ..., hp_stid_200/}.
Each hyper param has its own true tensorflow ckpt file. So for a single hyper param we have
mdl_NN10/hp_stid_1/mdl_ckpt_file.
'''

def ckpts_exist_for_job_mdl(path_to_task_exp):
    '''
    checks if the current job has had any checkpoints saved.
    looks in ./all_ckpts/expt_task/mdl_job and see if the directory exists to see
    if any ckpts directory structures has been created. Note it only checks if
    the directory e.g. ./all_ckpts/exp_task_name/mdl_job exists. If it doesn't
    its safe to run that specific mdl/job from scratch.

    Note: if the specific experiment ./all_ckpts/exp_task_name has not even been ran
    obviously neither has ./all_ckpts/exp_task_name/mdl_job been created, so the
    function returns false (correctly) since we need to start this experiemnt from
    scratch from the experiment folder (i.e. no job for this mdl has been even ran)

    path_to_task_exp = ./all_ckpts/exp_task_name/mdl_job
    '''
    return os.path.isdir(path_to_task_exp)

#

def get_hp_largest_stid(path_to_folder_with_hps_jobs):
    #pdb.set_trace()
    #For each directory in the tree rooted at directory top (including top itself), it yields a 3-tuple (dirpath, dirnames, filenames).
    for (dirpath, dirnames, filenames) in os.walk(top=path_to_folder_with_hps_jobs,topdown=True):
        # dirnames = [...,hp_stid_N,...] or []
        largest_stid = get_largest(hp_dirnames=dirnames)
        return largest_stid
    # if it gets here it means something bad happened and it starting all the hp's from the first one
    return -1

def get_hp_largest_stid2():
    # assuming the dir ./all_ckpts/expt_task_name/mdl_NN10/ is not empty and only has directories with hp_stid_N.
    dirnames = os.listdir() #dirnames = [...,hp_stid_N,...] or []
    largest_stid = get_largest(dirnames)
    return largest_stid

def get_largest(hp_dirnames):
    if len(hp_dirnames) == 0:
        return -1
    else:
        # since the hp_dirnames are in format hp_stid_N, try to extract the number and get largest
        largest_stid = np.max([ int(hp_dirname.split('_')[2]) for hp_dirname in hp_dirnames ])
        return largest_stid

def no_hp_exists(stid):
    # (stid == -1) means there were no hp that were previously ran
    return stid == -1

#

def get_latest_and_only_save_path_to_ckpt(largest_stid):
    # get the path where all the ckpts reside
    path_to_folder_with_ckpts = './%s/%s/%s/np_stid_%s'%(arg.root_cktps_folder, arg.experiment_name, arg.job_name, str(largest_stid))
    # now get the most recent (and only chekpoint) checkpoint
    save_path_to_ckpt = path_to_folder_with_ckpts+'/'+arg.prefix_ckpt
    return save_path_to_ckpt

def get_latest_save_path_to_ckpt(largest_stid):
    # TODO extend to have more than one ckpt
    return get_latest_and_only_save_path_to_ckpt(largest_stid)

#

def has_hp_iteration_ckpt(stid):
    # For each directory in the tree rooted at directory top (including top itself), it yields a 3-tuple (dirpath, dirnames, filenames).
    for (dirpath, dirnames, filenames) in os.walk(top=path_to_folder_with_hps_jobs,topdown=True):
        if dirpath == 'hp_stid_'+stid: # only processes (dirpath == ho_stid_stid) else: (nothing)
            # filenames = [mdl_ckpt] or [...,mdl_ckpt-N,...]
            nb_tf_ckpts = len(filenames)
            return nb_tf_ckpts > 0
    # it shouldn't really get here, but if it does it probably means we should start this hp from scratch
    return False

#

def main_ckpt(arg):
    '''
    '''
    # if (there is a ckpt structure for this experiment and job continue training) otherwise (start from scratch and run job for experiment)
    if ckpts_exist_for_job_mdl(arg.root_cktps_folder+'/'+arg.experiment_name+'/'+arg.job_name): # if any of the dirs don't exist then start from scratch
        # continue training, since there is a ckpt for this experiment
        path_to_folder_with_hps_jobs = './%s/%s/%s/'%(arg.root_cktps_folder, arg.experiment_name, arg.job_name)
        largest_stid = get_hp_largest_stid(path_to_folder_with_hps_jobs) # it can be -1
        print('largest_stid: ', largest_stid)
        if no_hp_exists(largest_stid): # (stid == -1) means there were no hp that were previously ran
            # start training hp from scratch
            arg.start_stid = 1
            arg.end_stid = arg.nb_array_jobs
            arg.restore = False
            # note we didn't create a hp_dir cuz the loop that deals with specific hp's does it at some point (in train)
        else:
            # if (a ckpt iteration exists continue from it) otherwise (start hp iteration from scratch)
            if has_hp_iteration_ckpt(largest_stid):
                #start from this specific hp from scratch
                arg.start_stid = largest_stid
                arg.end_stid = arg.nb_array_jobs
                arg.save_path_to_ckpt2restore = get_latest_save_path_to_ckpt(largest_stid) # /task_exp_name/mdl_nn10/hp_stid_N/ckptK
            else:
                # else continue from the most recent iteration ckpt
                arg.start_stid = largest_stid
                arg.end_stid = arg.nb_array_jobs
                arg.save_path_to_ckpt2restore = get_latest_save_path_to_ckpt(largest_stid) # /task_exp_name/mdl_nn10/hp_stid_N/ckptK
    else:
        #start from scratch, since there wasn't a ckpt structure for this experiment
        arg.start_stid = 1
        arg.end_stid = arg.nb_array_jobs
        arg.restore = False
        make_and_check_dir(path=arg.root_cktps_folder+'/'+arg.experiment_name+'/'+arg.job_name) #first create and make the ckpt directory
    run_hyperparam_search(arg)

##

def run_hyperparam_search(arg):
    #do hyper_params
    SLURM_ARRAY_TASK_IDS = list(range(int(arg.start_stid),int(arg.end_stid+1)))
    for job_array_index in SLURM_ARRAY_TASK_IDS:
        scope_name = 'stid_'+str(job_array_index)
        with tf.variable_scope(scope_name):
            arg.slurm_array_task_id = job_array_index
            train(arg)

#

def get_mdl(x):
    # get model
    W = tf.Variable(tf.truncated_normal([784, 10], mean=0.0, stddev=0.1),name='w')
    b = tf.Variable(tf.constant(0.1, shape=[10]),name='b')
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    return y

def train(arg):
    # placeholder for data
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    #
    y = get_mdl(x)
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
        # if (there is a restore ckpt mdl restore it) else (create a structure to save ckpt files)
        if arg.restore:
            saver.restore(sess=sess, save_path=arg.save_path_to_ckpt2restore) # e.g. saver.restore(sess=sess, save_path='./tmp/my-model')
        else:
            make_and_check_dir(path=arg.get_ckpt_structure(arg)) # creates ./all_ckpts/exp_task_name/mdl_nn10/hp_stid_N
            sess.run(tf.global_variables_initializer())
        for i in range(1001):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(fetches=train_step, feed_dict={x: batch_xs, y_: batch_ys})
            # check_point mdl
            if i % 200 == 0:
                # Append the step number to the checkpoint name:
                saver.save(sess=sess,save_path=arg.get_save_path(arg))
        # evaluate
        print(sess.run(fetches=accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#

def get_args_for_experiment():
    arg = ns.Namespace()
    #
    arg.nb_array_jobs = 3
    #
    arg.root_cktps_folder = 'tmp_all_ckpt'
    arg.experiment_name = 'experiment_task_test1'
    arg.job_name = 'job_mdl_nn10'
    arg.prefix_ckpt = 'mdl_ckpt' # checkpoint prefix
    #arg.save_path = './%s/%s/%s/%s'%(arg.root_cktps_folder, arg.experiment_name, arg.job_name, arg.prefix_ckpt) # ./all_ckpts/exp_task_name/mdl_nn10/hp_stid_N/ckptK
    arg.get_save_path = lambda arg: './%s/%s/%s/%s/%s'%(arg.root_cktps_folder, arg.experiment_name, arg.job_name, 'hp_stid_'+str(arg.slurm_array_task_id), arg.prefix_ckpt)
    arg.get_ckpt_structure = lambda arg: './%s/%s/%s/%s'%(arg.root_cktps_folder, arg.experiment_name, arg.job_name, 'hp_stid_'+str(arg.slurm_array_task_id))
    return arg

if __name__ == '__main__':
    arg = get_args_for_experiment()
    main_ckpt(arg)
