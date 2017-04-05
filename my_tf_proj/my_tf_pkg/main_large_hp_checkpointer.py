import os
#from multiprocessing import Pool
from multiprocessing import Process
import contextlib
import time

import maps
import pdb
import functools

#import my_tf_pkg as mtf
from my_tf_pkg import main_hp

import numpy as np
import tensorflow as tf
# download and install the MNIST data automatically
from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("tmp_MNIST_data/", one_hot=True)

print_func_flush_true = functools.partial(print, flush=True) # TODO fix hack

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
    print('=======> path_to_task_exp',path_to_task_exp)
    return os.path.isdir(path_to_task_exp)

#

def get_hp_largest_stid(path_to_folder_with_hps_jobs):
    '''
    goes to the experiment with the current model folder (as in /all_ckpts/expt_task_name/mdl_nn10/)
    and gets the most recent stid of the hps.
    '''
    print('>>path_to_folder_with_hps_jobs: ', path_to_folder_with_hps_jobs)
    #For each directory in the tree rooted at directory top (including top itself), it yields a 3-tuple (dirpath, dirnames, filenames).
    for (dirpath, dirnames, filenames) in os.walk(top=path_to_folder_with_hps_jobs,topdown=True):
        # dirnames = [...,hp_stid_N,...] or []
        print('dirnames: ', dirnames)
        largest_stid = get_largest(hp_dirnames=dirnames)
        return largest_stid
    # if it gets here it means something bad happened and it starting all the hp's from the first one
    return -1

def get_hp_largest_stid2(path_to_hp):
    '''
    goes to the experiment with the current model folder (as in /all_ckpts/expt_task_name/mdl_nn10/)
    and gets the most recent stid of the hps.

    shows an alternative way to run code using os.listdir(), this way we specify
    the directory directly and get the list of id's we need. They are basically
    identical because if the path is correct for either code, the loop for
    the other code won't even execture because it returns.
    '''
    # assuming the dir ./all_ckpts/expt_task_name/mdl_NN10/ is not empty and only has directories with hp_stid_N.
    dirnames = os.listdir(path_to_hp) #dirnames = [...,hp_stid_N,...] or []
    largest_stid = get_largest(dirnames)
    return largest_stid

def get_largest(hp_dirnames):
    '''
    gets the largest stid of from the hyper params that have been ran. Note that
    becuase the folder structure has been been decided, it can always extract
    the stid from the folder name (basically the end is has the stid).

    e.g.
    - get_largest([hp_stid_1,...,hp_stid_k,...hp_stid_100]) = 100
    - get_largest([]) = -1

    note: because I am extremely paranoid about counters sometimes starting from 1 or 0
    I explicitly choose stid=0 not to mean emptiness.
    '''
    if len(hp_dirnames) == 0:
        return -1
    else:
        # since the hp_dirnames are in format hp_stid_N, try to extract the number and get largest
        largest_stid = np.max([ int(hp_dirname.split('_')[2]) for hp_dirname in hp_dirnames ]) # extract the stid from string 'hp_stid_N' and then chooses the largest one
        return largest_stid

def no_hp_exists(stid):
    '''
    stid map to hp (hyper param) runs that we have made. They are number from
    1 to N. If no hp has been ran then stid=-1. Thus, this returns true when stid=-1
    since that means when there are no hp's (i.e. empty [] list of hps) the stid=-1.

    note: because I am extremely paranoid about counters sometimes starting from 1 or 0
    I explicitly choose stid=0 not to mean emptiness.
    '''
    # (stid == -1) means there were no hp that were previously ran
    return stid == -1

#

def get_latest_and_only_save_path_to_ckpt(arg, largest_stid):
    '''
    gets the path to ckpts
    '''
    #save_path_to_ckpt = arg.path_to_ckpt+arg.hp_folder_for_ckpt+arg.prefix_ckpt
    save_path_to_ckpt = arg.path_to_ckpt+'hp_stid_%s/%s'%(largest_stid,arg.prefix_ckpt)
    return save_path_to_ckpt

def get_latest_save_path_to_ckpt(arg, largest_stid):
    '''
    this was made with the idea that the ckpt files would be appended with
    maybe a counter or something to start ckpts from the most recent ckpt but
    save more than 1 ckpt. At this point there is only one chpt saved per hp
    and the ckpt itself keeps track of the iteration of trainin to continue from.
    '''
    # TODO extend to have more than one ckpt
    return get_latest_and_only_save_path_to_ckpt(arg, largest_stid)

#

def does_hp_have_tf_ckpt(path_to_hp):
    '''
    check if the current hp with stid has some iteration ckpt tensorflow file.
    '''
    # For each directory in the tree rooted at directory top (including top itself), it yields a 3-tuple (dirpath, dirnames, filenames).
    for (dirpath, dirnames, filenames) in os.walk(top=path_to_hp,topdown=True):
        # filenames = [mdl_ckpt] or [...,mdl_ckpt-N,...]
        nb_tf_ckpts = len(filenames)
        return nb_tf_ckpts > 0
    # it shouldn't really get here, but if it does it probably means we should start this hp from scratch
    return False

#

def main_large_hp_ckpt(arg,ckpt_arg):
    '''
    Runs all the hyper params the we require and restores models if they were
    interupted at any point by slurm.

    How it works: first if its a model from scratch it tries to check if the experiment/job exists (by looking at existence of folders),
    if it does not exist then it creates the folders and start training and saving model from scratch. If it does exist then it should
    check if any hp has been ran in the past. If no hp has been ran then it starts from scratch i.e stid=1.
    Else, if some hp's have been ran in the past it gets the latest one and continues from there.
    Note that if the most recent hp has a checkpoint it always loads the checkpoint and continues training
    from that checkpoint but then proceeds to do other hps if there are more to do.

    note: so if an experiment is re-ran exactly by accident only the last job/hp
    is re-ran but from the last iteration. Thus, the model might change but the
    change should be insignificant.
    '''
    arg = put_ckpt_args_to_args(arg,ckpt_arg)
    ##
    current_job_mdl_folder = 'job_mdl_folder_%s/'%arg.job_name
    #pdb.set_trace()
    arg.path_to_ckpt = arg.get_path_root_ckpts(arg)+current_job_mdl_folder
    # if (there is a ckpt structure for this experiment and job continue training) otherwise (start from scratch and run job for experiment)
    if ckpts_exist_for_job_mdl(arg.path_to_ckpt): # if any of the dirs don't exist then start from scratch
        # continue training, since there is a ckpt for this experiment
        print('>>>some chekpoint exists')
        path_to_folder_with_hps_jobs = arg.path_to_ckpt
        largest_stid = get_hp_largest_stid(path_to_folder_with_hps_jobs) # it can be -1
        print('largest_stid: ', largest_stid)
        if no_hp_exists(largest_stid): # (stid == -1) means there were no hp that were previously ran
            # start training hp from scratch
            ckpt_arg.start_stid = 1
            ckpt_arg.end_stid = arg.nb_array_jobs
            ckpt_arg.restore = False
            # note we didn't create a hp_dir cuz the loop that deals with specific hp's does it at some point (in train)
        else:
            path_to_hp_folder = arg.path_to_ckpt+'/hp_stid_'+str(largest_stid)
            # if (a ckpt iteration exists continue from it) otherwise (start hp iteration from scratch)
            if does_hp_have_tf_ckpt(path_to_hp_folder): # is there a tf ckpt for this hp?
                # start from this specific hp ckpt
                print('>>>found tf ckpt')
                ckpt_arg.start_stid = largest_stid
                ckpt_arg.end_stid = arg.nb_array_jobs
                ckpt_arg.restore = True
                ckpt_arg.save_path_to_ckpt2restore = get_latest_save_path_to_ckpt(arg,largest_stid) # /task_exp_name/mdl_nn10/hp_stid_N/ckptK
                print('>>>arg.save_path_to_ckpt2restore', ckpt_arg.save_path_to_ckpt2restore)
            else:
                # train hp from the first iteration
                print('>>>NOT found tf ckpt')
                ckpt_arg.start_stid = largest_stid
                ckpt_arg.end_stid = arg.nb_array_jobs
                ckpt_arg.restore = False
                #pdb.set_trace()
    else:
        #start from scratch, since there wasn't a ckpt structure for this experiment
        print('>>>Nothing has been run before so running something from scratch.')
        ckpt_arg.start_stid = 1
        ckpt_arg.end_stid = arg.nb_array_jobs
        ckpt_arg.restore = False
        make_and_check_dir(path=arg.path_to_ckpt) #first create and make the ckpt directory
    #pdb.set_trace()
    run_hyperparam_search2(arg,ckpt_arg)

##

# def run_hyperparam_search(arg):
#     '''
#     Runs all the hp (hyper param) jobs that it needs to run.
#     Assume that the code calling it has figured out weather it has to restore
#     a model or not. If it does have to restore a model then the stid should be
#     initialized correctly so that it doesn't overwrite old ckpts.
#     '''
#     #do hyper_params
#     SLURM_ARRAY_TASK_IDS = list(range(int(arg.start_stid),int(arg.end_stid+1)))
#     for job_array_index in SLURM_ARRAY_TASK_IDS:
#         scope_name = 'stid_'+str(job_array_index)
#         print('--> stid: ',job_array_index)
#         #with tf.variable_scope(scope_name):
#         arg.slurm_array_task_id = job_array_index
#         # trains the current hp
#         main_hp.main_hp(arg)

def run_hyperparam_search2(arg,ckpt_arg):
    '''
    Runs all the hp (hyper param) jobs that it needs to run.
    Assume that the code calling it has figured out weather it has to restore
    a model or not. If it does have to restore a model then the stid should be
    initialized correctly so that it doesn't overwrite old ckpts.
    '''
    #do hyper_params
    #pdb.set_trace()
    arg = put_ckpt_args_to_args(arg,ckpt_arg)
    SLURM_ARRAY_TASK_IDS = list(range(int(arg.start_stid),int(arg.end_stid+1)))
    for job_array_index in SLURM_ARRAY_TASK_IDS:
        print('\n')
        scope_name = 'stid_'+str(job_array_index)
        print('--> stid: ',job_array_index)
        #with tf.variable_scope(scope_name):
        arg = arg.get_arg_for_experiment()
        arg = put_ckpt_args_to_args(arg,ckpt_arg)
        arg.slurm_array_task_id = job_array_index
        # throw out process so that many tensorflow gpus can be used serially
        p = Process(target=main_hp.main_hp, args=(arg,))
        p.start()
        p.join()
        ckpt_arg.restore = False # after the model has been restored, we continue normal until all hp's are finished
        print('--> Done!!! with stid: ',job_array_index)


def put_ckpt_args_to_args(arg,ckpt_arg):
    '''
    load ckpt_arg to arg.
    '''
    for key, value in ckpt_arg.items():
        arg[key] = value
    return arg
#

def get_args_for_experiment_test():
    arg = maps.NamedDict()
    #
    arg.nb_array_jobs = 3
    arg.nb_iterations = 2001
    arg.get_batch_size = lambda arg: arg.slurm_array_task_id*10
    #arg.batch_size = 100
    #
    arg.root_experiment_folder = 'tmp_sim_results'
    arg.root_cktps_folder = 'tmp_all_ckpt'
    #
    arg.experiment_name = 'experiment_task_test1'
    arg.job_name = 'job_mdl_nn10'
    arg.prefix_ckpt = 'mdl_ckpt' # checkpoint prefix
    #arg.save_path = './%s/%s/%s/%s'%(arg.root_cktps_folder, arg.experiment_name, arg.job_name, arg.prefix_ckpt) # ./all_ckpts/exp_task_name/mdl_nn10/hp_stid_N/ckptK
    arg.get_save_path = lambda arg: './%s/%s/%s/%s/%s'%(arg.root_cktps_folder, arg.experiment_name, arg.job_name, 'hp_stid_'+str(arg.slurm_array_task_id), arg.prefix_ckpt)
    arg.get_hp_ckpt_structure = lambda arg: './%s/%s/%s/%s'%(arg.root_cktps_folder, arg.experiment_name, arg.job_name, 'hp_stid_'+str(arg.slurm_array_task_id))
    return arg
