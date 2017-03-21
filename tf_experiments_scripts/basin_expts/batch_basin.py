#!/usr/bin/env python
#SBATCH --mem=8000
#SBATCH --time=4-18:20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brando90@mit.edu

#from __future__ import #print_function
#tensorboard --logdir=/tmp/mdl_logs
#

print('#!/usr/bin/env python')
print('#!/usr/bin/python')

import os
import sys

#import pickle
import namespaces as ns
import argparse
import pdb
import functools

import numpy as np
import tensorflow as tf

import my_tf_pkg as mtf
from my_tf_pkg import main_hp
from my_tf_pkg import main_large_hp_checkpointer as large_main_hp
import sgd_lib

##
#print('In batch script', flush=True)
#print = functools.partial(print, flush=True)
#print(ns)
###
arg = ns.Namespace()

#
arg.get_errors_from = mtf.get_errors_based_on_train_error
#arg.get_errors_from = mtf.get_errors_based_on_validation_error
#

#arg.type_job, arg.nb_array_jobs = 'serial', 1 #careful when this is on and GPU is NOT on
arg.type_job = 'slurm_array_parallel'
#arg.type_job, arg.nb_array_jobs = 'main_large_hp_ckpt', 200
#arg.save_checkpoints = True
arg.save_checkpoints = False
#arg.save_last_mdl = True
arg.save_last_mdl = False

## debug mode
arg.data_dirpath = './data/' # path to datasets
prefix_path_sim_results = './tmp_simulation_results_scripts/%s/%s/' # folder where the results from script is saved
prefix_path_ckpts = './tmp_all_ckpts/%s/%s/' # folder where the results from script is saved
## to run locally: python batch_main.py -sj sj
#arg.data_dirpath = '../data/' # path to datasets
#prefix_path_sim_results = '../../../simulation_results_scripts/%s/%s/' # folder where the results from script is saved
#prefix_path_ckpts = '../../../all_ckpts/%s/%s/' # folder where the results from script is saved
## to run in docker
#arg.data_dirpath = '/home_simulation_research/hbf_tensorflow_code/tf_experiments_scripts/data/' # path to datasets
#prefix_path_sim_results = '/home_simulation_research/simulation_results_scripts/%s/%s/' # folder where the results from script is saved
#prefix_path_ckpts = '/home_simulation_research/all_ckpts/%s/%s/' # folder where the results from script is saved

# prefix_path_sim_results = '../../simulation_results_scripts/%s/%s'
# prefix_path_ckpts = '../../all_ckpts/%s/%s' # folder where the results from script is saved
arg.get_path_root =  lambda arg: prefix_path_sim_results%(arg.experiment_root_dir,arg.experiment_name)
arg.get_path_root_ckpts =  lambda arg: prefix_path_ckpts%(arg.experiment_root_dir,arg.experiment_name)

arg.prefix_ckpt = 'mdl_ckpt'
####
arg.data_filename = 'basin_expt'
#arg.data_filename = 'f_64D_product_binary'
#arg.task_folder_name = mtf.get_experiment_folder(arg.data_filename) #om_f_4d_conv
#arg.type_preprocess_data = None
#
#arg.N_frac = 60000

#arg.classificaton = mtf.classification_task_or_not(arg)

arg.experiment_name = 'task_Mar_12_1D_basin_dgx1'
#arg.job_name = 'BT_debug1'
arg.job_name = 'basin_1D'

#
arg.experiment_root_dir = mtf.get_experiment_folder(arg.data_filename)
#
arg.mdl = 'basin_1D'
if arg.mdl == 'debug_mdl':
    arg.act = tf.nn.relu
    arg.dims = None
    arg.get_dims = lambda arg: arg.dims
    arg.get_x_shape = lambda arg: [None,arg.D]
    arg.get_y_shape = lambda arg: [None,arg.D_out]
elif arg.mdl == 'basin_1D':
    arg.mdl_scope_name = arg.mdl
    arg.D = 1
    arg.get_x_shape = lambda arg: arg.D
    #
    arg.init_std = lambda: tf.constant([2.0,1.0])
    arg.init_mu = lambda: tf.constant([-2.2,9.0])
    arg.init_W = lambda: tf.constant([5.1345],shape=[1,1])
    #arg.init_W = lambda: tf.constant(0.0)
    def get_basins(arg):
        #pdb.set_trace()
        W = tf.get_variable(name='W', initializer=arg.init_W(), trainable=True)
        print('==> W.name', W.name)
        tf.summary.histogram('Weights', W)
        #tf.summary.scalar('Weights_scal', W)
        #
        init_std = arg.init_std()
        init_mu = arg.init_mu()
        #
        basin1 = sgd_lib.get_basin(W,init_std[0],init_mu[0],str(1))
        basin2 = sgd_lib.get_basin(W,init_std[1],init_mu[1],str(2))
        basins = [basin1, basin2]
        return basins
    arg.get_basins = get_basins

#arg.get_y_shape = lambda arg: [None, arg.D_out]
# float type
arg.float_type = tf.float32
#steps
#arg.steps_low = int(2.5*60000)
arg.steps_low = 10*int(1.3*10001)
arg.steps_high = arg.steps_low+1
arg.get_steps = lambda arg: int( np.random.randint(low=arg.steps_low ,high=arg.steps_high) )

arg.M_low = 32
arg.M_high = 33
arg.get_batch_size = lambda arg: int(np.random.randint(low=arg.M_low , high=arg.M_high))
#arg.potential_batch_sizes = [16,32,64,128,256,512,1024]
#arg.potential_batch_sizes = [4]
def get_power2_batch_size(arg):
    i = np.random.randint( low=0, high=len(arg.potential_batch_sizes) )
    batch_size = arg.potential_batch_sizes[i]
    return batch_size
#arg.get_batch_size = get_power2_batch_size
arg.report_error_freq = 10

arg.low_log_const_learning_rate, arg.high_log_const_learning_rate = -0.5, -4
arg.get_log_learning_rate =  lambda arg: np.random.uniform(low=arg.low_log_const_learning_rate, high=arg.high_log_const_learning_rate)
arg.get_start_learning_rate = lambda arg: 10**arg.log_learning_rate
## decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
#arg.decay_rate_low, arg.decay_rate_high = 0.1, 1.0
#arg.get_decay_rate = lambda arg: np.random.uniform(low=arg.decay_rate_low, high=arg.decay_rate_high)
arg.get_start_learning_rate = lambda arg: 0.02
arg.get_decay_rate = lambda arg: 1.0

#arg.decay_steps_low, arg.decay_steps_high = arg.report_error_freq, arg.M
#arg.get_decay_steps_low_high = lambda arg: arg.report_error_freq, arg.M
#arg.get_decay_steps = lambda arg: np.random.randint(low=arg.decay_steps_low, high=arg.decay_steps_high)
def get_decay_steps(arg):
    #arg.decay_steps_low, arg.decay_steps_high = arg.report_error_freq, arg.M
    arg.decay_steps_low, arg.decay_steps_high = 1000, 15000
    decay_steos = np.random.randint(low=arg.decay_steps_low, high=arg.decay_steps_high)
    return decay_steos
arg.get_decay_steps = get_decay_steps # when stair case, how often to shrink

#arg.staircase = False
arg.staircase = True

#optimization_alg = 'GD'
#optimization_alg = 'Momentum'
#optimization_alg = 'Adadelta'
#optimization_alg = 'Adagrad'
#optimization_alg = 'Adam'
#optimization_alg = 'RMSProp'
optimization_alg = 'GDL'
arg.optimization_alg = optimization_alg

if optimization_alg == 'GD':
    pass
elif optimization_alg=='Momentum':
    #arg.get_use_nesterov = lambda: False
    arg.get_use_nesterov = lambda: True
    arg.momentum_low, arg.momontum_high = 0.01, 0.99
    arg.get_momentum = lambda arg: np.random.uniform(low=arg.momentum_low,high=arg.momontum_high)
elif optimization_alg == 'Adadelta':
    arg.rho_low, arg.rho_high = 0.1, 0.99
    arg.get_rho = lambda arg: np.random.uniform(low=arg.rho_low,high=arg.rho_high)
elif optimization_alg == 'Adagrad':
    #only has learning rate
    pass
elif optimization_alg == 'Adam':
    arg.beta1 = 0.99
    arg.beta2 = 0.999
    arg.get_beta1 = lambda arg: arg.beta1 # m = b1m + (1 - b1)m
    arg.get_beta2 = lambda arg: arg.beta2 # v = b2 v + (1 - b2)v
    #arg.beta1_low, arg.beta1_high = beta1_low=0.7, beta1_high=0.99 # m = b1m + (1 - b1)m
    #arg.beta2_low, arg.beta2_high = beta2_low=0.8, beta2_high=0.999 # v = b2 v + (1 - b2)v
elif optimization_alg == 'RMSProp':
    arg.decay_loc, arg.decay_high = 0.75, 0.99
    arg.get_decay = lambda arg: np.random.uniform(low=arg.decay_loc,high=arg.decay_high)
    arg.momentum_low, arg.momontum_high = 0.0, 0.99
    arg.get_momentum = lambda arg: np.random.uniform(low=arg.momentum_low,high=arg.momontum_high)
elif optimization_alg == 'GDL':
    arg.get_gdl_mu_noise =  lambda arg: 0.0
    arg.get_gdl_stddev_noise = lambda arg: 6.0
else:
    pass

#saving pickle histogram data
arg.nb_bins = 35
#arg.p_path = './tmp_pickle'
arg.p_path = './tmp_om_pickle'
#arg.p_path = './folder_pickle_W_hist'
arg.p_filename = 'W_hist_data.p'
#arg.save_hist = False
arg.save_hist = True
arg.display_hist = False
#arg.display_hist = True

#arg.bn = True
#arg.trainable_bn = True #scale, shift BN
arg.bn = False
arg.trainable_bn = False #scale, shift BN

# NORMALIZE UNIT CIRCLE
arg.data_normalize='normalize_input'
arg.data_normalize='dont_normalize'

re_train = None
arg.re_train = re_train
#
arg.save_config_args = False
#arg.debug = False
# arg.slurm_jobid = os.environ['SLURM_JOBID']
# arg.slurm_array_task_id = os.environ['SLURM_ARRAY_TASK_ID']

####
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", help="debug mode: loads the old (pickle) config file to run in debug mode", action='store_true')
parser.add_argument("-tj", "--type_job", help="type_job for run")
parser.add_argument("-sj", "--SLURM_JOBID", help="SLURM_JOBID for run")
parser.add_argument("-stid", "--SLURM_ARRAY_TASK_ID", help="SLURM_ARRAY_TASK_ID for run.")
parser.add_argument("-sca", "--save_config_args", help="save_config_args saves the current config file to a picke file ")

parser.add_argument("-tb", "--tensorboard", dest='tensorboard', help="tensorboard mode: save tensorboard data", action='store_true')
parser.add_argument('-notb', "--no-tensorboard", dest='tensorboard', help="tensorboard mode: save tensorboard data", action='store_false')
parser.set_defaults(tensorboard=False)

parser.add_argument("-pr", "--path_root", help="path_root: specifies the path root to save hyper params")
#parser.add_argument("-hp", "--hyper_parameter", help="hyper_parameter: when searching for hp on needs to specify were to store the results of experiment or it will default.")

cmd_args = parser.parse_args()
# (do if first if statment is true) if (return true when cmd_args is initialized to None) else (do this)
cmd_args.type_job = cmd_args.type_job if cmd_args.type_job else arg.type_job
# if the flag is initialized (not None) then use it, otherwise use the flag from environment veriable
arg.slurm_jobid = cmd_args.SLURM_JOBID if cmd_args.SLURM_JOBID else os.environ['SLURM_JOBID']

######## try: TODO: fix this
#     #this should only succeed when slurm is being called in parallel with an array id i.e. if (arg.type_job == 'slurm_array_parallel') is true.
#     arg.slurm_array_task_id = cmd_args.SLURM_ARRAY_TASK_ID if cmd_args.SLURM_ARRAY_TASK_ID else os.environ['SLURM_ARRAY_TASK_ID']
# except:
#     if arg.type_job == 'slurm_array_parallel':
#         # stry should have been succesful: arg.slurm_array_task_id = cmd_args.SLURM_ARRAY_TASK_ID if cmd_args.SLURM_ARRAY_TASK_ID else os.environ['SLURM_ARRAY_TASK_ID']
#         pass
#     elif arg.type_job == 'slurm_array_parallel':
#         arg.slurm_array_task_id = arg.type_job
if arg.type_job == 'slurm_array_parallel':
    arg.slurm_array_task_id = cmd_args.SLURM_ARRAY_TASK_ID if cmd_args.SLURM_ARRAY_TASK_ID else os.environ['SLURM_ARRAY_TASK_ID']
else:
    arg.slurm_array_task_id = arg.type_job
#print('--> arg ', arg.slurm_jobid, arg.slurm_array_task_id)
#########

if cmd_args.save_config_args:
    # flag to save current config files to pickle file
    #print(cmd_args.save_config_args)
    arg.save_config_args = cmd_args.save_config_args
if cmd_args.debug:
    #arg.debug = cmd_args.debug
    # load old pickle config
    # pickled_arg_dict = pickle.load( open( "pickle-slurm-%s_%s.p"%(int(arg.slurm_jobid)+int(arg.slurm_array_task_id),arg.slurm_array_task_id), "rb" ) )
    # #print( pickled_arg_dict )
    # # values merged with the second dict's values overwriting those from the first.
    # arg_dict = {**dict(arg), **pickled_arg_dict}
    # arg = ns.Namespace(arg_dict)
    print('EMPTY IF STATEMENT') #TODO fix line above that gives syntax error
# if cmd_args.path_root:
#     arg.path_root = cmd_args.path_root
#     arg.get_path_root =  lambda arg: arg.path_root

#
arg.mdl_save = False
#arg.mdl_save = True

#
arg.use_tensorboard = cmd_args.tensorboard
arg.tensorboard_data_dump_train = './tmp/train'

arg.max_to_keep = 1

arg.get_dataset = lambda arg: (arg.X_train, arg.Y_train, arg.X_cv, arg.Y_cv, arg.X_test, arg.Y_test)

#arg.act_name = arg.act.__name__
arg.restore = False

#
arg.print_func = print
if arg.slurm_array_task_id == '1':
    print = functools.partial(print, flush=True)
    arg.print_func = print
#pickle.dump( dict(arg), open( "pickle_file" , "wb" ) )
#pdb.set_trace()
print(arg)
if __name__ == '__main__':
    #print('In __name__ == __main__')
    if cmd_args.type_job == 'serial':
        # jobs one job. No slurm array
        #arg.start_stid = 1
        #arg.end_stid = arg.nb_array_jobs
        #large_main_hp.run_hyperparam_search2(arg)
        raise ValueError('this cmd_args.type_job is not valid cuz it was %s'%(cmd_args.type_job))
    elif cmd_args.type_job == 'slurm_array_parallel':
        # run one single job according to slurm array command
        sgd_lib.main_basin(arg)
    elif cmd_args.type_job == 'main_large_hp_ckpt':
        #large_main_hp.main_large_hp_ckpt(arg)
        raise ValueError('this cmd_args.type_job is not valid cuz it was %s'%(cmd_args.type_job))
