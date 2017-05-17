#!/usr/bin/env python
#SBATCH --mem=5G
#SBATCH --mail-type=END
#SBATCH --mail-user=brando90@mit.edu
#SBATCH --ntask=1
#SBATCH --time=7-00:00
#SBATCH --array=1-200

#from __future__ import #print_function
#tensorboard --logdir=/tmp/mdl_logs
#

''' useful commands for slurm:
#SBATCH --array=1-200
#SBATCH --gres=gpu:1
'''

print('#!/usr/bin/env python')
print('#!/usr/bin/python')

import os
import sys
import platform

#import pickle
import maps
import argparse
import pdb
import functools

import numpy as np
import tensorflow as tf

import my_tf_pkg as mtf
from my_tf_pkg import main_hp
from my_tf_pkg import main_large_hp_checkpointer as large_main_hp


print( '===> os.listdir(.): ', os.listdir('.') )
print( 'os.getcwd(): ', os.getcwd())
print('os.path.dirname(os.path.abspath(__file__)): ', os.path.dirname(os.path.abspath(__file__)))


print_func_flush_true = functools.partial(print, flush=True) # TODO fix hack

## create args
arg = maps.NamedDict()

## test or train error
arg.get_errors_from = mtf.get_errors_based_on_train_error
#arg.get_errors_from = mtf.get_errors_based_on_validation_error

## use TensorBoard
#arg.use_tb = True
arg.use_tb = False

#arg.type_job, arg.nb_array_jobs = 'serial', 1 #careful when this is on and GPU is NOT on
arg.type_job, arg.nb_array_jobs = 'slurm_array_parallel', 1
#arg.type_job, arg.nb_array_jobs = 'main_large_hp_ckpt', 200
#arg.save_checkpoints = True
arg.save_checkpoints = False
#arg.save_last_mdl = True
arg.save_last_mdl = False

hostname = platform.node()
print('hostname: ', hostname)
print('in docker? ','IN_DOCKER_CONT' in os.environ)
if hostname == 'dhcp-18-189-23-174.dyn.mit.edu' or hostname == 'Yasmins-MacBook-Pro.local':
    ## debug mode
    #arg.data_dirpath = './data/' # path to datasets
    #prefix_path_sim_results = './tmp_simulation_results_scripts/%s/%s/' # folder where the results from script is saved
    #prefix_path_ckpts = './tmp_all_ckpts/%s/%s/' # folder where the results from script is saved
    #arg.tb_data_dump = './tb_dump' # folder where /train,/cv,/test tb stats are stored
    ## to run locally: python batch_main.py -sj sj
    # arg.data_dirpath = './data/' # path to datasets
    # prefix_path_sim_results = '../../simulation_results_scripts/%s/%s/' # folder where the results from script is saved
    # prefix_path_ckpts = '../../all_ckpts/%s/%s/' # folder where the results from script is saved
    # arg.tb_data_dump = '../../tb_dump' # folder where /train,/cv,/test tb stats are stored
    arg.data_dirpath = '/Users/brandomiranda/home_simulation_research/hbf_tensorflow_code/tf_experiments_scripts/data/' # path to datasets
    prefix_path_sim_results = '/Users/brandomiranda/home_simulation_research/simulation_results_scripts/%s/%s/' # folder where the results from script is saved
    prefix_path_ckpts = '/Users/brandomiranda/home_simulation_research/all_ckpts/%s/%s/' # folder where the results from script is saved
    arg.tb_data_dump = '/Users/brandomiranda/home_simulation_research/tb_dump' # folder where /train,/cv,/test tb stats are stored
else:
    # to run in OM
    arg.data_dirpath = '/om/user/brando90/home_simulation_research/hbf_tensorflow_code/tf_experiments_scripts/data/' # path to datasets
    prefix_path_sim_results = '/om/user/brando90/home_simulation_research/simulation_results_scripts/%s/%s/' # folder where the results from script is saved
    prefix_path_ckpts = '/om/user/brando90/home_simulation_research/all_ckpts/%s/%s/' # folder where the results from script is saved
    arg.tb_data_dump = '/om/user/brando90/home_simulation_research/tb_dump' # folder where /train,/cv,/test tb stats are stored
if 'IN_DOCKER_CONT' in os.environ:
    ## to run in docker
    arg.data_dirpath = '/home_simulation_research/hbf_tensorflow_code/tf_experiments_scripts/data/' # path to datasets
    prefix_path_sim_results = '/home_simulation_research/simulation_results_scripts/%s/%s/' # folder where the results from script is saved
    prefix_path_ckpts = '/home_simulation_research/all_ckpts/%s/%s/' # folder where the results from script is saved
    arg.tb_data_dump = '/home_simulation_research/tb_dump' # folder where /train,/cv,/test tb stats are stored

# prefix_path_sim_results = '../../simulation_results_scripts/%s/%s'
# prefix_path_ckpts = '../../all_ckpts/%s/%s' # folder where the results from script is saved
arg.get_path_root =  lambda arg: prefix_path_sim_results%(arg.experiment_root_dir,arg.experiment_name)
arg.get_path_root_ckpts =  lambda arg: prefix_path_ckpts%(arg.experiment_root_dir,arg.experiment_name)

arg.prefix_ckpt = 'mdl_ckpt'
####
arg.data_filename = 'f_32D_binary_parity_N80000'
arg.task_folder_name = mtf.get_experiment_folder(arg.data_filename) #om_f_4d_conv
arg.type_preprocess_data = None
#
arg.N_frac = int(8*10**4)
#print('arg.N_frac: ', arg.N_frac)

## Classification Task related flags
arg.classification = mtf.classification_task_or_not(arg)
arg.classification = True
#arg.classification = False
arg.softmax, arg.one_hot = True, True
#arg.softmax, arg.one_hot = False, False

arg.experiment_name = 'task_May10_NN_32D_parity_prod_80K_softmax' # experiment_name e.g. task_Oct_10_NN_MGD_xavier_relu_N2000
#arg.experiment_name = 'TMP3'
arg.job_name = 'NN_32D_units31x2_Adam'

arg.experiment_root_dir = mtf.get_experiment_folder(arg.data_filename)
#
arg.mdl = 'standard_nn'
#arg.mdl = 'binary_tree_16D_conv_hidden_layer'
#
if arg.mdl == 'debug_mdl':
    arg.act = tf.nn.relu
    arg.dims = None
    arg.get_dims = lambda arg: arg.dims
    arg.get_x_shape = lambda arg: [None,arg.D]
    arg.get_y_shape = lambda arg: [None,arg.D_out]
elif arg.mdl == 'standard_nn':
    arg.init_type = 'truncated_normal'
    arg.init_type = 'data_xavier_kern'
    arg.init_type = 'xavier'

    K = 31*2
    arg.units = [K]
    #arg.mu = 0.0
    #arg.std = 0.5

    arg.get_W_mu_init = lambda arg: [None, None, 0]
    #arg.get_W_mu_init = lambda arg: [None, None, None, None, None, 0]
    #arg.get_W_std_init = lambda arg: [None, None, 0.1]
    arg.std_low, arg.std_high = 0.001, 3.0
    arg.get_W_std_init = lambda arg: [None, None, float(np.random.uniform(low=arg.std_low, high=arg.std_high, size=1)) ]
    #arg.get_W_std_init = lambda arg: [None, None, None, None, None, float(np.random.uniform(low=arg.std_low, high=arg.std_high, size=1)) ]

    #arg.get_W_mu_init = lambda arg: len(arg.dims)*[arg.mu]
    #arg.get_W_std_init = lambda arg: len(arg.dims)*[arg.std]

    arg.b = 0.1
    arg.get_b_init = lambda arg: len(arg.get_dims(arg))*[arg.b]

    arg.act = tf.nn.relu
    #arg.act = tf.nn.elu
    #arg.act = tf.nn.softplus
    #
    arg.get_x_shape = lambda arg: [None,arg.D]
    arg.get_dims = lambda arg: [arg.D]+arg.units+[arg.D_out]
else:
    raise ValueError('Need to use a valid model, incorrect or unknown model %s give.'%arg.mdl)

arg.get_y_shape = lambda arg: [None, arg.D_out]
# float type
arg.float_type = tf.float32
#steps
arg.steps_low = int(2.5*80000)
#arg.steps_low = int(1*4001)
arg.steps_high = arg.steps_low+1
arg.get_steps = lambda arg: int( np.random.randint(low=arg.steps_low ,high=arg.steps_high) )

arg.M_low = 32
arg.M_high = 15000
arg.get_batch_size = lambda arg: int(np.random.randint(low=arg.M_low , high=arg.M_high))
#arg.potential_batch_sizes = [16,32,64,128,256,512,1024]
#arg.potential_batch_sizes = [128]
def get_power2_batch_size(arg):
    i = np.random.randint( low=0, high=len(arg.potential_batch_sizes) )
    batch_size = arg.potential_batch_sizes[i]
    return batch_size
#arg.get_batch_size = get_power2_batch_size
## report freqs
arg.report_error_freq = 50
arg.get_save_ckpt_freq = lambda arg: int(0.25*arg.nb_steps)

## learning step/rate
arg.low_log_const_learning_rate, arg.high_log_const_learning_rate = -0.5, -6
arg.get_log_learning_rate =  lambda arg: np.random.uniform(low=arg.low_log_const_learning_rate, high=arg.high_log_const_learning_rate)
arg.get_start_learning_rate = lambda arg: 10**arg.log_learning_rate
#arg.get_log_learning_rate =  lambda arg: None
#arg.get_start_learning_rate = lambda arg: 0.01

## decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
arg.decay_rate_low, arg.decay_rate_high = 0.1, 1.0
arg.get_decay_rate = lambda arg: np.random.uniform(low=arg.decay_rate_low, high=arg.decay_rate_high)
#arg.get_decay_rate = lambda arg: 0.1
def get_decay_steps(arg):
    #arg.decay_steps_low, arg.decay_steps_high = arg.report_error_freq, arg.M
    arg.decay_steps_low, arg.decay_steps_high = 5000, 30000
    decay_steos = np.random.randint(low=arg.decay_steps_low, high=arg.decay_steps_high)
    return decay_steos
#get_decay_steps = lambda arg: 10000
arg.get_decay_steps = get_decay_steps # when stair case, how often to shrink

#arg.staircase = False
arg.staircase = True

# optimization_alg = 'GD'
# optimization_alg = 'Momentum'
# optimization_alg = 'Adadelta'
# optimization_alg = 'Adagrad'
optimization_alg = 'Adam'
# optimization_alg = 'RMSProp'
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
else:
    pass

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
if arg.type_job == 'slurm_array_parallel':
    arg.slurm_array_task_id = cmd_args.SLURM_ARRAY_TASK_ID if cmd_args.SLURM_ARRAY_TASK_ID else os.environ['SLURM_ARRAY_TASK_ID']
else:
    arg.slurm_array_task_id = arg.type_job

if cmd_args.save_config_args:
    # flag to save current config files to pickle file
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
arg.use_tensorboard = cmd_args.tensorboard
arg.get_dataset = lambda arg: (arg.X_train, arg.Y_train, arg.X_cv, arg.Y_cv, arg.X_test, arg.Y_test)

arg.act_name = arg.act.__name__
##
arg.cmd_args = cmd_args
##
arg.print_func = print_func_flush_true # functools.partial(print, flush=True)
arg.flush = True
#arg.flush = False
# makes things serial or not (ie. not spin serial processes or not)
#arg.debug = True
arg.debug = False
#
arg.display_training = True
#arg.display_training = False
#
#arg.collect_generalization = True
arg.collect_generalization = False
##
arg.start_stid = 1
arg.end_stid = arg.nb_array_jobs
arg.restore = False
#pdb.set_trace()
arg.rand_x = None
if __name__ == '__main__':
    cmd_args = arg.cmd_args
    #print('In __name__ == __main__')
    if cmd_args.type_job == 'serial':
        # jobs one job. No slurm array
        arg.start_stid = 1
        arg.end_stid = arg.nb_array_jobs
        large_main_hp.run_hyperparam_search2(arg)
    elif cmd_args.type_job == 'slurm_array_parallel':
        # run one single job according to slurm array command
        main_hp.main_hp(arg)
    elif cmd_args.type_job == 'main_large_hp_ckpt':
        large_main_hp.main_large_hp_ckpt(arg)
    #elif cmd_args.type_job = 'dgx1_multiprocess':
