#!/usr/bin/env python
#SBATCH --qos=cbmm
#SBATCH --job-name=Python
#SBATCH --array=1-2
#SBATCH --mem=1000
#SBATCH --time=0-2:20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

#from __future__ import print_function
#tensorboard --logdir=/tmp/mdl_logs
#

import os
import sys

import pickle
import namespaces as ns
import argparse
import pdb

import numpy as np
import tensorflow as tf

import my_tf_pkg as mtf

##
print('In batch script', flush=True)
print(ns)
###
arg = ns.Namespace()

arg.data_dirpath = './data/'
##

#arg.data_file_name = 'h_gabor_data_and_mesh'
#arg.data_file_name = 'f_1D_cos_no_noise_data' #task_qianli_func
#arg.data_file_name = 'f_4D_conv_2nd'
#arg.data_file_name = 'f_4D_conv_2nd_noise_3_0_25std'
#arg.data_file_name = 'f_4D_conv_2nd_noise_6_0_5std'
#arg.data_file_name = 'f_4D_conv_2nd_noise_12_1std'
#arg.data_file_name = 'f_4D_cos_x2_BT'
#arg.data_file_name = 'f_4D_simple_ReLu_BT_2_units_1st'
arg.data_file_name = 'f_8D_conv_cos_poly1_poly1'
#arg.data_file_name = 'f_8D_single_relu'
#arg.data_file_name = 'f_4D_cos_x2_BT'
#arg.data_file_name = 'f_4D_simple_ReLu_BT'
#arg.data_file_name = 'MNIST_flat'
#arg.data_file_name = 'MNIST_flat_auto_encoder'
arg.task_folder_name = mtf.get_experiment_folder(arg.data_file_name) #om_f_4d_conv
#
arg.N_frac = 2000
print('arg.N_frac: ', arg.N_frac)
#
arg.experiment_name = 'tmp_task_Oct_19_BT4D_MGD_xavier_relu_N2000' # task_Oct_10_BT4D_MGD_xavier_relu_N2000 e.g. task_August_10_BT
arg.experiment_root_dir = mtf.get_experiment_folder(arg.data_file_name)
arg.job_name = 'BTHL_4D_6_12_MGD_200' # job name e.g BTHL_4D_6_12_MGD_200
#
arg.experiment_name = 'tmp_task_Oct_10_NN_MGD_xavier_relu_N2000' # experiment_name e.g. task_Oct_10_NN_MGD_xavier_relu_N2000
arg.experiment_root_dir = mtf.get_experiment_folder(arg.data_file_name)
arg.job_name = 'NN_4D_31_MGD_200' # job name e.g NN_4D_31_MGD_200
#
arg.mdl = 'standard_nn'
#arg.mdl = 'hbf'
#arg.mdl = 'binary_tree_4D_conv_hidden_layer'
#arg.mdl = "binary_tree_4D_conv_hidden_layer_automatic"
arg.mdl = 'binary_tree_8D_conv_hidden_layer'
if arg.mdl == 'standard_nn':
    arg.init_type = 'truncated_normal'
    arg.init_type = 'xavier'

    arg.units = [107]
    #arg.units = [110]
    #arg.units = [237]
    #arg.units = [412]
    #arg.units = [635]
    #arg.units = [906]
    #arg.units = [237]

    #arg.mu = 0.0
    #arg.std = 0.01

    arg.get_W_mu_init = lambda arg: [None, None, 0]
    arg.get_W_std_init = lambda arg: [None, None, 0.1]
    #arg.std_low, arg.std_high = 0.001, 0.1
    #arg.get_W_std_init = lambda arg: [None, None, np.random.uniform(low=arg.std_low, high=arg.std_high, size=1)]
    #arg.get_W_mu_init = lambda arg: len(arg.dims)*[arg.mu]
    #arg.get_W_std_init = lambda arg: len(arg.dims)*[arg.std]

    arg.b = 0.1
    arg.get_b_init = lambda arg: len(arg.dims)*[arg.b]

    arg.act = tf.nn.relu
    #arg.act = tf.nn.elu
    #arg.act = tf.nn.softplus
elif arg.mdl == 'hbf':
    #arg.init_type = 'truncated_normal'
    #arg.init_type = 'data_init'
    arg.init_type = 'kern_init'
    arg.init_type = 'kpp_init'

    arg.units = [31]
    # arg.units = [6,6]
    # arg.units = [6,6,6]

    arg.mu = 0.0
    arg.std = 0.0

    arg.W_mu_init = lambda arg: len(arg.dims)*[arg.mu]
    arg.W_std_init = lambda arg: len(arg.dims)*[arg.std]

    # train shape of Gaussians
    arg.trainable_S = 'train_S'
    #arg.trainable_S = 'dont_train_S'
    arg.train_S_type = 'multiple_S'
    #arg.train_S_type = 'single_S'

    arg.S = [None, 0.4]
    arg.get_b_init = lambda arg: arg.S

    arg.act = tf.nn.exp
elif arg.mdl == "binary_tree_4D_conv_hidden_layer_automatic":
    arg.L, arg.padding, arg.scope_name, arg.verbose = 2, 'VALID', 'BT_4D', False
    #
    arg.init_type = 'xavier'
    arg.weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
    arg.biases_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)
    #
    F1 = 6
    arg.F = [None, F1, 2*F1]
    #
    #arg.normalizer_fn = None
    arg.trainable = True
    arg.normalizer_fn = tf.contrib.layers.batch_norm

    arg.act = tf.nn.relu
    #arg.act = tf.nn.elu
    #arg.act = tf.nn.softplus
elif arg.mdl == 'binary_tree_8D_conv_hidden_layer':
    arg.L, arg.padding, arg.scope_name, arg.verbose = 3, 'VALID', 'BT_8D', False
    #
    arg.init_type = 'xavier'
    arg.weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
    arg.biases_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)
    #
    F1 = 7
    arg.F = [None, F1, 2*F1, 4*F1]
    #
    arg.normalizer_fn = None
    arg.trainable = True
    #arg.normalizer_fn = tf.contrib.layers.batch_norm

    arg.act = tf.nn.relu
    #arg.act = tf.nn.elu
    #arg.act = tf.nn.softplus
else:
    raise ValueError('Need to use a valid model, incorrect or unknown model %s give.'%arg.mdl)
#steps
#arg.steps_low = 100
#arg.steps_high = 101
#arg.get_steps = lambda arg: int( np.random.randint(low=arg.steps_low ,high=arg.steps_high) )
arg.steps = int(1*arg.N_frac)
arg.get_steps = lambda arg: int( arg.steps )

#arg.M_low = 51
#arg.M_high = 52
#arg.get_batch_size = lambda arg: int(np.random.randint(low=arg.M_low , high=arg.M_high))
arg.M = 2
arg.get_batch_size = lambda arg: arg.M #M
arg.report_error_freq = 50

#arg.low_log_const_learning_rate, arg.high_log_const_learning_rate = -0.01, -6
#arg.get_log_learning_rate =  lambda arg: np.random.uniform(low=arg.low_log_const_learning_rate, high=arg.high_log_const_learning_rate)
#arg.get_start_learning_rate = lambda arg: 10**arg.log_learning_rate
arg.get_log_learning_rate = lambda arg: None
arg.starter_learning_rate = 0.01
arg.get_start_learning_rate = lambda arg: arg.starter_learning_rate
## decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
#arg.decay_rate_low, arg.decay_rate_high = 0.3, 0.99
#arg.get_decay_rate = lambda arg: np.random.uniform(low=arg.decay_rate_low, high=arg.decay_rate_high)
arg.decay_rate  = 0.95
arg.get_decay_rate = lambda arg: arg.decay_rate

#arg.decay_steps_low, arg.decay_steps_high = arg.report_error_freq, arg.M
#arg.get_decay_steps_low_high = lambda arg: arg.report_error_freq, arg.M
#arg.get_decay_steps = lambda arg: np.random.randint(low=arg.decay_steps_low, high=arg.decay_steps_high)
# def get_decay_steps(arg):
#     arg.decay_steps_low, arg.decay_steps_high = arg.report_error_freq, arg.M
#     decay_steos = np.random.randint(low=arg.decay_steps_low, high=arg.decay_steps_high)
#     return decay_steos
arg.decay_steps = 2000
arg.get_decay_steps = lambda arg: arg.decay_steps # when stair case, how often to shrink

# If the argument staircase is True, then global_step / decay_steps is an integer division and the decayed earning rate follows a staircase function.
#arg.staircase = False
arg.staircase = True

optimization_alg = 'GD'
#optimization_alg = 'Momentum'
#optimization_alg = 'Adadelta'
#optimization_alg = 'Adagrad'
#arg.optimization_alg = 'Adam'
#optimization_alg = 'RMSProp'
arg.optimization_alg = optimization_alg

if optimization_alg == 'GD':
    pass
elif optimization_alg=='Momentum':
    #arg.get_use_nesterov = lambda: False
    arg.get_use_nesterov = lambda: True
    #arg.momentum_low, arg.momontum_high = 0.1, 0.99
    #arg.get_momentum = lambda arg: np.random.uniform(low=arg.momentum_low,high=arg.momontum_high)
    arg.momentum = 0.9
    arg.get_momentum = lambda arg: arg.momentum
elif optimization_alg == 'Adadelta':
    #arg.rho_low, arg.rho_high = 0.1, 0.99
    #arg.get_rho = lambda arg: np.random.uniform(low=arg.rho_low,high=arg.rho_high)
    arg.rho = 0.8
    arg.get_rho = lambda arg: arg.rho
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
    #arg.decay_low, arg.decay_high = 0.75, 0.99
    #arg.get_decay = lambda arg: np.random.uniform(low=arg.decay_low,high=arg.decay_high)
    arg.decay = 0.9
    arg.get_decay = lambda arg: arg.decay
    #arg.momentum_low, arg.momontum_high = 0.0, 0.99
    #arg.get_momentum = lambda arg: np.random.uniform(low=arg.momentum_low,high=arg.momontum_high)
    arg.momentum = 0.1
    arg.get_momentum = lambda arg: arg.momentum
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
arg.save_config_args = False
#arg.debug = False
arg.slurm_jobid = 'TB_slurm_jobid'
arg.slurm_array_task_id = 'TB_slurm_array_task_id'
#
arg.path_root = '%s/%s'%(arg.experiment_root_dir,arg.experiment_name)
arg.get_path_root =  lambda arg: arg.path_root
#
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", help="debug mode: loads the old (pickle) config file to run in debug mode", action='store_true')
parser.add_argument("-sj", "--SLURM_JOBID", help="SLURM_JOBID for run")
parser.add_argument("-stid", "--SLURM_ARRAY_TASK_ID", help="SLURM_ARRAY_TASK_ID for run")
parser.add_argument("-sca", "--save_config_args", help="save_config_args saves the current config file to a picke file ")

parser.add_argument("-tb", "--tensorboard", dest='tensorboard', help="tensorboard mode: save tensorboard data", action='store_true') #if the flag is present then DO use tensorboard else don't
parser.add_argument('-notb', "--no-tensorboard", dest='tensorboard', help="tensorboard mode: save tensorboard data", action='store_false')
parser.set_defaults(tensorboard=False)

parser.add_argument("-pr", "--path_root", help="path_root: specifies the path root to save hyper params")
#parser.add_argument("-hp", "--hyper_parameter", help="hyper_parameter: when searching for hp on needs to specify were to store the results of experiment or it will default.")

cmd_args = parser.parse_args()
# if sj, stid arguments given set them else leave them default values
arg.slurm_jobid = cmd_args.SLURM_JOBID if cmd_args.SLURM_JOBID else arg.slurm_jobid
arg.slurm_array_task_id = cmd_args.SLURM_ARRAY_TASK_ID if cmd_args.SLURM_ARRAY_TASK_ID else arg.slurm_array_task_id
if cmd_args.save_config_args:
    # flag to save current config files to pickle file
    print(cmd_args.save_config_args)
    arg.save_config_args = cmd_args.save_config_args
if cmd_args.debug:
    #arg.debug = cmd_args.debug
    # load old pickle config
    pickled_arg_dict = pickle.load( open( "pickle-slurm-%s_%s.p"%(int(arg.slurm_jobid)+int(arg.slurm_array_task_id),arg.slurm_array_task_id), "rb" ) )
    print( pickled_arg_dict )
    # values merged with the second dict's values overwriting those from the first.
    arg_dict = {**dict(arg), **pickled_arg_dict}
    arg = ns.Namespace(arg_dict)
if cmd_args.path_root:
    arg.path_root = cmd_args.path_root
    arg.get_path_root =  lambda arg: arg.path_root

#
arg.mdl_save = False
#arg.mdl_save = True

#
#if the flag is present then DO use tensorboard else don't
arg.use_tensorboard = cmd_args.tensorboard
print('---> arg.use_tensorboard: ', arg.use_tensorboard)
print('---> cmd_args.tensorboard: ', cmd_args.tensorboard)

arg.max_to_keep = 1

arg.get_errors_from = mtf.get_errors_based_on_train_error
#arg.get_errors_from = mtf.get_errors_based_on_validation_error

# scp -r brando90@polestar.mit.edu:/cbcl/cbcl01/brando90/Documents/MEng/hbf_tensorflow_code/tf_experiments_scripts/hp1  /Users/brandomiranda/Documents/MIT/MEng/hbf_tensorflow_code/tf_experiments_scripts

# scp -r brando90@polestar.mit.edu:/cbcl/cbcl01/brando90/Documents/MEng/hbf_tensorflow_code/tf_experiments_scripts/hp2  /Users/brandomiranda/Documents/MIT/MEng/hbf_tensorflow_code/tf_experiments_scripts

# scp -r brando90@polestar.mit.edu:/cbcl/cbcl01/brando90/Documents/MEng/hbf_tensorflow_code/tf_experiments_scripts/hp3  /Users/brandomiranda/Documents/MIT/MEng/hbf_tensorflow_code/tf_experiments_scripts

# scp -r brando90@polestar.mit.edu:/cbcl/cbcl01/brando90/Documents/MEng/hbf_tensorflow_code/tf_experiments_scripts/hp4  /Users/brandomiranda/Documents/MIT/MEng/hbf_tensorflow_code/tf_experiments_scripts

if __name__ == '__main__':
    print('In __name__ == __main__')
    #main_nn.main_old()
    mtf.main_nn(arg)
