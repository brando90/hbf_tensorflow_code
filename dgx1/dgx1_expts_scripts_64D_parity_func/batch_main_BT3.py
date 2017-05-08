#!/usr/bin/env python
#SBATCH --mem=4500
#SBATCH --time=3-18:20
#SBATCH --array=1-200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com
#SBATCH --gres=gpu:1

#from __future__ import #print_function
#tensorboard --logdir=/tmp/mdl_logs
#

print('#!/usr/bin/env python')
print('#!/usr/bin/python')

import os
import sys

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

##
#print('In batch script', flush=True)
#print = functools.partial(print, flush=True)
#print(ns)
###
arg = maps.NamedDict()

#
arg.get_errors_from = mtf.get_errors_based_on_train_error
#arg.get_errors_from = mtf.get_errors_based_on_validation_error
#

#arg.type_job, arg.nb_array_jobs = 'serial', 1 #careful when this is on and GPU is NOT on
#arg.type_job = 'slurm_array_parallel'
arg.type_job, arg.nb_array_jobs = 'main_large_hp_ckpt', 200
arg.save_checkpoints = True
#arg.save_checkpoints = False
#arg.save_last_mdl = True
arg.save_last_mdl = False

## debug mode
#arg.data_dirpath = './data/' # path to datasets
#prefix_path_sim_results = './tmp_simulation_results_scripts/%s/%s/' # folder where the results from script is saved
#prefix_path_ckpts = './tmp_all_ckpts/%s/%s/' # folder where the results from script is saved
## to run locally: python batch_main.py -sj sj
#arg.data_dirpath = './data/' # path to datasets
#prefix_path_sim_results = '../../simulation_results_scripts/%s/%s/' # folder where the results from script is saved
#prefix_path_ckpts = '../../all_ckpts/%s/%s/' # folder where the results from script is saved
## to run in docker
arg.data_dirpath = '/home_simulation_research/hbf_tensorflow_code/tf_experiments_scripts/data/' # path to datasets
prefix_path_sim_results = '/home_simulation_research/simulation_results_scripts/%s/%s/' # folder where the results from script is saved
prefix_path_ckpts = '/home_simulation_research/all_ckpts/%s/%s/' # folder where the results from script is saved

# prefix_path_sim_results = '../../simulation_results_scripts/%s/%s'
# prefix_path_ckpts = '../../all_ckpts/%s/%s' # folder where the results from script is saved
arg.get_path_root =  lambda arg: prefix_path_sim_results%(arg.experiment_root_dir,arg.experiment_name)
arg.get_path_root_ckpts =  lambda arg: prefix_path_ckpts%(arg.experiment_root_dir,arg.experiment_name)

arg.prefix_ckpt = 'mdl_ckpt'
####
#arg.data_filename = 'h_gabor_data_and_mesh'
#arg.data_filename = 'f_1D_cos_no_noise_data' #task_qianli_func
#arg.data_filename = 'f_4D_conv_2nd'
#arg.data_filename = 'f_4D_conv_2nd_noise_3_0_25std'
#arg.data_filename = 'f_4D_conv_2nd_noise_6_0_5std'
#arg.data_filename = 'f_4D_conv_2nd_noise_12_1std'
#arg.data_filename = 'f_4D_cos_x2_BT'
#arg.data_filename = 'f_4D_simple_ReLu_BT_2_units_1st'
#arg.data_filename = 'f_8D_conv_cos_poly1_poly1'
#arg.data_filename = 'f_8D_single_relu'
#arg.data_filename = 'f_8D_conv_quad_cubic_sqrt'
#arg.data_filename = 'f_8D_conv_quad_cubic_sqrt'
#arg.data_filename = 'f_8D_product_continuous'
arg.data_filename = 'f_64D_product_binary'
#arg.data_filename = 'f_16D_ppt'
#arg.data_filename = 'f_32D_ppt'
#arg.data_filename = 'f_64D_ppt'
#arg.data_filename = 'f_256D_L8_ppt_1'
#arg.data_filename = 'f_8D_conv_quad_cubic_sqrt_shuffled'
#arg.data_filename = 'f_4D_simple_ReLu_BT'
#arg.data_filename = 'MNIST'
arg.task_folder_name = mtf.get_experiment_folder(arg.data_filename) #om_f_4d_conv
arg.type_preprocess_data = None
#
arg.N_frac = 60000
#print('arg.N_frac: ', arg.N_frac)

arg.classificaton = mtf.classification_task_or_not(arg)

#arg.experiment_name = 'task_Nov_22_BTSG1_2_3_8D_Adam_xavier_relu_N60000' # task_Oct_10_BT4D_MGD_xavier_relu_N2000 e.g. task_August_10_BT
#arg.experiment_name = 'task_Nov_22_BTSG2_3_2_8D_Adam_xavier_relu_N60000'
#arg.experiment_name = 'task_Nov_22_BTSG3_3_3_8D_Adam_xavier_relu_N60000'
#arg.experiment_name = 'tmp_task_Nov_22_BTSG4_4_2_8D_Adam_xavier_relu_N60000'
#arg.experiment_name = 'task_Jan_19_BT_256D_Adam_xavier_relu_N60000'
#arg.experiment_name = 'task_Feb_28_BT_32D_Adam_xavier_relu_N60000_100'
#arg.experiment_name = 'task_Feb_28_NN_32D_Adam_xavier_relu_N60000_100'
#arg.experiment_name = 'task_Mar_2_BT_8D_Adam_xavier_relu_N60000_original_setup'
#arg.experiment_name = 'task_Mar_2_NN_8D_Adam_xavier_relu_N60000_original_setup'
#arg.experiment_name = 'task_Mar_12_BT_8D_Adam_xavier_relu_N60000_original_setup_dgx1'
#arg.experiment_name = 'task_Mar_12_NN_8D_Adam_xavier_relu_N60000_original_setup_dgx1'
arg.experiment_name = 'task_Apr_5_BT_64D_Adam_xavier_relu_N60000_original_setup_dgx1'
#arg.experiment_name = 'task_Apr_5_NN_64D_Adam_xavier_relu_N60000_original_setup_dgx1'
#arg.experiment_name = 'TMP3'
#arg.experiment_name = 'TMP_hp_test'
#arg.experiment_name = 'dgx1_Feb_8_256D_Adam_xavier_relu_N60000'
#arg.job_name = 'BTSG1_8D_a19_Adam_200' # job name e.g BTHL_4D_6_12_MGD_200
#arg.job_name = 'BT_debug1'
arg.job_name = 'BT_64D_units3_Adam'

#arg.experiment_name = 'task_Nov_19_NN_Adam_xavier_relu_N60000' # experiment_name e.g. task_Oct_10_NN_MGD_xavier_relu_N2000
#arg.experiment_name = 'TMP_task_Jan_19_NN_256D_Adam_xavier_relu_N60000'
#arg.job_name = 'NN_8D_31_Adam_200' # job name e.g NN_4D_31_MGD_200
#arg.job_name = 'NN_debug2'
#arg.job_name = 'NN_64D_units31x2_Adam'
#
arg.experiment_root_dir = mtf.get_experiment_folder(arg.data_filename)
#
#arg.mdl = 'standard_nn'
#arg.mdl = 'hbf'
#arg.mdl = 'binary_tree_4D_conv_hidden_layer'
#arg.mdl = "binary_tree_4D_conv_hidden_layer_automatic"
#arg.mdl = 'binary_tree_8D_conv_hidden_layer'
#arg.mdl = 'binary_tree_16D_conv_hidden_layer'
#arg.mdl = 'binary_tree_32D_conv_hidden_layer'
arg.mdl = 'binary_tree_64D_conv_hidden_layer'
#arg.mdl = 'binary_tree_256D_conv_hidden_layer'
#arg.mdl = 'bt_subgraph'
#arg.mdl = 'debug_mdl'
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

    K = 3
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
elif arg.mdl == 'hbf':
    pass
    # arg.init_type = 'truncated_normal'
    # arg.init_type = 'data_init'
    # arg.init_type = 'kern_init'
    # arg.init_type = 'kpp_init'
    #
    # arg.units = [5]
    # arg.units = [6,6]
    # arg.units = [6,6,6]
    #
    # arg.mu = 0.0
    # arg.std = 0.0
    #
    # arg.W_mu_init = lambda arg: len(arg.dims)*[arg.mu]
    # arg.W_std_init = lambda arg: len(arg.dims)*[arg.std]
    #
    # # train shape of Gaussians
    # #arg.trainable_S = 'train_S'
    # arg.trainable_S = 'dont_train_S'
    # #arg.train_S_type = 'multiple_S'
    # arg.train_S_type = 'single_S'
    #
    # arg.b_init = lambda: [525.32626263]
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
    arg.normalizer_fn = None
    arg.trainable = True
    #arg.normalizer_fn = tf.contrib.layers.batch_norm

    arg.act = tf.nn.relu
    #arg.act = tf.nn.elu
    #arg.act = tf.nn.softplus
    #
    arg.get_x_shape = lambda arg: [None,1,arg.D,1]
    arg.type_preprocess_data = 're_shape_X_to_(N,1,D,1)'
    #
    arg.get_dims = lambda arg: [arg.D]+arg.nb_filters[1:]+[arg.D_out]
elif arg.mdl == 'binary_tree_8D_conv_hidden_layer':
    arg.L, arg.padding, arg.scope_name, arg.verbose = 3, 'VALID', 'BT_8D', False
    #
    arg.init_type = 'xavier'
    arg.weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
    arg.biases_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)
    #
    F1 = 1
    arg.F = [None, F1, 2*F1, 4*F1]
    #
    arg.normalizer_fn = None
    arg.trainable = True
    #arg.normalizer_fn = tf.contrib.layers.batch_norm

    arg.act = tf.nn.relu
    #arg.act = tf.nn.elu
    #arg.act = tf.nn.softplus
    #
    arg.get_x_shape = lambda arg: [None,1,arg.D,1]
    arg.type_preprocess_data = 're_shape_X_to_(N,1,D,1)'
    #
    arg.get_dims = lambda arg: [arg.D]+arg.nb_filters[1:]+[arg.D_out]
elif arg.mdl == 'binary_tree_16D_conv_hidden_layer':
    logD = 4
    L = logD
    arg.L, arg.padding, arg.scope_name, arg.verbose = L, 'VALID', 'BT_8D', False
    #
    arg.init_type = 'xavier'
    arg.weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
    arg.biases_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)
    #
    F1 = 6
    arg.F = [None] + [ F1*(2**l) for l in range(1,L+1) ]
    arg.nb_filters = arg.F
    #
    arg.normalizer_fn = None
    arg.trainable = True
    #arg.normalizer_fn = tf.contrib.layers.batch_norm

    arg.act = tf.nn.relu
    #arg.act = tf.nn.elu
    #arg.act = tf.nn.softplus
    #
    arg.get_x_shape = lambda arg: [None,1,arg.D,1]
    arg.type_preprocess_data = 're_shape_X_to_(N,1,D,1)'
    #
    arg.get_dims = lambda arg: [arg.D]+arg.nb_filters[1:]+[arg.D_out]
elif arg.mdl == 'binary_tree_32D_conv_hidden_layer':
    logD = 5
    L = logD
    arg.L, arg.padding, arg.scope_name, arg.verbose = L, 'VALID', 'BT_8D', False
    #
    arg.init_type = 'xavier'
    arg.weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
    arg.biases_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)
    #
    F1 = 1
    arg.F = [None] + [ F1*(2**l) for l in range(1,L+1) ]
    arg.nb_filters = arg.F
    #
    arg.normalizer_fn = None
    arg.trainable = True
    #arg.normalizer_fn = tf.contrib.layers.batch_norm

    arg.act = tf.nn.relu
    #arg.act = tf.nn.elu
    #arg.act = tf.nn.softplus
    #
    arg.get_x_shape = lambda arg: [None,1,arg.D,1]
    arg.type_preprocess_data = 're_shape_X_to_(N,1,D,1)'
    #
    arg.get_dims = lambda arg: [arg.D]+arg.nb_filters[1:]+[arg.D_out]
elif arg.mdl == 'binary_tree_64D_conv_hidden_layer':
    logD = 6
    L = logD
    arg.L, arg.padding, arg.scope_name, arg.verbose = L, 'VALID', 'BT_8D', False
    #
    arg.init_type = 'xavier'
    arg.weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
    arg.biases_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)
    #
    F1 = 3
    arg.F = [None] + [ F1*(2**l) for l in range(1,L+1) ]
    arg.nb_filters = arg.F
    #
    arg.normalizer_fn = None
    arg.trainable = True
    #arg.normalizer_fn = tf.contrib.layers.batch_norm

    arg.act = tf.nn.relu
    #arg.act = tf.nn.elu
    #arg.act = tf.nn.softplus
    #
    arg.get_x_shape = lambda arg: [None,1,arg.D,1]
    arg.type_preprocess_data = 're_shape_X_to_(N,1,D,1)'
    #
    arg.get_dims = lambda arg: [arg.D]+arg.nb_filters[1:]+[arg.D_out]
elif arg.mdl == 'binary_tree_256D_conv_hidden_layer':
    logD = 8
    L = logD
    arg.L, arg.padding, arg.scope_name, arg.verbose = L, 'VALID', 'BT_8D', False
    #
    arg.init_type = 'xavier'
    arg.weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
    arg.biases_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)
    #
    F1 = 6
    arg.F = [None] + [ F1*(2**l) for l in range(1,L+1) ]
    arg.nb_filters = arg.F
    #
    arg.normalizer_fn = None
    arg.trainable = True
    #arg.normalizer_fn = tf.contrib.layers.batch_norm

    arg.act = tf.nn.relu
    #arg.act = tf.nn.elu
    #arg.act = tf.nn.softplus
    #
    arg.get_x_shape = lambda arg: [None,1,arg.D,1]
    arg.type_preprocess_data = 're_shape_X_to_(N,1,D,1)'
    #
    arg.get_dims = lambda arg: [arg.D]+arg.nb_filters[1:]+[arg.D_out]
elif arg.mdl == 'bt_subgraph':
    arg.L, arg.padding, arg.scope_name, arg.verbose = 3, 'VALID', 'BT_subgraph', False
    #
    arg.init_type = 'xavier'
    arg.weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
    arg.biases_initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)
    #
    a = 5
    # nb of filters per unit
    #F1, F2, F3 = a, 2*a, 4*a #BT
    #F1, F2, F3 = a, 2*a, 4*a
    #F1, F2, F3 = a, 2*a, 6*a
    #F1, F2, F3 = 2*a, 3*a, 12*a
    F1, F2, F3 = 4*a, 5*a, 20*a
    nb_filters=[None,F1,F2,F3]
    u1, u2, u3 = F1, F2, F3
    # width of filters
    #w1, w2, w3 = 2,2*u1,2*u2 #BT
    #w1, w2, w3 = 2,3*u1,2*u2
    #w1, w2, w3 = 3,2*u1,3*u2
    #w1, w2, w3 = 3,3*u1,4*u2
    w1, w2, w3 = 4,2*u1,4*u2
    list_filter_widths=[None,w1,w2,w3]
    # stride
    #s1, s2, s3 = 2, 2*F1, 1 #BT
    #s1, s2, s3 = 2, 1*F1, 1
    #s1, s2, s3 = 1, 2*F1, 1
    #s1, s2, s3 = 1, 1*F1, 1
    s1, s2, s3 = 1, 1*F1, 1
    list_strides=[None,s1,s2,s3]
    #
    arg.nb_filters = nb_filters
    arg.list_filter_widths = list_filter_widths
    arg.list_strides = list_strides
    #
    arg.normalizer_fn = None
    arg.trainable = True
    #arg.normalizer_fn = tf.contrib.layers.batch_norm

    arg.act = tf.nn.relu
    #arg.act = tf.nn.elu
    #arg.act = tf.nn.softplus
    #
    arg.get_x_shape = lambda arg: [None,1,arg.D,1]
    arg.type_preprocess_data = 're_shape_X_to_(N,1,D,1)'
    #
    arg.get_dims = lambda arg: [arg.D]+arg.nb_filters[1:]+[arg.D_out]
else:
    raise ValueError('Need to use a valid model, incorrect or unknown model %s give.'%arg.mdl)

arg.get_y_shape = lambda arg: [None, arg.D_out]
# float type
arg.float_type = tf.float32
#steps
arg.steps_low = int(2.5*60000)
#arg.steps_low = int(1*801)
arg.steps_high = arg.steps_low+1
arg.get_steps = lambda arg: int( np.random.randint(low=arg.steps_low ,high=arg.steps_high) )

arg.M_low = 32
arg.M_high = 15000
arg.get_batch_size = lambda arg: int(np.random.randint(low=arg.M_low , high=arg.M_high))
#arg.potential_batch_sizes = [16,32,64,128,256,512,1024]
#arg.potential_batch_sizes = [4]
def get_power2_batch_size(arg):
    i = np.random.randint( low=0, high=len(arg.potential_batch_sizes) )
    batch_size = arg.potential_batch_sizes[i]
    return batch_size
#arg.get_batch_size = get_power2_batch_size
## report freqs
arg.report_error_freq = 100
arg.get_save_ckpt_freq = lambda arg: int(0.25*arg.nb_steps)

arg.low_log_const_learning_rate, arg.high_log_const_learning_rate = -0.5, -4
arg.get_log_learning_rate =  lambda arg: np.random.uniform(low=arg.low_log_const_learning_rate, high=arg.high_log_const_learning_rate)
arg.get_start_learning_rate = lambda arg: 10**arg.log_learning_rate
## decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
arg.decay_rate_low, arg.decay_rate_high = 0.1, 1.0
arg.get_decay_rate = lambda arg: np.random.uniform(low=arg.decay_rate_low, high=arg.decay_rate_high)

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
# optimization_alg = 'Adadelta'
# optimization_alg = 'Adagrad'
optimization_alg = 'Adam'
#optimization_alg = 'RMSProp'
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
