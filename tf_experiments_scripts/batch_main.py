#!/usr/bin/env
#SBATCH --job-name=Python
#SBATCH --array=1-10
#SBATCH --mem=14000
#SBATCH --time=30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

#from __future__ import print_function

import os

import namespaces as ns
import numpy as np

import main_nn
import my_tf_pkg as mtf

##
print('In batch script', flush=True)
print(ns)
##
arg = ns.Namespace()

task_name = 'task_qianli_func'
task_name = 'task_f_2D_task2'
task_name = 'task_f_2d_task2_xsinglog1_x_depth2'
task_name = 'task_f_2d_task2_xsinglog1_x_depth3'
task_name = 'task_f2d_2x2_1_cosx1x2_depth2'
task_name = 'task_f2d_2x2_1_cosx1_plus_x2_depth2'
task_name = 'task_f_4d_conv'
task_name = 'task_f_8d_conv'
task_name = 'task_f_4d_conv_2nd'
#task_name = 'task_f_4d_non_conv'
#task_name = 'task_f_8d_non_conv'
#task_name = 'task_MNIST_flat'
#task_name = 'task_MNIST_flat_auto_encoder'
arg.task_name = task_name

arg.mdl = 'standard_nn'
#arg.mdl = 'hbf'
#arg.mdl = 'binary_tree_4D_conv'
if arg.mdl == 'standard_nn':
    arg.init_type = 'truncated_normal'
    arg.init_type = 'data_xavier_kern'
    arg.init_type = 'xavier'

    arg.units = [5]
    #arg.units = [6,6]
    #arg.units = [6,6,6]

    arg.mu = 0.0
    arg.std = 0.0

    arg.get_W_mu_init = lambda arg: len(arg.dims)*[arg.mu]
    arg.get_W_std_init = lambda arg: len(arg.dims)*[arg.std]

    arg.b = 0.1
    arg.get_b_init = lambda arg: len(arg.dims)*[arg.b]
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
elif arg.mdl == 'binary_tree_4D_conv':
    nb_filters = 6
    nb_filters = 12
    nb_filters = 18
    mu = 0.0
    std = 1.0

    arg.mu = mu
    arg.std = std
    arg.nb_filters = nb_filters
elif arg.mdl == 'binary_tree_8D_conv':
    # 8D
    #nb_filters=[9,18]
    #bn_tree_init_stats=[0.0,0.1,0.0,0.1,0.0,0.1]
    nb_filters = [9, 18]
    mu = 0.0
    std = 1.0

    arg.mu = mu
    arg.std = std
    arg.nb_filters = nb_filters
else:
    raise ValueError('Need to use a valid model, incorrect or unknown model %s give.'%arg.mdl)

#steps
arg.steps_low = 100
arg.steps_high = 101
arg.get_steps = lambda arg: int( np.random.randint(low=arg.steps_low ,high=arg.steps_high) )

arg.M_low = 51
arg.M_high = 52
arg.get_batch_size = lambda arg: int(np.random.randint(low=arg.M_low , high=arg.M_high))
arg.report_error_freq = 50

arg.low_log_const_learning_rate, arg.high_log_const_learning_rate = -0.01, -6
arg.get_log_learning_rate =  lambda arg: np.random.uniform(low=arg.low_log_const_learning_rate, high=arg.high_log_const_learning_rate)
arg.get_start_learning_rate = lambda arg: 10**arg.log_learning_rate
## decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
arg.decay_rate_low, arg.decay_rate_high = 0.3, 0.99
arg.get_decay_rate = lambda arg: np.random.uniform(low=arg.decay_rate_low, high=arg.decay_rate_high)

#arg.decay_steps_low, arg.decay_steps_high = arg.report_error_freq, arg.M
#arg.get_decay_steps_low_high = lambda arg: arg.report_error_freq, arg.M
#arg.get_decay_steps = lambda arg: np.random.randint(low=arg.decay_steps_low, high=arg.decay_steps_high)
def get_decay_steps(arg):
    arg.decay_steps_low, arg.decay_steps_high = arg.report_error_freq, arg.M
    decay_steos = np.random.randint(low=arg.decay_steps_low, high=arg.decay_steps_high)
    return decay_steos
arg.get_decay_steps = get_decay_steps

#arg.staircase = False
arg.staircase = True

optimization_alg = 'GD'
optimization_alg = 'Momentum'
# optimization_alg = 'Adadelta'
# optimization_alg = 'Adagrad'
# optimization_alg = 'Adam'
# optimization_alg = 'RMSProp'
arg.optimization_alg = optimization_alg

if optimization_alg == 'GD':
    pass
elif optimization_alg=='Momentum':
    arg.get_use_nesterov = lambda: False
    #arg.get_use_nesterov = lambda: True
    arg.momentum_low, arg.momontum_high = 0.1, 0.99
    arg.get_momentum = lambda arg: np.random.uniform(low=arg.momentum_low,high=arg.momontum_high)
elif optimization_alg == 'Adadelta':
    arg.rho_low, arg.rho_high = 0.1, 0.99
    arg.get_rho = lambda arg: np.random.uniform(low=arg.rho_low,high=arg.rho_high)
elif optimization_alg == 'Adagrad':
    #only has learning rate
    pass
elif optimization_alg == 'Adam':
    arg.get_beta1 = lambda: 0.99 # m = b1m + (1 - b1)m
    arg.get_beta2w = lambda: 0.999 # v = b2 v + (1 - b2)v
    #arg.beta1_low, arg.beta1_high = beta1_low=0.7, beta1_high=0.99 # m = b1m + (1 - b1)m
    #arg.beta2_low, arg.beta2_high = beta2_low=0.8, beta2_high=0.999 # v = b2 v + (1 - b2)v
elif optimization_alg == 'RMSProp':
    arg.decay_loc, arg.decay_high = 0.75, 0.99
    arg.momentum_low, arg.momontum_high = 0.0, 0.99
    arg.get_momentum = lambda arg: np.random.uniform(low=arg.momentum_low,high=arg.momontum_high)
else:
    pass


arg.bn = False
arg.trainable_bn = False #scale, shift BN

# NORMALIZE UNIT CIRCLE
arg.data_normalize='normalize_input'
arg.data_normalize='dont_normalize'

re_train = None
arg.re_train = re_train

#
arg.experiment_name = 'tmp_experiment'
arg.experiment_root_dir = mtf.get_experiment_folder(task_name)

#
slurm_jobid = 'TB'
slurm_array_task_id = 'TB'
job_name = 'TB'

arg.slurm_jobid = slurm_jobid
arg.slurm_array_task_id = slurm_array_task_id
arg.job_name = job_name

#
arg.mdl_save = False
arg.mdl_save = True

arg.max_to_keep = 1
#
#arg.use_tensorboard = False
arg.use_tensorboard = True

if __name__ == '__main__':
    print('In __name__ == __main__')
    #main_nn.main_old()
    main_nn.main(arg)
