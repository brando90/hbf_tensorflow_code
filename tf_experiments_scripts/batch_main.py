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

import main_nn

print('In batch script', flush=True)
print(ns)

arg = ns.Namespace()

task_name = 'task_qianli_func'
task_name = 'task_f_2D_task2'
task_name = 'task_f_2d_task2_xsinglog1_x_depth2'
task_name = 'task_f_2d_task2_xsinglog1_x_depth3'
task_name = 'task_f2d_2x2_1_cosx1x2_depth2'
task_name = 'task_f2d_2x2_1_cosx1_plus_x2_depth2'
task_name = 'task_f_4d_conv'
task_name = 'task_f_8d_conv'
task_name = 'task_f_4d_task_conv_2nd'
task_name = 'task_f_4d_non_conv'
task_name = 'task_f_8d_non_conv'
task_name = 'task_MNIST_flat'
task_name = 'task_MNIST_flat_auto_encoder'
arg.task_name = task_name

mdl = 'standard_nn'
mdl = 'hbf'
mdl = 'binary_tree_4D_conv'
arg.mdl = mdl

# train shape of Gaussians
if arg.mdl == 'hbf':
    trainable_S = 'train_S'
    trainable_S = 'dont_train_S'

    train_S_type = 'multiple_S'
    train_S_type = 'single_S'

    arg.trainable_S = trainable_S
    arg.train_S_type = train_S_type

# BIAS or Guassian Shape
init_S = [525.32626263]
init_b = [0.1]

# Filter/W inits
init_type = 'truncated_normal'
init_type = 'data_init'
init_type = 'kern_init'
init_type = 'kpp_init'
init_type = 'data_trunc_norm_kern'
init_type = 'data_trunc_norm_trunc_norm'
init_type = 'data_xavier_kern'
init_type = 'xavier'
arg.init_type = init_type

# UNITS
if arg.mdl == 'standard_nn':
    units=[5]
    units=[6,6]
    units=[6,6,6]
    arg.units = units
elif arg.mdl == 'hbf':
    units=[5]
    units=[6,6]
    units=[6,6,6]
    arg.units = units
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

# OPTIMIZER
optimization_alg = 'GD'
optimization_alg = 'Momentum'
optimization_alg = 'Adadelta'
optimization_alg = 'Adagrad'
optimization_alg = 'Adam'
optimization_alg = 'RMSProp'
arg.optimization_alg = optimization_alg

# NORMALIZE UNIT CIRCLE
#data_normalize='normalize_input'
data_normalize='dont_normalize'
arg.data_normalize = data_normalize

if __name__ == '__main__':
    main_nn.main_old()
    #main_nn.main(arg)
