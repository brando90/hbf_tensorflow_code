#!/bin/bash
#SBATCH --job-name=NN2_6_6
#SBATCH --nodes=1
#SBATCH --mem=14000
#SBATCH --time=30:00
#SBATCH --array=1-8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

#SLURM_JOBID=SLURM_JOBID
#SLURM_ARRAY_TASK_ID=SLURM_ARRAY_TASK_ID

#task_name=task_f2d_2x2_1_cosx1_plus_x2_depth2
#task_name=task_f2d_2x2_1_cosx1x2_depth2
#task_name=task_f_2d_task2_xsinglog1_x_depth2
#task_name=task_MNIST_flat_auto_encoder
#task_name=task_f_4d
task_name=task_f_4d_conv
main_folder=om_f_4d_conv
#main_folder=om_f4d
#main_folder=om_2x2_1_cosx1_plus_x2_depth2
#main_folder=om_xsinlog1_x_depth2
#main_folder=om_mnist

#data_normalize=normalize_input
data_normalize=dont_normalize

trainable_S=train_S
#trainable_S=dont_train_S

#mdl=standard_nn
#mdl=hbf
mdl=binary_tree_4D_conv

#train_S_type=multiple_S
#train_S_type=single_S
#init_type=truncated_normal
#init_type=data_init
#init_type=kern_init
#init_type=kpp_init
#init_type=data_trunc_norm_kern
init_type=data_trunc_norm_trunc_norm
#init_type=data_xavier_kern
#init_type=xavier

#init=all_same_const-525.32626263
init=all_same_const-0.1
#init=first_constant_rest_uniform_random-[525.32626263,[0.9,2.5]]
#init=first_constant_rest_specific_consts-[1250.32,3]
#init=first_rand_same_uniform_rest_uniform_random-[[1,1250.32],[2,4]]

units=6,6
#nb_filters=18
# 4D
nb_filters=6
bn_tree_init_stats=[0.0,0.1]
# 8D
#nb_filters=[9,18]
#bn_tree_init_stats=[0.0,0.1,0.0,0.1,0.0,0.1]

#optimization_alg=GD
#optimization_alg=Momentum
#optimization_alg=Adadelta
#optimization_alg=Adagrad
#optimization_alg=Adam
optimization_alg=RMSProp

python main_nn.py $SLURM_JOBID $SLURM_ARRAY_TASK_ID $main_folder task_August_10_B BT_6_6_5_RMSProp_Test True $units multiple_S $task_name True True $mdl $init_type $data_normalize $trainable_S $init $optimization_alg $nb_filters $bn_tree_init_stats
