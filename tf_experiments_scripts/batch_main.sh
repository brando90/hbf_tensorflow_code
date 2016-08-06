#!/bin/bash
#SBATCH --job-name=HBF1_12
#SBATCH --nodes=1
#SBATCH --mem=14000
#SBATCH --time=6-23
#SBATCH --array=1-1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com
#SBATCH --exclude=node047

#task_name=task_f2d_2x2_1_cosx1_plus_x2_depth2
#task_name=task_f2d_2x2_1_cosx1x2_depth2
#task_name=task_f_2d_task2_xsinglog1_x_depth2
task_name=task_MNIST_flat_auto_encoder
#main_folder=om_2x2_1_cosx1_plus_x2_depth2
main_folder=om_mnist

#data_normalize=normalize_input
data_normalize=dont_normalize

trainable_S=train_S
#trainable_S=dont_train_S

#mdl=standard_nn
mdl=hbf

#train_S_type=multiple_S
#train_S_type=single_S
#init_type=truncated_normal
#init_type=data_init
init_type=kern_init
#init_type=kpp_init
#init_type=data_trunc_norm_kern
#init_type=data_xavier_kern
#init_type=xavier

init=all_same_const-525.32626263
#init=first_constant_rest_uniform_random-[525.32626263,[0.9,2.5]]

units=12

optimization_alg=GD
optimization_alg=momentum
optimization_alg=Adadelta
optimization_alg=Adagrad
optimization_alg=Adam
optimization_alg=RMSProp

python main_nn.py $SLURM_JOBID $SLURM_ARRAY_TASK_ID $main_folder task_August_HBF1_depth_2_1000_dont_train_S run_HBF1_12_multiple_1000 True $units multiple_S $task_name False False $mdl $init_type $data_normalize $trainable_S $init $optimization_alg
