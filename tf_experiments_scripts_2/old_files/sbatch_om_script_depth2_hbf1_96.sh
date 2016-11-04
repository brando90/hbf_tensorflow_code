#!/bin/bash
#SBATCH --job-name=HBF1_96
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --array=1-1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com
#SBATCH --exclude=node047

#task_name=task_f2d_2x2_1_cosx1_plus_x2_depth2
#task_name=task_f2d_2x2_1_cosx1x2_depth2
task_name=task_f_2d_task2_xsinglog1_x_depth2
#folder=om_2x2_1_cosx1_plus_x2_depth2
folder=om_xsinlog1_x_depth2_hbf

#data_normalize=normalize_input
data_normalize=dont_normalize

#dont_train_S=train_S
trainable_S=dont_train_S

python main_nn.py $SLURM_JOBID $SLURM_ARRAY_TASK_ID $folder task_1_August_HBF1_depth_2_1000_dont_train_S HBF1_96_multiple_1000 True 96 multiple_S $task_name False False hbf kern_init $data_normalize $trainable_S all_same_const-0.61545455
