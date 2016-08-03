#!/bin/bash
#SBATCH --job-name=HBF2_48_48
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --array=1-1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

#data_normalize=normalize_input
data_normalize=dont_normalize

#dont_train_S=train_S
trainable_S=dont_train_S

python main_nn.py $SLURM_JOBID $SLURM_ARRAY_TASK_ID om_2x2_1_cosx1_plus_x2_depth2 task_2_August_HBF2_depth_2_1000_dont_train_S HBF2_48_48_multiple_1000 True 48,48 multiple_S task_f2d_2x2_1_cosx1_plus_x2_depth2 False False hbf data_trunc_norm_kern $data_normalize $trainable_S
