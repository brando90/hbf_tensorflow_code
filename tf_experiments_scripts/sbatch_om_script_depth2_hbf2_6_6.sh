#!/bin/bash
#SBATCH --job-name=HBF2_6_6
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

python main_nn.py $SLURM_JOBID $SLURM_ARRAY_TASK_ID om_xsinlog1_x_depth2_hbf task_2_August_HBF2_depth_2_1000_dont_train_S HBF2_6_6_multiple_1000 True 6,6 multiple_S task_f_2d_task2_xsinglog1_x_depth2 False False hbf data_trunc_norm_kern $data_normalize $trainable_S
