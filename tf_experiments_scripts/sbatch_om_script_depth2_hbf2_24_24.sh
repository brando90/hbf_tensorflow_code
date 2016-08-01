#!/bin/bash
#SBATCH --job-name=HBF2_48_48
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --array=1-2020
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

python main_nn.py $SLURM_JOBID $SLURM_ARRAY_TASK_ID om_xsinlog1_x_depth2_hbf task_29_july_HBF1_depth_2_2020 HBF2_48_48_multiple_1000 True 48,48 multiple_S task_f_2d_task2_xsinglog1_x_depth2 False False hbf data_trunc_norm_kern
