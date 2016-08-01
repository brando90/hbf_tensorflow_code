#!/bin/bash
#SBATCH --job-name=HBF2_24_24
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --array=1-2020
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

python main_nn.py $SLURM_JOBID $SLURM_ARRAY_TASK_ID om_xsinlog1_x_depth2_hbf task_29_july_HBF2_depth_2_2020 HBF2_24_24_multiple_2020 True 24,24 multiple_S task_f_2d_task2_xsinglog1_x_depth2 False False hbf data_trunc_norm_kern