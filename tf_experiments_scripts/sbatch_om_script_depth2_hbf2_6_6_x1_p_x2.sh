#!/bin/bash
#SBATCH --job-name=HBF2_6_6
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --array=1-1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

python main_nn.py $SLURM_JOBID $SLURM_ARRAY_TASK_ID om_2x2_1_cosx1_plus_x2_depth2 task_1_August_HBF2_depth_2_1000 HBF2_6_6_multiple_1000 True 6,6 multiple_S task_f2d_2x2_1_cosx1_plus_x2_depth2 False False hbf data_xavier_kern
