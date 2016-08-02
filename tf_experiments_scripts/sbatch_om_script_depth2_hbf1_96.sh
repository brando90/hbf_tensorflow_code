#!/bin/bash
#SBATCH --job-name=HBF1_96
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --array=1-1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

python main_nn.py $SLURM_JOBID $SLURM_ARRAY_TASK_ID om_xsinlog1_x_depth2_hbf task_30_july_HBF1_depth_2_1000 HBF1_96_multiple_1000 True 96 multiple_S task_f_2d_task2_xsinglog1_x_depth2 False False hbf kern_init
