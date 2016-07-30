#!/bin/bash
#SBATCH --job-name=HBF2_6_6
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --array=1-5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

#python main_nn.py $SLURM_JOBID $SLURM_ARRAY_TASK_ID om_xsinlog1_x_depth2 task_29_july_NN2_10 NN1_12_12_multiple_10 True 12,12 multiple_S task_f_2d_task2_xsinglog1_x_depth2 False
python main_nn.py $SLURM_JOBID $SLURM_ARRAY_TASK_ID om_xsinlog1_x_depth2 task_27_july_NN2_depth_2_1000 NN2_3_3_multiple_10 True 3,3 multiple_S task_f_2d_task2_xsinglog1_x_depth2 False
