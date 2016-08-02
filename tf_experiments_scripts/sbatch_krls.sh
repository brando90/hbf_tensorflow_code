#!/bin/bash
#SBATCH --job-name=KRLS
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --array=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

task_name=task_f2d_2x2_1_cosx1_plus_x2_depth_2
#task_name=task_f2d_2x2_1_cosx1x2_depth_2
#task_name=task_f_2d_task2_xsinglog1_x_depth2
python krls_collect_data.py $task_name krls_2x2_1_cosx1_plus_x2_depth_2_12_48_96_246_360 100 100 12,48,96,246,360
#python krls_collect_data.py $task_name krls_2x2_1_cosx1x2_depth_2_12_48_96_246_360 15 15 12,48,96,246,360
#python krls_collect_data.py $task_name krls_xsinglog1_x_depth2_12_48_96_246_360 15 15 12,48,96,246,360
