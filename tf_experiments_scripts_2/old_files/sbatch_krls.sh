#!/bin/bash
#SBATCH --job-name=KRLS
#SBATCH --nodes=1
#SBATCH --mem=70000
#SBATCH --time=6-23
#SBATCH --array=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

#task_name=task_f2d_2x2_1_cosx1_plus_x2_depth
task_name=task_MNIST_flat_auto_encoder
python krls_collect_data.py $task_name krls_xsinglog1_x_depth2_12_48_96_246_360 30 30 12,48,96,246,360
