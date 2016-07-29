#!/bin/bash
#SBATCH --job-name=KRLS
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --array=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

python krls_collect_data.py task_f_2d_task2_xsinglog1_x_depth2 krls_f_2d_task2_xsinglog1_x_depth2 100 100 6,12,24,48,96
