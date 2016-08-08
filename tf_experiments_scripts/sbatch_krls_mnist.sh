#!/bin/bash
#SBATCH --job-name=KRLS
#SBATCH --nodes=1
#SBATCH --mem=70000
#SBATCH --time=6-23
#SBATCH --array=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com


#python krls_collect_data.py task_MNIST_flat_auto_encoder om_krls krls_MNIST_flat_2_2_units_6_12_24_48_96 2 2 6,12,24,48,96
python krls_collect_data.py task_MNIST_flat_auto_encoder om_krls krls_MNIST_flat_150_150_units_6_12_24_48_96 150 150 6,12,24,48,96
