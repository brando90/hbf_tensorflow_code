#!/bin/bash
#SBATCH --mem=10000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brando90@mit.edu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00

nvidia-docker run --rm -v /raid/poggio/home/brando90/home_simulation_research:/home_simulation_research tf_gpu_py3.4 python3 /home_simulation_research/hbf_tensorflow_code/dgx1_expts_scripts_8D_continuous_product_func/batch_main_NN7x3.py -sj sj
