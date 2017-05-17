#!/bin/bash
#SBATCH --mem=50000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brando90@mit.edu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00

nvidia-docker run --rm -v /raid/poggio/home/brando90/home_simulation_research:/home_simulation_research tf_gpu_py3_4_2 python3 /home_simulation_research/hbf_tensorflow_code/dgx1/cont_prod/dgx1_expt_32D_cont_prod_80000_80K/bt2_elu.py -sj sj
