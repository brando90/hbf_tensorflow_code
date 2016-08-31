#!/usr/bin python
#SBATCH --job-name=Python

import sys
import os

print(os.system('pwd'))
print(sys.path)

sys.path = ['/om/user/brando90/MEng/hbf_tensorflow_code/tf_experiments_scripts'] + sys.path

import main_nn

