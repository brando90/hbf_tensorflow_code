#!/bin/python
#SBATCH --job-name=Python

#import namespaces as ns
import os

path = '/home/brando90/envs/tensorflow/lib/python2.7/site-packages' 
ls = os.listdir(path)
print(ls)
print('namespaces' in ls)

import namespaces as ns

print ns
