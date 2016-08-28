#!/bin/python
#SBATCH --job-name=Python

#import namespaces as ns
import os
import sys

#location it should be installed
path = '/home/brando90/envs/tensorflow/lib/python2.7/site-packages'
ls = os.listdir(path)
print(ls)
print('namespaces' in ls) #does exist

#when using sbatch these lines fail
print sys.path
import namespaces as ns

print ns
