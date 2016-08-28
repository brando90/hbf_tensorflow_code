#!/bin/python
#SBATCH --job-name=Python

# import sys
# env_path = '/home/brando90/envs/tensorflow/lib/python2.7/site-packages'
# if not env_path in sys.path:
#     sys.path.append(env_path)
# else:
#     print('env_path %s present in path'%env_path)

import sys
env_path = '/home/brando90/envs/tensorflow/lib/python2.7/site-packages'
if not env_path in sys.path:
    sys.path.append(env_path)
else:
    print('env_path %s present in path'%env_path)

#import namespaces as ns
import os
import sys

#location it should be installed
path = env_path
ls = os.listdir(path)
print(ls)
print('namespaces' in ls) #does exist

#when using sbatch these lines fail
print sys.path
import namespaces as ns

print ns
