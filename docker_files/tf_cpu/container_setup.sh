#!/bin/sh

#python3 /hbf_tensorflow_code/my_tf_proj/setup.py develop
pip3 install /hbf_tensorflow_code/my_tf_proj

#pip3 list
#echo $@
#pip list
#python3 /hbf_tensorflow_code/docker_files/tf_cpu/batch_main.py $@
$@
