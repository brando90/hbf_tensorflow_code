#!/bin/sh

python /hbf_tensorflow_code/my_tf_proj/setup.py develop

#echo $@
python3 /hbf_tensorflow_code/docker_files/tf_cpu/batch_main.py $@

