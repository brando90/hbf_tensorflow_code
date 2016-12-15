#!/bin/sh

python /hbf_tensorflow_code/my_tf_proj/setup.py develop

#echo $@
python batch_main.py $@
