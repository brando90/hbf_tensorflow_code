# This means you derive your docker image from the tensorflow docker image
# FROM gcr.io/tensorflow/tensorflow:latest-devel-gpu
FROM gcr.io/tensorflow/tensorflow
FROM python
FROM ubuntu

RUN mkdir ~/my_tf_proj/
# mounts my tensorflow lib/proj from host to the container
VOLUME /Users/brandomiranda/Documents/MIT/MEng/hbf_tensorflow_code/my_tf_proj ~/my_tf_proj

#
RUN apt-get install -y python-pip
RUN apt-get install -y python python-dev python-distribute python-pip
RUN apt-get install -y python3-pip
RUN apt-get install -qy python3
RUN apt-get install -qy python3-pip
RUN apt-get install -y python python-pip wget

# have the dependecies for my tensorflow library
RUN pip install numpy
RUN pip install keras
RUN pip install namespaces
RUN pip install pdb

# install my library (only when the a container is spun)
CMD python ~/my_tf_proj/setup.py develop
