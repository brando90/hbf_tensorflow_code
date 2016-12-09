# This means you derive your docker image from the tensorflow docker image
# FROM gcr.io/tensorflow/tensorflow:latest-devel-gpu
FROM gcr.io/tensorflow/tensorflow
#FROM python
FROM ubuntu

RUN mkdir ~/my_tf_proj/
# mounts my tensorflow lib/proj from host to the container
VOLUME /Users/brandomiranda/Documents/MIT/MEng/hbf_tensorflow_code/my_tf_proj ~/my_tf_proj

#
RUN apt-get update

#
apt-get install vim

#
RUN apt-get install -qy python3
RUN apt-get install -qy python3-pip
RUN pip3 install --upgrade pip

#RUN apt-get install -y python python-dev python-distribute python-pip

# have the dependecies for my tensorflow library
RUN pip3 install numpy
RUN pip3 install keras
RUN pip3 install namespaces
RUN pip3 install pdb

# install my library (only when the a container is spun)
CMD python ~/my_tf_proj/setup.py develop
