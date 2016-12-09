# This means you derive your docker image from the tensorflow docker image
# FROM gcr.io/tensorflow/tensorflow:latest-devel-gpu
FROM gcr.io/tensorflow/tensorflow
FROM python
FROM ubuntu

# mounts my tensorflow lib/proj from host to the container
VOLUME ../../my_tf_proj ~/my_tf_proj

# have the dependecies for my tensorflow library
RUN pip install numpy
RUN pip install keras
RUN pip install namespaces
RUN pip install pdb

# install my library (only when the a container is spun)
CMD python ~/my_tf_proj
