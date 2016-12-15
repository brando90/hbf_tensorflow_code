# This means you derive your docker image from the tensorflow docker image
# FROM gcr.io/tensorflow/tensorflow:latest-devel-gpu
FROM gcr.io/tensorflow/tensorflow
FROM ubuntu

#RUN mkdir /my_tf_proj/
# to mount my_tf library, use -v flag since,
# The host directory for a volume cannot be specified from a Dockerfile, in order to preserve portability
# (the host directory may not be available on all systems). When this form of the -v
# argument is used any files in the image under the directory are not copied into the volume.

#
RUN apt-get update
RUN apt-get install vim

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
# bash script that configures running container
ENTRYPOINT ["sh", "container_setup.py"]
